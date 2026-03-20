#ifndef SOLVER_H
#define SOLVER_H
#pragma once

#include "discretization.h"
#include "mass_matrix.h"
#include "stiffness_matrix.h"

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

// ── Traits ──────────────────────────────────────────────────────
template <typename T, typename = void>
struct has_final_time : std::false_type {};
template <typename T>
struct has_final_time<T, std::void_t<decltype(std::declval<T>().final_time)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_initial_data : std::false_type {};
template <typename T>
struct has_initial_data<T, std::void_t<decltype(std::declval<T>().initial_data)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_derivative_function : std::false_type {};
template <typename T>
struct has_derivative_function<T, std::void_t<decltype(std::declval<T>().derivative_function)>>
    : std::true_type {};

// ── Solver class ────────────────────────────────────────────────
template <int dim, typename SolutionSet>
class Solver
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  Solver(SolutionSet                     sol,
         const Discretization<dim>      &discretization,
         StiffnessMatrixOperator<dim>   &stiffness,
         TrilinosWrappers::SparseMatrix &system_matrix,
         const int                       pde_type,
         const double                    cfl)
    : sol(std::move(sol))
    , discretization(discretization)
    , stiffness(stiffness)
    , pde_type(pde_type)
    , cfl(cfl)
  {
    // copy so we own the matrix — avoids dangling reference
    this->system_matrix.copy_from(system_matrix);
    solver_direct.initialize(this->system_matrix);
    discretization.initialize_dof_vector(solution);
  }

  // ── Main entry point ────────────────────────────────────────
  void solve()
  {
    if (pde_type==2)
      solve_poisson();
    else if (pde_type==1)
      solve_heat();
    else
      solve_wave();
  }

  Function<dim> * get_analytical_solution() const
  {
      return sol.analytical_solution.get();
  }

  const VectorType &          get_solution()            const { return solution; }

  double get_final_time() const
  {
    if constexpr (has_final_time<SolutionSet>::value)
      return sol.final_time;
    else
      return 0.0;
  }

private:
  SolutionSet                             sol;
  const Discretization<dim>             &discretization;
  StiffnessMatrixOperator<dim>          &stiffness;
  int                                    pde_type;
  double                                 cfl;
  TrilinosWrappers::SparseMatrix         system_matrix;
  TrilinosWrappers::SolverDirect         solver_direct;
  VectorType                             solution;

  // ── Solve: system_matrix * solution_out = rhs(t) - K*previous_u
  void solve_at_time(const double      t,
                     const VectorType &previous_u,
                     VectorType       &solution_out)  // must be non-const ref
  {
    VectorType rhs, tmp;
    discretization.initialize_dof_vector(rhs);
    discretization.initialize_dof_vector(tmp);

    rhs = stiffness.get_rhs_matrix(t);

    if (pde_type != 2)  // Heat/Wave: subtract K*u term
    {
      stiffness.get_stiffness_matrix().vmult(tmp, previous_u);
      rhs.add(-1.0, tmp);
    }

    solver_direct.solve(solution_out, rhs);
  }

  // ── Poisson ─────────────────────────────────────────────────
  void solve_poisson()
  {
    VectorType dummy;
    discretization.initialize_dof_vector(dummy);
    solve_at_time(0.0, dummy, solution);
    solution.update_ghost_values();
  }

  // ── Heat ────────────────────────────────────────────────────
  void solve_heat()
  {
    if constexpr (has_initial_data<SolutionSet>::value &&
                  has_final_time<SolutionSet>::value)
    {
      VectorType old_u;
      discretization.initialize_dof_vector(old_u);
      VectorTools::interpolate(discretization.get_dof_handler(),
                               *sol.initial_data, old_u);

      double t        = sol.initial_time;
      const double dt = cfl * std::pow(discretization.get_dx(), 2);

      while (t < sol.final_time - 1e-10)
      {
        const double step = std::min(dt, sol.final_time - t);

        VectorType k1, k2, k3, k4, tmp;
        discretization.initialize_dof_vector(k1);
        discretization.initialize_dof_vector(k2);
        discretization.initialize_dof_vector(k3);
        discretization.initialize_dof_vector(k4);
        discretization.initialize_dof_vector(tmp);

        // k1
        solve_at_time(t, old_u, k1);

        // k2
        tmp = old_u; tmp.add(step/2.0, k1);
        solve_at_time(t + step/2.0, tmp, k2);

        // k3
        tmp = old_u; tmp.add(step/2.0, k2);
        solve_at_time(t + step/2.0, tmp, k3);

        // k4
        tmp = old_u; tmp.add(step, k3);
        solve_at_time(t + step, tmp, k4);

        // combine
        solution = old_u;
        solution.add(step/6.0, k1);
        solution.add(step/3.0, k2);
        solution.add(step/3.0, k3);
        solution.add(step/6.0, k4);

        old_u = solution;
        t += step;
      }
      solution.update_ghost_values();
    }
  }

  // ── Wave ─────────────────────────────────────────────────────
  void solve_wave()
  {
    if constexpr (has_initial_data<SolutionSet>::value       &&
                  has_final_time<SolutionSet>::value          &&
                  has_derivative_function<SolutionSet>::value)
    {
      VectorType old_u, old_v;
      discretization.initialize_dof_vector(old_u);
      discretization.initialize_dof_vector(old_v);
      VectorTools::interpolate(discretization.get_dof_handler(),
                               *sol.initial_data,        old_u);
      VectorTools::interpolate(discretization.get_dof_handler(),
                               *sol.derivative_function, old_v);

      double t        = sol.initial_time;
      const double dt = cfl * discretization.get_dx();

      while (t < sol.final_time - 1e-10)
      {
        const double step = std::min(dt, sol.final_time - t);

        VectorType ku1, ku2, ku3, ku4;
        VectorType kv1, kv2, kv3, kv4;
        VectorType tmp_u, tmp_v;

        discretization.initialize_dof_vector(ku1);
        discretization.initialize_dof_vector(ku2);
        discretization.initialize_dof_vector(ku3);
        discretization.initialize_dof_vector(ku4);
        discretization.initialize_dof_vector(kv1);
        discretization.initialize_dof_vector(kv2);
        discretization.initialize_dof_vector(kv3);
        discretization.initialize_dof_vector(kv4);
        discretization.initialize_dof_vector(tmp_u);
        discretization.initialize_dof_vector(tmp_v);

        // k1: ku1 = v_n,  kv1 = M^{-1}(f(t) - K*u_n)
        ku1 = old_v;
        solve_at_time(t, old_u, kv1);

        // k2
        tmp_u = old_u; tmp_u.add(step/2.0, ku1);
        tmp_v = old_v; tmp_v.add(step/2.0, kv1);
        ku2 = tmp_v;
        solve_at_time(t + step/2.0, tmp_u, kv2);

        // k3
        tmp_u = old_u; tmp_u.add(step/2.0, ku2);
        tmp_v = old_v; tmp_v.add(step/2.0, kv2);
        ku3 = tmp_v;
        solve_at_time(t + step/2.0, tmp_u, kv3);

        // k4
        tmp_u = old_u; tmp_u.add(step, ku3);
        tmp_v = old_v; tmp_v.add(step, kv3);
        ku4 = tmp_v;
        solve_at_time(t + step, tmp_u, kv4);

        // combine u
        solution = old_u;
        solution.add(step/6.0, ku1);
        solution.add(step/3.0, ku2);
        solution.add(step/3.0, ku3);
        solution.add(step/6.0, ku4);

        // combine v
        old_v.add(step/6.0, kv1);
        old_v.add(step/3.0, kv2);
        old_v.add(step/3.0, kv3);
        old_v.add(step/6.0, kv4);

        old_u = solution;
        t += step;
      }
      solution.update_ghost_values();
    }
  }
};

#endif