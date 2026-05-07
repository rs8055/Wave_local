#ifndef SOLVERComposite_H
#define SOLVERComposite_H
#pragma once

#include "discretization.h"
#include "mass_matrix.h"
#include "stiffness_matrix.h"

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

// ── Solver class ────────────────────────────────────────────────
template <int dim, typename SolutionSet>
class SolverComposite
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  SolverComposite(SolutionSet                     sol,
         const Discretization<dim>      &discretization,
         StiffnessMatrixOperator<dim>   &stiffness,
         TrilinosWrappers::SparseMatrix &system_matrix_0,
         TrilinosWrappers::SparseMatrix &system_matrix_1,
         const int                       pde_type,
         const double                    cfl)
    : sol(std::move(sol))
    , discretization(discretization)
    , stiffness(stiffness)
    , pde_type(pde_type)
    , cfl(cfl)
    , dof_handlers(discretization.get_dof_handlers())
  {
    system_matrix[0].copy_from(system_matrix_0);
    system_matrix[1].copy_from(system_matrix_1);

    solver_direct[0].initialize(system_matrix[0]);
    solver_direct[1].initialize(system_matrix[1]);

    // solution is a 2-block vector; each block lives on the same DoF layout
    VectorType tmp;
    discretization.initialize_dof_vector(tmp, 0);
    solution.reinit(2);
    solution.block(0) = tmp;
    solution.block(1) = tmp;
    solution.collect_sizes();
  }

  // ── Main entry point ────────────────────────────────────────
  void solve()
  {
    if      (pde_type == 0) solve_poisson();
    else if (pde_type == 1) solve_heat();
    else if (pde_type == 2) solve_wave();
  }

  Function<dim> *get_analytical_solution() const
  {
    return sol.analytical_solution.get();
  }

  // Individual block accessors
  const VectorType &get_solution(const unsigned int block = 0) const
  {
    return solution.block(block);
  }

  // Full block-vector accessor
  const BlockVectorType &get_block_solution() const { return solution; }

  double get_final_time() const
  {
    return actual_final_time;
  }

private:
  SolutionSet                             sol;
  const Discretization<dim>             &discretization;
  StiffnessMatrixOperator<dim>          &stiffness;
  int                                    pde_type;
  double                                 cfl;
  double                                 actual_final_time = 0.0;
  const std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;

  // One matrix + solver per block
  TrilinosWrappers::SparseMatrix  system_matrix[2];
  TrilinosWrappers::SolverDirect  solver_direct[2];

  // Two-block solution vector
  BlockVectorType solution;

  // ── Helper: initialise a zero 2-block vector ─────────────────
  BlockVectorType make_block_vector() const
  {
    VectorType tmp;
    discretization.initialize_dof_vector(tmp, 0);
    BlockVectorType bv;
    bv.reinit(2);
    bv.block(0) = tmp;
    bv.block(1) = tmp;
    bv.collect_sizes();
    return bv;
  }
  void solve_at_time(const double           t,
                     const BlockVectorType &previous_u,
                     BlockVectorType       &solution_out)
  {
    BlockVectorType rhs = make_block_vector();

    // Stiffness operator fills both blocks (coupling lives here)
    stiffness.get_rhs_matrix(rhs, t, previous_u);

    // Solve each block independently with its own system matrix
    solver_direct[0].solve(solution_out.block(0), rhs.block(0));
    solver_direct[1].solve(solution_out.block(1), rhs.block(1));
  }

  // ── Poisson ─────────────────────────────────────────────────
  void solve_poisson()
  {
    BlockVectorType dummy = make_block_vector();
    solve_at_time(0.0, dummy, solution);
    solution.block(0).update_ghost_values();
    solution.block(1).update_ghost_values();
    actual_final_time = 0.0;
  }

  // ── Heat ────────────────────────────────────────────────────
  void solve_heat()
  {
    if constexpr (has_initial_data<SolutionSet>::value &&
                  has_final_time<SolutionSet>::value)
    {
      // Interpolate initial data into both blocks
      BlockVectorType old_u = make_block_vector();
      VectorTools::interpolate(*dof_handlers[0],
                               *sol.initial_data, old_u.block(0));
      const Function<dim> *required_initial_data = (sol.initial_data_other != nullptr)
                                                            ? sol.initial_data_other.get()
                                                            : sol.initial_data.get();                                  
      VectorTools::interpolate(*dof_handlers[0],
                               *required_initial_data, old_u.block(1));

      double       t        = sol.initial_time;
      const double dt       = cfl * std::pow(discretization.get_dx(), 2);

      while (t < sol.final_time - 1e-10)
      {
        const double step = std::min(dt, sol.final_time - t);

        BlockVectorType k1 = make_block_vector();
        BlockVectorType k2 = make_block_vector();
        BlockVectorType k3 = make_block_vector();
        BlockVectorType k4 = make_block_vector();
        BlockVectorType tmp;

        // k1
        solve_at_time(t, old_u, k1);

        // k2
        tmp = old_u; tmp.add(step / 2.0, k1);
        solve_at_time(t + step / 2.0, tmp, k2);

        // k3
        tmp = old_u; tmp.add(step / 2.0, k2);
        solve_at_time(t + step / 2.0, tmp, k3);

        // k4
        tmp = old_u; tmp.add(step, k3);
        solve_at_time(t + step, tmp, k4);

        // Combine: u_{n+1} = u_n + (h/6)(k1 + 2k2 + 2k3 + k4)
        solution = old_u;
        solution.add(step / 6.0, k1);
        solution.add(step / 3.0, k2);
        solution.add(step / 3.0, k3);
        solution.add(step / 6.0, k4);

        old_u = solution;
        t += step;
      }
      actual_final_time = t;
      solution.block(0).update_ghost_values();
      solution.block(1).update_ghost_values();
    }
  }

  // ── Wave ─────────────────────────────────────────────────────
  void solve_wave()
  {
    if constexpr (has_initial_data<SolutionSet>::value            &&
                  has_final_time<SolutionSet>::value               &&
                  has_derivative_initial_data<SolutionSet>::value)
    {
      // Displacement and velocity block-vectors
      BlockVectorType old_u = make_block_vector();
      BlockVectorType old_v = make_block_vector();

      VectorTools::interpolate(*dof_handlers[0],
                               *sol.initial_data,            old_u.block(0));
      VectorTools::interpolate(*dof_handlers[0],
                               *sol.derivative_initial_data, old_v.block(0));
      
      const Function<dim> *required_initial_data = (sol.initial_data_other != nullptr)
                                                            ? sol.initial_data_other.get()
                                                            : sol.initial_data.get();
      const Function<dim> *required_derivative_initial_data = (sol.derivative_initial_data_other != nullptr)
                                                            ? sol.derivative_initial_data_other.get()
                                                            : sol.derivative_initial_data.get();                                  
      VectorTools::interpolate(*dof_handlers[0],
                               *required_initial_data, old_u.block(1));
      VectorTools::interpolate(*dof_handlers[0],
                               *required_derivative_initial_data, old_v.block(1)); 

      double       t  = sol.initial_time;
      const double dt = cfl * discretization.get_dx();

      while (t < sol.final_time - 1e-10)
      {
        const double step = std::min(dt, sol.final_time - t);

        BlockVectorType ku1 = make_block_vector(), kv1 = make_block_vector();
        BlockVectorType ku2 = make_block_vector(), kv2 = make_block_vector();
        BlockVectorType ku3 = make_block_vector(), kv3 = make_block_vector();
        BlockVectorType ku4 = make_block_vector(), kv4 = make_block_vector();
        BlockVectorType tmp_u, tmp_v;

        // k1: u' = v,  v' = M^{-1}(f - K*u)
        ku1 = old_v;
        solve_at_time(t, old_u, kv1);

        // k2
        tmp_u = old_u; tmp_u.add(step / 2.0, ku1);
        tmp_v = old_v; tmp_v.add(step / 2.0, kv1);
        ku2 = tmp_v;
        solve_at_time(t + step / 2.0, tmp_u, kv2);

        // k3
        tmp_u = old_u; tmp_u.add(step / 2.0, ku2);
        tmp_v = old_v; tmp_v.add(step / 2.0, kv2);
        ku3 = tmp_v;
        solve_at_time(t + step / 2.0, tmp_u, kv3);

        // k4
        tmp_u = old_u; tmp_u.add(step, ku3);
        tmp_v = old_v; tmp_v.add(step, kv3);
        ku4 = tmp_v;
        solve_at_time(t + step, tmp_u, kv4);

        // Combine displacement
        solution = old_u;
        solution.add(step / 6.0, ku1);
        solution.add(step / 3.0, ku2);
        solution.add(step / 3.0, ku3);
        solution.add(step / 6.0, ku4);

        // Combine velocity (in-place on old_v)
        old_v.add(step / 6.0, kv1);
        old_v.add(step / 3.0, kv2);
        old_v.add(step / 3.0, kv3);
        old_v.add(step / 6.0, kv4);

        old_u = solution;
        t += step;
      }
      actual_final_time = t;
      solution.block(0).update_ghost_values();
      solution.block(1).update_ghost_values();
    }
  }
};

#endif