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
         std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &system_matrix_in,
         const int                       pde_type,
         const double                    cfl)
    : sol(std::move(sol))
    , discretization(discretization)
    , stiffness(stiffness)
    , pde_type(pde_type)
    , cfl(cfl)
    , dof_handlers(discretization.get_dof_handlers())
  {        
    solution.reinit(discretization.get_level_sets().size());
    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      system_matrix.push_back(system_matrix_in[domain_idx]);
      solver_direct.push_back(std::make_shared<TrilinosWrappers::SolverDirect>());
      solver_direct[domain_idx]->initialize(*system_matrix[domain_idx]);
      discretization.initialize_dof_vector(solution.block(domain_idx), domain_idx);
    }
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
  std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> system_matrix;
  std::vector<std::shared_ptr<TrilinosWrappers::SolverDirect>> solver_direct;

  // n-block solution vector
  BlockVectorType solution;

  // ── Helper: initialise a zero n-block vector ─────────────────
  BlockVectorType make_block_vector() const
  {
    BlockVectorType bv;
    bv.reinit(discretization.get_level_sets().size());
    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      discretization.initialize_dof_vector(bv.block(domain_idx), domain_idx);
    }
    return bv;
  }
  void solve_at_time(const double           t,
                     const BlockVectorType &previous_u,
                     BlockVectorType       &solution_out)
  {
    BlockVectorType rhs = make_block_vector();

    stiffness.get_rhs_matrix(rhs, t, previous_u);

    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      solver_direct[domain_idx]->solve(solution_out.block(domain_idx), rhs.block(domain_idx));
    }
  }

  // ── Poisson ─────────────────────────────────────────────────
  void solve_poisson()
  {
    BlockVectorType dummy = make_block_vector();
    solve_at_time(0.0, dummy, solution);
    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      solution.block(domain_idx).update_ghost_values();
    }
    actual_final_time = 0.0;
  }

  // ── Heat ────────────────────────────────────────────────────
  void solve_heat()
  {
    if constexpr (has_initial_data<SolutionSet>::value &&
                  has_final_time<SolutionSet>::value)
    {
      // Interpolate initial data into all blocks
      BlockVectorType old_u = make_block_vector();

      for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
        VectorTools::interpolate(*dof_handlers[domain_idx],
                               *sol.initial_data[domain_idx].get(), old_u.block(domain_idx));                          
      }

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
      for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
        solution.block(domain_idx).update_ghost_values();
      }
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

      for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
        VectorTools::interpolate(*dof_handlers[domain_idx],
                               *sol.initial_data[domain_idx].get(), old_u.block(domain_idx));
      VectorTools::interpolate(*dof_handlers[domain_idx],
                               *sol.derivative_initial_data[domain_idx].get(), old_v.block(domain_idx));                          
      }

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
      for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
        solution.block(domain_idx).update_ghost_values();
      }
    }
  }
};

#endif