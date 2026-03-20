#ifndef STIFFNESS_MATRIX_H
#define STIFFNESS_MATRIX_H
#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include "discretization.h"

using namespace dealii;

template <unsigned int dim, typename Number = double>
class StiffnessMatrixOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number>    &discretization, const double &gps, const double &np, Function<dim> *rhs,
                    Function<dim> *bv, double speed)
    : discretization(discretization)
    , ghost_parameter_S(gps)
    , nitsche_parameter(np)
    , rhs_function(rhs)
    , boundary_condition(bv)
    , quadrature_1D(discretization.get_quadrature_1D())
    , face_quadrature(discretization.get_face_quadrature())
    , constraints(discretization.get_affine_constraints())
    , mesh_classifier(discretization.get_mesh_classifier())
    , fe_collection(discretization.get_fe_collection())
    , level_set(discretization.get_level_set())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handler(discretization.get_dof_handler())    
  {}

  const TrilinosWrappers::SparseMatrix &
  get_stiffness_matrix() const
  {
    compute_sparse_matrix();

    return sparse_matrix;
  }

  const VectorType & get_rhs_matrix(const double evaluating_time) const
  {
    compute_rhs(evaluating_time);

    return rhs;
  }

private:
  const Discretization<dim, Number>    &discretization;
  double ghost_parameter_S;
  double nitsche_parameter;
  Function<2> *rhs_function;
  Function<2> *boundary_condition;
  const QGauss<1> &quadrature_1D;
  const QGauss<dim - 1> &face_quadrature;
  const AffineConstraints<Number> &constraints;
  const NonMatching::MeshClassifier<dim> &mesh_classifier;
  const hp::FECollection<dim> &fe_collection;
  const VectorType            &level_set;
  const DoFHandler<dim>       &level_set_dof_handler;
  const DoFHandler<dim>       &dof_handler;

  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;
  mutable VectorType rhs;

  void
  compute_sparse_matrix() const
  {    
    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifier.location_to_level_set(cell);

      const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifier.location_to_level_set(cell->neighbor(face_index));

      if (cell_location == NonMatching::LocationToLevelSet::intersected &&
          neighbor_location != NonMatching::LocationToLevelSet::outside)
        return true;

      if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
          cell_location != NonMatching::LocationToLevelSet::outside)
        return true;

      return false;
    };
    

    sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                            dof_handler.get_communicator());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         sparsity_pattern,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_ghost_penalty);
    sparsity_pattern.compress();
    sparse_matrix.reinit(sparsity_pattern);

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                               update_hessians |
                                                 update_JxW_values |
                                                 update_normal_vectors);
    
    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_hessians | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients | update_hessians | 
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);
                                                      
    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() |
           IteratorFilters::ActiveFEIndexEqualTo(Discretization<dim>::ActiveFEIndex::lagrange))
      {
        local_stiffness = 0;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                }
            }

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                // const Point<dim> &point =
                //   surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          (-normal * surface_fe_values->shape_grad(i, q) *
                             surface_fe_values->shape_value(j, q) +
                           -normal * surface_fe_values->shape_grad(j, q) *
                             surface_fe_values->shape_value(i, q) +
                           nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) *
                             surface_fe_values->shape_value(j, q)) *
                          surface_fe_values->JxW(q);
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        sparse_matrix.add(local_dof_indices, local_stiffness);  

        for (const unsigned int f : cell->face_indices())
          if (face_has_ghost_penalty(cell, f))
            {
              const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;

              fe_interface_values.reinit(cell,
                                         f,
                                         invalid_subface,
                                         cell->neighbor(f),
                                         cell->neighbor_of_neighbor(f),
                                         invalid_subface);

              const unsigned int n_interface_dofs =
                fe_interface_values.n_current_interface_dofs();
              FullMatrix<double> local_stabilization(n_interface_dofs,
                                                     n_interface_dofs);
              for (unsigned int q = 0;
                   q < fe_interface_values.n_quadrature_points;
                   ++q)
                {
                  const Tensor<1, dim> normal =
                    fe_interface_values.normal(q);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    for (unsigned int j = 0; j < n_interface_dofs; ++j)
                      {
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter_S  *  cell_side_length * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter_S  *  std::pow(cell_side_length,3) * normal *
                          fe_interface_values.jump_in_shape_hessians(i, q) * normal *
                          normal *
                          fe_interface_values.jump_in_shape_hessians(j, q) * normal *
                          fe_interface_values.JxW(q);       
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              sparse_matrix.add(local_interface_dof_indices,
                                   local_stabilization);
            }
      }

    sparse_matrix.compress(VectorOperation::add);    
    for (auto &entry : sparse_matrix)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0; 
  }

  void
  compute_rhs(const double evaluating_time) const
  {
    discretization.initialize_dof_vector(rhs); 
    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    Vector<double> local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_JxW_values | 
                                 update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    rhs_function->set_time(evaluating_time);
    boundary_condition->set_time(evaluating_time);                         

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() |
           IteratorFilters::ActiveFEIndexEqualTo(Discretization<dim>::ActiveFEIndex::lagrange))
      if(cell->is_locally_owned())
      {
        local_rhs = 0;
        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        // ============================================================
        // VOLUME SOURCE TERM: ∫ f φᵢ dx
        // ============================================================
        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          {
            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                
                // Evaluate f at θ*t^n + (1-θ)*t^{n-1}
                const double f_value = rhs_function->value(point);

                for (const unsigned int i : inside_fe_values->dof_indices())
                  {
                    local_rhs(i) += f_value *
                                    inside_fe_values->shape_value(i, q) *
                                    inside_fe_values->JxW(q);
                  }
              }
          }

        // ============================================================
        // BOUNDARY TERMS: Nitsche RHS
        // ============================================================
        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);

                // Evaluate g at θ*t^n + (1-θ)*t^{n-1}
                const double g_value = boundary_condition->value(point);

                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    local_rhs(i) +=
                      g_value *
                      (nitsche_parameter / cell_side_length *
                         surface_fe_values->shape_value(i, q) -
                       normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        rhs.add(local_dof_indices, local_rhs);
      }

    rhs.compress(VectorOperation::add);
  }
};
#endif
