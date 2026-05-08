#ifndef MASS_MATRIX_H
#define MASS_MATRIX_H
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
class MassMatrixOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  MassMatrixOperator(const Discretization<dim, Number>    &discretization, const double &gpm)
    : discretization(discretization)
    , ghost_parameter_M(gpm)
    , quadrature_1D(discretization.get_quadrature_1D())
    , face_quadrature(discretization.get_face_quadrature())
    , constraints(discretization.get_affine_constraints())
    , mesh_classifiers(discretization.get_mesh_classifiers())
    , fe_collection(discretization.get_fe_collection())
    , level_sets(discretization.get_level_sets())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handlers(discretization.get_dof_handlers())
  {}

  const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
  get_mass_matrix() const
  {
      block_sparse_matrix.clear();
      block_sparse_matrix.resize(discretization.get_level_sets().size());
      for (size_t i = 0; i < discretization.get_level_sets().size(); ++i)
      {
          block_sparse_matrix[i] = std::make_shared<TrilinosWrappers::SparseMatrix>();
          compute_sparse_matrix(i, *block_sparse_matrix[i]);
      }
      return block_sparse_matrix;
  }


private:
  const Discretization<dim, Number>    &discretization;
  double ghost_parameter_M;
  const QGauss<1> &quadrature_1D;
  const QGauss<dim - 1> &face_quadrature;
  const std::vector<AffineConstraints<Number>> &constraints;
  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> & mesh_classifiers;
  const hp::FECollection<dim> &fe_collection;
  const std::vector<VectorType> &level_sets;
  const DoFHandler<dim>       &level_set_dof_handler;
  const std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;
  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> block_sparse_matrix;


  void
  compute_sparse_matrix(size_t domain_idx, TrilinosWrappers::SparseMatrix &mat) const
  { 
    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell);

      const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell->neighbor(face_index));

      if (cell_location == NonMatching::LocationToLevelSet::intersected &&
          neighbor_location != NonMatching::LocationToLevelSet::outside)
        return true;

      if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
          cell_location != NonMatching::LocationToLevelSet::outside)
        return true;

      return false;
    };
    

    sparsity_pattern.reinit(dof_handlers[domain_idx]->locally_owned_dofs(),
                            dof_handlers[domain_idx]->get_communicator());

    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(*dof_handlers[domain_idx],
                                         sparsity_pattern,
                                         constraints[domain_idx],
                                         keep_constrained_dofs,
                                         numbers::invalid_subdomain_id);
    sparsity_pattern.compress();
    mat.reinit(sparsity_pattern);

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_mass(n_dofs_per_cell, n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                   update_hessians | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients | update_hessians | 
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;                              

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                               update_hessians |
                                                 update_JxW_values |
                                                 update_normal_vectors);                                                  
                                                      
    for (const auto &cell :
         dof_handlers[domain_idx]->active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() |
           IteratorFilters::ActiveFEIndexEqualTo(Discretization<dim>::ActiveFEIndex::lagrange))
      // if (mesh_classifier.location_to_level_set(cell) !=
      if (mesh_classifiers[domain_idx]->location_to_level_set(cell) != NonMatching::LocationToLevelSet::outside)
        {
        local_mass = 0;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const auto &fe_values = non_matching_fe_values.get_inside_fe_values() ;

        if (fe_values)
          for (const unsigned int q :
               fe_values->quadrature_point_indices())
            {
              for (const unsigned int i : fe_values->dof_indices())
                {
                  for (const unsigned int j : fe_values->dof_indices())
                    {
                      local_mass(i, j) +=
                        fe_values->shape_value(i, q) *
                        fe_values->shape_value(j, q) *
                        fe_values->JxW(q);
                    }
                }
            }

        cell->get_dof_indices(local_dof_indices);

        mat.add(local_dof_indices, local_mass);  

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
              FullMatrix<double> local_mass_stabilization(n_interface_dofs,
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
                        local_mass_stabilization(i, j) +=
                          .5 * ghost_parameter_M  * std::pow(cell_side_length,3) * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);    
                        local_mass_stabilization(i, j) +=
                          .5 * ghost_parameter_M  * std::pow(cell_side_length,5) * normal *
                          fe_interface_values.jump_in_shape_hessians(i, q) * normal *
                          normal *
                          fe_interface_values.jump_in_shape_hessians(j, q) * normal *
                          fe_interface_values.JxW(q);         
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              mat.add(local_interface_dof_indices,
                                   local_mass_stabilization);
            }
        }

    mat.compress(VectorOperation::add);       
    for (auto &entry : mat)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0;
  }
};
#endif
