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
#include <deal.II/lac/la_parallel_block_vector.h>
#include "discretization.h"

using namespace dealii;

template <unsigned int dim, typename Number = double>
class StiffnessMatrixOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number>    &discretization, const double &gps, const double &np, Function<dim> *rhs,
                    Function<dim> *ibc, Function<dim> *obc, Function<dim> *igf, Function<dim> *ogf, std::vector<std::unique_ptr<Function<dim>>> speed, const bool onc_in = false, const bool inc_in = false, const bool composite_in = false)
    : discretization(discretization)
    , ghost_parameter_S(gps)
    , nitsche_parameter(np)
    , speed(std::move(speed))
    , onc(onc_in)
    , inc(inc_in)
    , composite(composite_in)
    , rhs_function(rhs)
    , interface_boundary_condition(ibc)
    , outer_boundary_condition(obc)
    , interface_gradient_function(igf)
    , outer_gradient_function(ogf)
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
  get_stiffness_matrix() const
  {
    block_sparse_matrix.clear();
    block_sparse_matrix.resize(discretization.get_level_sets().size());
    for(size_t i = 0; i < discretization.get_level_sets().size(); ++i){
      block_sparse_matrix[i] = std::make_shared<TrilinosWrappers::SparseMatrix>();
      compute_sparse_matrix(i, *block_sparse_matrix[i]);
    }
    return block_sparse_matrix;
  }

  void get_rhs_matrix(const size_t domain_idx, VectorType &vec_rhs, const double evaluating_time, const VectorType &previous_u) const
  {
    compute_rhs(domain_idx, vec_rhs, evaluating_time, previous_u);
  }

  void
  get_rhs_matrix(BlockVectorType &vec_rhs, const double evaluating_time, const BlockVectorType &previous_u) const
  {

    previous_u.update_ghost_values();
    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      compute_rhs(domain_idx, vec_rhs.block(domain_idx), evaluating_time, previous_u.block(domain_idx));    
    }

    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size() - 1; ++domain_idx){
      NonMatching::RegionUpdateFlags region_update_flags;
      region_update_flags.surface = update_values | update_gradients |
                                    update_JxW_values | update_quadrature_points |
                                    update_normal_vectors;

      NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                        quadrature_1D,
                                                        region_update_flags,
                                                        *mesh_classifiers[domain_idx],
                                                        level_set_dof_handler,
                                                        level_sets[domain_idx]);

      

      for (const auto &cell_0 : dof_handlers[domain_idx]->active_cell_iterators())
        if (cell_0->is_locally_owned() &&
            (mesh_classifiers[domain_idx]->location_to_level_set(cell_0) ==
            NonMatching::LocationToLevelSet::intersected))
          {
            typename DoFHandler<dim>::active_cell_iterator cell_1(
              &discretization.get_triangulation(),
              cell_0->level(),
              cell_0->index(),
              dof_handlers[domain_idx + 1].get());
            if (mesh_classifiers[domain_idx + 1]->location_to_level_set(cell_1) ==
                NonMatching::LocationToLevelSet::outside)
              continue;

            if (cell_1->active_fe_index() == 
                Discretization<dim>::ActiveFEIndex::nothing)
              continue;
            non_matching_fe_values.reinit(cell_0);

            const double cell_side_length =
              cell_0->minimum_vertex_distance();

            const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
            std::vector<types::global_dof_index> dof_indices_0(n_dofs_per_cell);
            std::vector<types::global_dof_index> dof_indices_1(n_dofs_per_cell);

            cell_0->get_dof_indices(dof_indices_0);
            cell_1->get_dof_indices(dof_indices_1);

            Vector<Number> cell_vector_0(n_dofs_per_cell);
            Vector<Number> cell_vector_1(n_dofs_per_cell);

            // (II) surface integral to apply BC
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;

                std::vector<Number> quadrature_values_0(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(previous_u.block(domain_idx),
                                                      dof_indices_0,
                                                      quadrature_values_0);

                std::vector<Number> quadrature_values_1(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(previous_u.block(domain_idx + 1),
                                                      dof_indices_1,
                                                      quadrature_values_1);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients_0(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(previous_u.block(domain_idx),
                                                        dof_indices_0,
                                                        quadrature_gradients_0);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients_1(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(previous_u.block(domain_idx + 1),
                                                        dof_indices_1,
                                                        quadrature_gradients_1);


                for (const unsigned int q :
                    surface_fe_values.quadrature_point_indices())
                  {
                    const Point<dim> point= surface_fe_values.quadrature_point(q);
                        double c_surface= speed[domain_idx]->value(point);
                        double c_surface_other= speed[domain_idx + 1]->value(point);
                        double k_1 = c_surface_other/(c_surface + c_surface_other);
                        double k_2 = c_surface/(c_surface + c_surface_other);
                    const Tensor<1, dim> normal =
                      surface_fe_values.normal_vector(q);

                    // const auto tau_parameter = (0.5) * nitsche_parameter;
                    const auto tau_parameter = ((c_surface * c_surface_other)/(c_surface + c_surface_other)) * nitsche_parameter;

                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        const auto quadrature_value_jump =
                          (quadrature_values_0[q] - quadrature_values_1[q]);
                        const auto quadrature_gradient_avg =
                          (k_1 * c_surface * quadrature_gradients_0[q] + k_2 * c_surface_other * quadrature_gradients_1[q]);
                          //  (quadrature_gradients_0[q] +  quadrature_gradients_1[q]);

                        cell_vector_0(i) -=
                          (-(k_1) * c_surface * normal * surface_fe_values.shape_grad(i, q) *
                            quadrature_value_jump -
                          surface_fe_values.shape_value(i, q) * normal *
                            quadrature_gradient_avg +
                          tau_parameter / cell_side_length * 
                            surface_fe_values.shape_value(i, q) *
                            quadrature_value_jump) *
                          surface_fe_values.JxW(q);

                        cell_vector_1(i) -=
                          (-(k_2) * c_surface_other * normal * surface_fe_values.shape_grad(i, q) *
                            quadrature_value_jump +
                          surface_fe_values.shape_value(i, q) * normal * 
                            quadrature_gradient_avg -
                          tau_parameter / cell_side_length *
                            surface_fe_values.shape_value(i, q) *
                            quadrature_value_jump) *
                          surface_fe_values.JxW(q);
                      }
                  }
              }

            // cell->get_dof_indices(dof_indices);
            vec_rhs.block(domain_idx).add(dof_indices_0, cell_vector_0);
            vec_rhs.block(domain_idx + 1).add(dof_indices_1, cell_vector_1);
          }
        }

    vec_rhs.compress(VectorOperation::add);
  }

private:
  const Discretization<dim, Number>    &discretization;
  double ghost_parameter_S;
  double nitsche_parameter;
  std::vector<std::unique_ptr<Function<dim>>> speed;
  const bool onc;
  const bool inc;
  const bool composite;
  Function<2> *rhs_function;
  Function<2> *interface_boundary_condition;
  Function<2> *outer_boundary_condition;
  Function<2> *interface_gradient_function;
  Function<2> *outer_gradient_function;
  const QGauss<1> &quadrature_1D;
  const QGauss<dim - 1> &face_quadrature;
  const std::vector<AffineConstraints<Number>> &constraints;
  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> & mesh_classifiers;
  const hp::FECollection<dim> &fe_collection;
  const std::vector<VectorType> &level_sets;
  const DoFHandler<dim>       &level_set_dof_handler;
  const std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;
  // mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> block_sparse_matrix;

  void
  compute_sparse_matrix(size_t & domain_idx, TrilinosWrappers::SparseMatrix &mat) const
  {    
    const Function<dim> *required_speed = speed[domain_idx].get();                                      

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
    

     TrilinosWrappers::SparsityPattern sparsity_pattern;
    sparsity_pattern.reinit(dof_handlers[domain_idx]->locally_owned_dofs(),
                            dof_handlers[domain_idx]->get_communicator());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(*dof_handlers[domain_idx],
                                         sparsity_pattern,
                                         constraints[domain_idx],
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_ghost_penalty);
                                         
    sparsity_pattern.compress();
    mat.reinit(sparsity_pattern);

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                               update_hessians |
                                                 update_JxW_values |
                                                 update_normal_vectors|
                                             update_quadrature_points);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                   update_hessians | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    NonMatching::RegionUpdateFlags region_update_flags_face;
      region_update_flags_face.inside =
        update_values | update_gradients | update_JxW_values | update_hessians | 
        update_quadrature_points | update_normal_vectors;
    
    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe_collection,
      quadrature_1D,
      region_update_flags_face,
      *mesh_classifiers[domain_idx],
      level_set_dof_handler,
      level_sets[domain_idx]);
                                                      
    for (const auto &cell :
         dof_handlers[domain_idx]->active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() )//|
      if (mesh_classifiers[domain_idx]->location_to_level_set(cell) != NonMatching::LocationToLevelSet::outside)
        {
          local_stiffness = 0;

          const double cell_side_length = cell->minimum_vertex_distance();

          non_matching_fe_values.reinit(cell);

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values() ;

          if (fe_values)
          {
            for (const unsigned int q :
                fe_values->quadrature_point_indices())
              {
                const Point<dim> point= fe_values->quadrature_point(q);
                    double speed_cell= required_speed->value(point);
                for (const unsigned int i : fe_values->dof_indices())
                  {
                    for (const unsigned int j : fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          speed_cell*fe_values->shape_grad(i, q) *
                          fe_values->shape_grad(j, q) *
                          fe_values->JxW(q);
                      }
                  }
              }
          }
  
            if (!inc)
            {
              const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
                &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

              if (surface_fe_values)
                {
                  for (const unsigned int q :
                      surface_fe_values->quadrature_point_indices())
                    {
                      const Point<dim> point= surface_fe_values->quadrature_point(q);
                        double c_surface= required_speed->value(point);
                      Tensor<1, dim> normal =
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
                                  surface_fe_values->shape_value(j, q)) * c_surface *
                                surface_fe_values->JxW(q);
                              
                            }
                        }
                    }
                }
            }
        

          if(!onc)
          {
            for (const unsigned int f : cell->face_indices())
              if (cell->at_boundary(f))
                {
                  non_matching_fe_interface_values.reinit(cell,f);        
                  if (const auto &surface_fe_value_ptr = non_matching_fe_interface_values
                              .get_inside_fe_values() )
                  {
                    const auto &surface_fe_values =
                            surface_fe_value_ptr->get_fe_face_values(0);
                    for (const unsigned int q :
                    surface_fe_values.quadrature_point_indices())
                    {
                      const Point<dim> point= surface_fe_values.quadrature_point(q);
                        double c_surface= required_speed->value(point);
                      const Tensor<1, dim> &normal =
                        surface_fe_values.normal_vector(q);
                      for (const unsigned int i : surface_fe_values.dof_indices())
                        {
                          for (const unsigned int j : surface_fe_values.dof_indices())
                            {
                                local_stiffness(i, j) +=
                                  (-normal * surface_fe_values.shape_grad(i, q) *
                                    surface_fe_values.shape_value(j, q) +
                                  -normal * surface_fe_values.shape_grad(j, q) *
                                    surface_fe_values.shape_value(i, q) +
                                  nitsche_parameter / cell_side_length *
                                    surface_fe_values.shape_value(i, q) *
                                    surface_fe_values.shape_value(j, q)) * c_surface *
                                  surface_fe_values.JxW(q);
                            }
                        }
                    }
                  }
                }
          }

          cell->get_dof_indices(local_dof_indices);

          mat.add(local_dof_indices, local_stiffness);  

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
                    const Point<dim> point= fe_interface_values.quadrature_point(q);
                    double c_interface= required_speed->value(point);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      for (unsigned int j = 0; j < n_interface_dofs; ++j)
                        {
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter_S  *  c_interface * cell_side_length * normal *
                            fe_interface_values.jump_in_shape_gradients(i, q) *
                            normal *
                            fe_interface_values.jump_in_shape_gradients(j, q) *
                            fe_interface_values.JxW(q);
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter_S  *  c_interface * std::pow(cell_side_length,3) * normal *
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
                                    local_stabilization);
              }
        }

    mat.compress(VectorOperation::add);    
    for (auto &entry : mat)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0; 
  }

  void
  compute_rhs(size_t domain_idx, VectorType &vec_rhs, const double evaluating_time, const VectorType &previous_u) const
  {
    const Function<dim> *required_speed = speed[domain_idx].get();  

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
    discretization.initialize_dof_vector(vec_rhs, domain_idx); 
    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    Vector<double> local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                               update_hessians |
                                                 update_JxW_values |
                                                 update_normal_vectors|
                                             update_quadrature_points);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                   update_hessians | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    rhs_function->set_time(evaluating_time);
    interface_boundary_condition->set_time(evaluating_time);  
    outer_boundary_condition->set_time(evaluating_time);                       

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    NonMatching::RegionUpdateFlags region_update_flags_face;
    region_update_flags_face.inside =
        update_values | update_gradients | update_JxW_values | update_hessians | 
        update_quadrature_points | update_normal_vectors;   
    
    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe_collection,
      quadrature_1D,
      region_update_flags_face,
      *mesh_classifiers[domain_idx],
      level_set_dof_handler,
      level_sets[domain_idx]);                                                  

    for (const auto &cell :
         dof_handlers[domain_idx]->active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() )
      if (mesh_classifiers[domain_idx]->location_to_level_set(cell) !=
           NonMatching::LocationToLevelSet::outside)
      {
        local_rhs = 0;
        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        // ============================================================
        // VOLUME SOURCE TERM: ∫ f φᵢ dx
        // ============================================================
        const auto &fe_values = non_matching_fe_values.get_inside_fe_values() ;

        if (fe_values)
          {
            std::vector<Tensor<1,dim,double>> solution_gradient_values(fe_values->n_quadrature_points);
                fe_values->get_function_gradients(previous_u, solution_gradient_values);
            for (const unsigned int q :
                 fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);

                double speed_cell= required_speed->value(point);
                
                // Evaluate f at θ*t^n + (1-θ)*t^{n-1}
                const double f_value = rhs_function->value(point);

                for (const unsigned int i : fe_values->dof_indices())
                  {
                    local_rhs(i) -=    speed_cell *
                      fe_values->shape_grad(i, q) *
                          solution_gradient_values.at(q) *
                          fe_values->JxW(q);

                    local_rhs(i) += f_value *
                                    fe_values->shape_value(i, q) *
                                    fe_values->JxW(q);
                  }
              }
          }

        // ============================================================
        // BOUNDARY TERMS: Nitsche RHS
        // ============================================================
        if(!composite)
        {
          const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

          if (surface_fe_values)
            {
              std::vector<double> solution_values(surface_fe_values->n_quadrature_points);
                  surface_fe_values->get_function_values(previous_u, solution_values);
              std::vector<Tensor<1,dim,double>> solution_gradient_values(surface_fe_values->n_quadrature_points);
                  surface_fe_values->get_function_gradients(previous_u, solution_gradient_values);
              for (const unsigned int q :
                  surface_fe_values->quadrature_point_indices())
                {
                  const Point<dim> &point =
                    surface_fe_values->quadrature_point(q);
                  Tensor<1, dim> normal =
                    surface_fe_values->normal_vector(q);
                  double c_surface= required_speed->value(point);

                  if (inc)
                  {
                    // // if we have neumann interface boundary data
                    interface_gradient_function->set_time(evaluating_time);
                    const Tensor<1,dim> g_value = interface_gradient_function->gradient(point);
                    for (const unsigned int i : surface_fe_values->dof_indices())
                      {
                        local_rhs(i) += 
                        normal* g_value * c_surface *
                        (surface_fe_values->shape_value(i, q) ) *
                        surface_fe_values->JxW(q);
                      }
                  }
                  else
                  {
                    // // if we have dirichlet interface boundary data
                    const double g_value = interface_boundary_condition->value(point);
                    for (const unsigned int i : surface_fe_values->dof_indices())
                      {
                        local_rhs(i) -=
                                 (-normal * surface_fe_values->shape_grad(i, q) *
                                  solution_values.at(q) +
                                -normal * solution_gradient_values.at(q) *
                                  surface_fe_values->shape_value(i, q) +
                                nitsche_parameter / cell_side_length *
                                  surface_fe_values->shape_value(i, q) *
                                  solution_values.at(q)) * c_surface *
                                surface_fe_values->JxW(q) ;

                        local_rhs(i) +=
                          g_value * c_surface *
                          (nitsche_parameter / cell_side_length *
                            surface_fe_values->shape_value(i, q) -
                          normal * surface_fe_values->shape_grad(i, q)) *
                          surface_fe_values->JxW(q);
                      }
                  }
                }
            }
        }

          for (const unsigned int f : cell->face_indices())
          if (cell->at_boundary(f))
            {
              non_matching_fe_interface_values.reinit(cell,f);        
              if (const auto &surface_fe_value_ptr = non_matching_fe_interface_values
                            .get_inside_fe_values() )
              {
                const auto &surface_fe_values =
                        surface_fe_value_ptr->get_fe_face_values(0);
                std::vector<double> solution_values(surface_fe_values.n_quadrature_points);
                    surface_fe_values.get_function_values(previous_u, solution_values);

                std::vector<Tensor<1,dim,double>> solution_gradient_values(surface_fe_values.n_quadrature_points);
                    surface_fe_values.get_function_gradients(previous_u, solution_gradient_values);
                        for (const unsigned int q :
                 surface_fe_values.quadrature_point_indices())
                {
                  const Point<dim> point= surface_fe_values.quadrature_point(q);
                    double c_surface= required_speed->value(point);
                  const Tensor<1, dim> &normal =
                    surface_fe_values.normal_vector(q);

                  if (onc)
                  {
                    // // if we have neumann outer boundary data
                    outer_gradient_function->set_time(evaluating_time);
                    const Tensor<1,dim> g_value = outer_gradient_function->gradient(point);
                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        local_rhs(i) += 
                          normal* g_value * c_surface *
                          (surface_fe_values.shape_value(i, q) ) *
                          surface_fe_values.JxW(q);
                      }
                  }
                  else
                  {
                    // // if we have dirichlet outer boundary data
                    const double g_value = outer_boundary_condition->value(point);
                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        local_rhs(i) -=
                               (-normal * surface_fe_values.shape_grad(i, q) *
                                solution_values.at(q) +
                              -normal * solution_gradient_values.at(q) *
                                surface_fe_values.shape_value(i, q) +
                              nitsche_parameter / cell_side_length *
                                surface_fe_values.shape_value(i, q) *
                                solution_values.at(q)) * c_surface *
                              surface_fe_values.JxW(q);
                        local_rhs(i) +=
                          g_value * c_surface *
                          (nitsche_parameter / cell_side_length *
                            surface_fe_values.shape_value(i, q) -
                          normal *  surface_fe_values.shape_grad(i, q)) *
                          surface_fe_values.JxW(q);
                      }
                  }
                }
              }
            }

        cell->get_dof_indices(local_dof_indices);
        vec_rhs.add(local_dof_indices, local_rhs);

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
              Vector<double> local_rhs_stabilization(n_interface_dofs); 
              
              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              std::vector<Tensor<1, dim>> jump_in_shape_gradients(
                    fe_interface_values.n_quadrature_points);
              std::vector<Tensor<2, dim>> jump_in_shape_hessians(
                    fe_interface_values.n_quadrature_points); 
                    
              const FEValuesExtractors::Scalar scalar(0);

                  std::vector<double> local_dof_values(n_interface_dofs);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    local_dof_values[i] =
                      previous_u[local_interface_dof_indices[i]];

                  fe_interface_values[scalar]
                    .get_jump_in_function_gradients_from_local_dof_values(
                      local_dof_values, jump_in_shape_gradients);
                  fe_interface_values[scalar]
                    .get_jump_in_function_hessians_from_local_dof_values(
                      local_dof_values, jump_in_shape_hessians);    

              for (unsigned int q = 0;
                   q < fe_interface_values.n_quadrature_points;
                   ++q)
                {
                  const Tensor<1, dim> normal =
                    fe_interface_values.normal(q);
                  const Point<dim> point= fe_interface_values.quadrature_point(q);
                  double c_interface= required_speed->value(point);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      {              
                        local_rhs_stabilization(i) -=
                          .5 * ghost_parameter_S * c_interface * cell_side_length * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          jump_in_shape_gradients[q] *
                          fe_interface_values.JxW(q);
                        local_rhs_stabilization(i) -=
                          .5 * ghost_parameter_S * c_interface * std::pow(cell_side_length,3) * normal *
                          fe_interface_values.jump_in_shape_hessians(i, q) * normal *
                          normal *
                          jump_in_shape_hessians[q] * normal *
                          fe_interface_values.JxW(q);
                      }
                }

              vec_rhs.add(local_interface_dof_indices, local_rhs_stabilization);
            }
      }

    vec_rhs.compress(VectorOperation::add);
  }
};
#endif
