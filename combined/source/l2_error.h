#ifndef L2_ERROR_H
#define L2_ERROR_H

#include "discretization.h"
using namespace dealii;

template <unsigned int dim, typename Number = double>
class L2ErrorOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  L2ErrorOperator(const Discretization<dim, Number>    &discretization, Function<2> *analytical_solution, VectorType &solution_vector, const size_t &domain_idx)
    : discretization(discretization)
    , analytical_solution(analytical_solution)
    , solution(solution_vector)
    , domain_idx(domain_idx)
    , quadrature_1D(discretization.get_quadrature_1D())
    , mesh_classifiers(discretization.get_mesh_classifiers())
    , fe_collection(discretization.get_fe_collection())
    , level_sets(discretization.get_level_sets())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handlers(discretization.get_dof_handlers())    
  {}

    double get_l2_error(const double final_time) const {
      solution.update_ghost_values();
    
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

    analytical_solution->set_time(final_time);
    double                  error_L2_squared = 0;

    for (const auto &cell :
         dof_handlers[domain_idx]->active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() )//|
      if (mesh_classifiers[domain_idx]->location_to_level_set(cell) != NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell);

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          if (fe_values)
            {
              std::vector<double> solution_values(fe_values->n_quadrature_points);
              fe_values->get_function_values(solution, solution_values);

              for (const unsigned int q : fe_values->quadrature_point_indices())
                {
                  const Point<2> &point = fe_values->quadrature_point(q);
                  const double      error_at_point =
                    solution_values.at(q) - analytical_solution->value(point);
                  error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) * fe_values->JxW(q);
                }
            }
        }
    error_L2_squared = Utilities::MPI::sum(error_L2_squared, dof_handlers[domain_idx]->get_communicator());
    return std::sqrt(error_L2_squared);
  }

  private:
  const Discretization<dim, Number>    &discretization;
  Function<2> *analytical_solution;
  mutable VectorType solution;
  const size_t domain_idx;
  const QGauss<1> &quadrature_1D;
  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> & mesh_classifiers;
  const hp::FECollection<dim> &fe_collection;
  const std::vector<VectorType> &level_sets;
  const DoFHandler<dim>       &level_set_dof_handler;
  const std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;
};

#endif