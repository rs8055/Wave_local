#ifndef L2_ERROR_H
#define L2_ERROR_H

#include "discretization.h"
using namespace dealii;

template <unsigned int dim, typename Number = double>
class L2ErrorOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  L2ErrorOperator(const Discretization<dim, Number>    &discretization, Function<2> *analytical_solution, VectorType &solution_vector, const NonMatching::LocationToLevelSet location_in =
                       NonMatching::LocationToLevelSet::inside)
    : discretization(discretization)
    , analytical_solution(analytical_solution)
    , solution(solution_vector)
    , location(location_in)
    , quadrature_1D(discretization.get_quadrature_1D())
    , mesh_classifiers(discretization.get_mesh_classifiers())
    , fe_collection(discretization.get_fe_collection())
    , level_sets(discretization.get_level_sets())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handlers(discretization.get_dof_handlers())    
  {}

    double get_l2_error(const double final_time) const {
      solution.update_ghost_values();
    
    const NonMatching::LocationToLevelSet inverse_location =
      (location == NonMatching::LocationToLevelSet::inside) ?
        NonMatching::LocationToLevelSet::outside :
        NonMatching::LocationToLevelSet::inside;

    NonMatching::RegionUpdateFlags region_update_flags;
    if (location == NonMatching::LocationToLevelSet::inside)
      region_update_flags.inside = update_values | update_gradients |
                                   update_hessians | update_JxW_values | update_quadrature_points;
    else if (location == NonMatching::LocationToLevelSet::outside)
      region_update_flags.outside = update_values | update_gradients |
                                    update_hessians | update_JxW_values | update_quadrature_points;
    else
      AssertThrow(false, ExcNotImplemented());
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[0],
                                                      level_set_dof_handler,
                                                      level_sets[0]);

    // const auto analytical_solution=sol.analytical_solution.get();
    analytical_solution->set_time(final_time);
    double                  error_L2_squared = 0;

    for (const auto &cell :
         dof_handlers[0]->active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() )//|
      if (mesh_classifiers[0]->location_to_level_set(cell) !=
           inverse_location)
        {
          non_matching_fe_values.reinit(cell);

          const auto &fe_values =
            (location == NonMatching::LocationToLevelSet::inside) ?
              non_matching_fe_values.get_inside_fe_values() :
              non_matching_fe_values.get_outside_fe_values();

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
    error_L2_squared = Utilities::MPI::sum(error_L2_squared, dof_handlers[0]->get_communicator());
    return std::sqrt(error_L2_squared);
  }

  private:
  const Discretization<dim, Number>    &discretization;
  Function<2> *analytical_solution;
  mutable VectorType solution;
  const NonMatching::LocationToLevelSet location;
  const QGauss<1> &quadrature_1D;
  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> & mesh_classifiers;
  const hp::FECollection<dim> &fe_collection;
  const std::vector<VectorType> &level_sets;
  const DoFHandler<dim>       &level_set_dof_handler;
  const std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;
};

#endif