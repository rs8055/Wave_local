#ifndef L2_ERROR_H
#define L2_ERROR_H

#include "discretization.h"
using namespace dealii;

template <unsigned int dim, typename Number = double>
class L2ErrorOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  L2ErrorOperator(const Discretization<dim, Number>    &discretization, Function<2> *analytical_solution, VectorType &solution_vector)
    : discretization(discretization)
    , analytical_solution(analytical_solution)
    , solution(solution_vector)
    , quadrature_1D(discretization.get_quadrature_1D())
    , mesh_classifier(discretization.get_mesh_classifier())
    , fe_collection(discretization.get_fe_collection())
    , level_set(discretization.get_level_set())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handler(discretization.get_dof_handler())    
  {}

  const double get_l2_error(const double final_time) const {
    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<2> non_matching_fe_values(discretization.get_fe_collection(),
                                                      discretization.get_quadrature_1D(),
                                                      region_update_flags,
                                                      discretization.get_mesh_classifier(),
                                                      discretization.get_level_set_dof_handler(),
                                                      discretization.get_level_set());

    // const auto analytical_solution=sol.analytical_solution.get();
    analytical_solution->set_time(final_time);
    double                  error_L2_squared = 0;

    for (const auto &cell :
         discretization.get_dof_handler().active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell() |
           IteratorFilters::ActiveFEIndexEqualTo(Discretization<2>::ActiveFEIndex::lagrange))
      {
        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<2>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

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


    // solution.zero_out_ghost_values();

    error_L2_squared = Utilities::MPI::sum(error_L2_squared, discretization.get_dof_handler().get_communicator());
    return std::sqrt(error_L2_squared);
  }

  private:
  const Discretization<dim, Number>    &discretization;
  Function<2> *analytical_solution;
  mutable VectorType solution;
  const QGauss<1> &quadrature_1D;
  const NonMatching::MeshClassifier<dim> &mesh_classifier;
  const hp::FECollection<dim> &fe_collection;
  const VectorType            &level_set;
  const DoFHandler<dim>       &level_set_dof_handler;
  const DoFHandler<dim>       &dof_handler;

};

#endif