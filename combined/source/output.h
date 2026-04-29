#ifndef OUTPUT_H
#define OUTPUT_H
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
template <unsigned int dim, typename Number = double>
class Output
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  Output(const Discretization<dim, Number>    &discretization, Function<dim> *analytical_solution)
    : discretization(discretization)
    , analytical_solution(analytical_solution)
    , quadrature_1D(discretization.get_quadrature_1D())
    , mesh_classifier(discretization.get_mesh_classifier())
    , fe_collection(discretization.get_fe_collection())
    , level_set(discretization.get_level_set())
    , level_set_dof_handler(discretization.get_level_set_dof_handler())
    , dof_handler(discretization.get_dof_handler())    
  {}

  void output_result(VectorType &solution_vector, const NonMatching::LocationToLevelSet location_in, const double &final_time, const std::string &name) const 
  {
    std::cout << "Writing vtu file" << std::endl;
    const NonMatching::LocationToLevelSet inverse_location =
      (location_in == NonMatching::LocationToLevelSet::inside) ?
        NonMatching::LocationToLevelSet::outside :
        NonMatching::LocationToLevelSet::inside;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution_vector, name);
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    LinearAlgebra::distributed::Vector<double> as_vector;
    LinearAlgebra::distributed::Vector<double> error_vector;
    as_vector.reinit(solution_vector);
    analytical_solution->set_time(final_time);

    VectorTools::interpolate(dof_handler,
                             *analytical_solution,
                             as_vector);

    error_vector.reinit(solution_vector);
    error_vector = as_vector;
    error_vector -= solution_vector;
    data_out.add_data_vector(dof_handler, error_vector, "error_" + name);                         

    data_out.add_data_vector(dof_handler, as_vector, "analytical");

    data_out.set_cell_selection(
      [this, inverse_location](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() && cell->is_locally_owned() &&
               mesh_classifier.location_to_level_set(cell) != inverse_location;
      });

    data_out.build_patches();
    std::ofstream output(name+".vtu");
    data_out.write_vtu(output);
  }


private:
  const Discretization<dim>             &discretization;
  Function<dim> *analytical_solution;
  mutable VectorType solution;
  // const NonMatching::LocationToLevelSet location;
  const QGauss<1> &quadrature_1D;
  const NonMatching::MeshClassifier<dim> &mesh_classifier;
  const hp::FECollection<dim> &fe_collection;
  const VectorType            &level_set;
  const DoFHandler<dim>       &level_set_dof_handler;
  const DoFHandler<dim>       &dof_handler;

};
#endif