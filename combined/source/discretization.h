#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H
#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <unsigned int dim, typename Number = double>
class Discretization
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  enum ActiveFEIndex
    {
      lagrange = 0,
      nothing  = 1
    };



  // ─── Constructor ───────────────────────────────────────────────
  Discretization(
    const unsigned int                          fe_degree,
    unsigned int                          n_subdivisions_1D,
    const double                                geometry_left,
    const double                                geometry_right
    // const unsigned int                          n_components        = 1,
    // const unsigned int                          level_set_fe_degree = 1,
    // std::shared_ptr<Function<dim>>              level_set_function  = nullptr,
    // std::function<Point<dim>(const Point<dim>&)> mapping_q_cache_function = nullptr
  ): tria(MPI_COMM_WORLD)
    , level_set_dof_handler(tria)
    , dof_handler(tria)
    , quadrature_1D(fe_degree + 1)  
    , face_quadrature(fe_degree + 1)
  {
    GridGenerator::subdivided_hyper_cube(tria,
                                         n_subdivisions_1D,
                                         geometry_left,
                                         geometry_right);
    dx = (geometry_right - geometry_left) / n_subdivisions_1D;


    // ── Level set ────────────────────────────────────────────────
    level_set_dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

    const auto level_set_partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
      level_set_dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(level_set_dof_handler),
      level_set_dof_handler.get_communicator());
    level_set.reinit(level_set_partitioner);

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
    level_set.update_ghost_values();

    mesh_classifier = std::make_shared<NonMatching::MeshClassifier<dim>>(
      level_set_dof_handler, level_set);
    mesh_classifier->reclassify();

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier->location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(ActiveFEIndex::nothing);
        else
          cell->set_active_fe_index(ActiveFEIndex::lagrange);
      }

    dof_handler.distribute_dofs(fe_collection);
    

    // ── Constraints ──────────────────────────────────────────────
    constraints.close();

    // ── Quadrature ───────────────────────────────────────────────
    // quadrature_1D   = QGauss<1>(fe_degree + 1);
    // face_quadrature = QGauss<dim - 1>(fe_degree + 1);

    partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_active_dofs(dof_handler),
      dof_handler.get_communicator());
  }

  // ─── Public getters ────────────────────────────────────────────
  const QGauss<1> &
  get_quadrature_1D() const { return quadrature_1D; }

  const QGauss<dim - 1> &
  get_face_quadrature() const { return face_quadrature; }

  const AffineConstraints<Number> &
  get_affine_constraints() const { return constraints; }

  const DoFHandler<dim> &
  get_level_set_dof_handler() const { return level_set_dof_handler; }

  const VectorType &
  get_level_set() const { return level_set; }

  const hp::FECollection<dim> &
  get_fe_collection() const { return fe_collection; }

  const DoFHandler<dim> &
  get_dof_handler() const { return dof_handler; }

  const NonMatching::MeshClassifier<dim> &
  get_mesh_classifier() const { return *mesh_classifier; }

  double
  get_dx() const { return dx; }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(partitioner);
  }

private:
  parallel::distributed::Triangulation<dim>          tria;
  QGauss<1>                                      quadrature_1D;
  QGauss<dim - 1>                                face_quadrature;
  AffineConstraints<Number>                          constraints;
  DoFHandler<dim>                                    level_set_dof_handler;
  VectorType                                         level_set;
  std::shared_ptr<NonMatching::MeshClassifier<dim>>  mesh_classifier;
  hp::FECollection<dim>                              fe_collection;
  DoFHandler<dim>                                    dof_handler;
  double                                             dx;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
};

#endif