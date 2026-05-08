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
    const unsigned int            fe_degree,
    unsigned int                  n_subdivisions_1D,
    const double                  geometry_left,
    const double                  geometry_right,
    std::vector<std::unique_ptr<Function<dim>>> lsf,
    const bool                    composite_in      
  ): tria(MPI_COMM_WORLD)
    , level_set_dof_handler(tria)
    , fe_degree(fe_degree)
    , n_subdivisions_1D(n_subdivisions_1D)
    , geometry_left(geometry_left)
    , geometry_right(geometry_right)
    , level_set_functions(std::move(lsf))
    , composite(composite_in)
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

    if (level_set_functions.size() > 0)
    {
      level_sets.resize(level_set_functions.size());
      for (unsigned int i = 0; i < level_set_functions.size(); ++i)
      {
        level_sets[i].reinit(level_set_partitioner);
        VectorTools::interpolate(level_set_dof_handler,
                                *level_set_functions[i],
                                level_sets[i]);
        level_sets[i].update_ghost_values();

        mesh_classifiers.push_back(
        std::make_shared<NonMatching::MeshClassifier<dim>>(
          level_set_dof_handler, level_sets[i]));
        mesh_classifiers[i]->reclassify();

        dof_handlers.push_back(std::make_unique<DoFHandler<dim>>(tria));
      }
    }
    else
    {
      level_sets.resize(2);
      const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
      for (unsigned int i = 0; i < 2; ++i)
      {
        level_sets[i].reinit(level_set_partitioner);
        VectorTools::interpolate(level_set_dof_handler,
                                signed_distance_sphere,
                                level_sets[i]);
        if (i == 1)
            level_sets[i] *= -1.0;
        level_sets[i].update_ghost_values();

        mesh_classifiers.push_back(
            std::make_shared<NonMatching::MeshClassifier<dim>>(
                level_set_dof_handler, level_sets[i]));
        mesh_classifiers[i]->reclassify();

        dof_handlers.push_back(std::make_unique<DoFHandler<dim>>(tria));
      }
    }

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());
   
    // ── Constraints ──────────────────────────────────────────────
    const int n_domains = level_sets.size();
    constraints.resize(n_domains); 
    partitioners.resize(n_domains);

    for(int i = 0; i < n_domains; ++i){
      for (const auto &cell : dof_handlers[i]->active_cell_iterators() |
            IteratorFilters::LocallyOwnedCell())
        {        
          if(composite)
          {
            cell->set_active_fe_index(ActiveFEIndex::lagrange);
          }
          else
          {
            const NonMatching::LocationToLevelSet cell_location =
              mesh_classifiers[i]->location_to_level_set(cell);
            if (cell_location == NonMatching::LocationToLevelSet::outside)
              cell->set_active_fe_index(ActiveFEIndex::nothing);
            else
              cell->set_active_fe_index(ActiveFEIndex::lagrange); 
          }
        }

      dof_handlers[i]->distribute_dofs(fe_collection);
      constraints[i].close();
      partitioners[i] = std::make_shared<const Utilities::MPI::Partitioner>(
        dof_handlers[i]->locally_owned_dofs(),
        DoFTools::extract_locally_active_dofs(*dof_handlers[i]),
        dof_handlers[i]->get_communicator());
    }
  }

  // ─── Public getters ────────────────────────────────────────────
  const QGauss<1> &
  get_quadrature_1D() const { return quadrature_1D; }

  const QGauss<dim - 1> &
  get_face_quadrature() const { return face_quadrature; }

  const std::vector<AffineConstraints<Number>> &
  get_affine_constraints() const { return constraints; }

  const DoFHandler<dim> &
  get_level_set_dof_handler() const { return level_set_dof_handler; }

  const std::vector<VectorType> &
  get_level_sets() const { return level_sets; }

  const hp::FECollection<dim> &
  get_fe_collection() const { return fe_collection; }

  const std::vector<std::shared_ptr<DoFHandler<dim>>> &
  get_dof_handlers() const { return dof_handlers; }

  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &
  get_mesh_classifiers() const { return mesh_classifiers; }

  double
  get_dx() const { return dx; }

  void
  initialize_dof_vector(VectorType &vec, const int & i) const
  {
    vec.reinit(partitioners[i]);
  }

private:
  parallel::distributed::Triangulation<dim>          tria;
  DoFHandler<dim>                                    level_set_dof_handler;
  std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handlers;
  const unsigned int                                 fe_degree;
  unsigned int                                       n_subdivisions_1D;
  const double                                       geometry_left;
  const double                                       geometry_right;
  std::vector<std::unique_ptr<Function<dim>>>        level_set_functions;
  bool                                               cavity;
  bool                                               composite;
  QGauss<1>                                          quadrature_1D;
  QGauss<dim - 1>                                    face_quadrature;
  std::vector<VectorType>                            level_sets;
  std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>>      mesh_classifiers;
  hp::FECollection<dim>                              fe_collection; 
  double                                             dx;
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners; 
  std::vector<AffineConstraints<Number>>                          constraints;    
};

#endif