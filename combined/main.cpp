#include "main.hpp"
#include "wave_solutions.hpp"
#include "heat_solutions.hpp"
#include "poisson_solutions.hpp"
#include "discretization.h"
#include "mass_matrix.h"
#include "stiffness_matrix.h"
#include "solver.h"
#include "l2_error.h"

using namespace dealii;

template <typename T, typename = void>
struct has_speed : std::false_type {};
template <typename T>
struct has_speed<T, std::void_t<decltype(std::declval<T>().speed)>>
    : std::true_type {};

int main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("solution,s", boost::program_options::value<int>()->default_value(0), "Select the solution")
    ("degree,k", boost::program_options::value<size_t>()->default_value(2), "The polynomial degree of the sequence")
    ("pde", boost::program_options::value<int>()->default_value(0), "Select the PDE; 0~wave, 1~heat, and 2~poisson")
    ("gpm", boost::program_options::value<double>()->default_value(0.43), "Ghost Penalty for Mass Matrix")
    ("gps", boost::program_options::value<double>()->default_value(0.86), "Ghost Penalty for Stiffness Matrix")
    ("np",  boost::program_options::value<double>()->default_value(20), "Nitsche Penalty")
    ("cfl", boost::program_options::value<double>()->default_value(0.1), "CFL Parameter")
    ("solver", boost::program_options::value<std::string>()->default_value("direct"), "Choice of solver")
    ("lower,l", boost::program_options::value<double>()->default_value(-2.), "Left end point of the domain")
    ("upper,u", boost::program_options::value<double>()->default_value(2.),  "Right end point of the domain")
    ("partition,n", boost::program_options::value<double>()->default_value(40.), "Number of partitions");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  const int          S                = vm["solution"].as<int>();
  const size_t       K                = vm["degree"].as<size_t>();
  const int          pde              = vm["pde"].as<int>();
  const double       gpm              = vm["gpm"].as<double>();
  const double       gps              = vm["gps"].as<double>();
  const double       np               = vm["np"].as<double>();
  const double       cfl              = vm["cfl"].as<double>();
  const unsigned int n_subdivisions_1D = static_cast<unsigned int>(vm["partition"].as<double>());
  const double       geometry_left    = vm["lower"].as<double>();
  const double       geometry_right   = vm["upper"].as<double>();
  const std::string  lin_solver_type  = vm["solver"].as<std::string>();

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  auto run = [&](auto sol)
  {
    // ── Discretization ─────────────────────────────────────────
    Discretization<2> discretization(K,
                                     n_subdivisions_1D,
                                     geometry_left,
                                     geometry_right);

    VectorType solution;
    discretization.initialize_dof_vector(solution);

    double speed = 1.0;
    if constexpr (has_speed<decltype(sol)>::value)
        speed = sol.speed;

    // ── Matrices ────────────────────────────────────────────────
    StiffnessMatrixOperator<2> stiffness_matrix(discretization,
                                             gps,
                                             np,
                                             sol.rhs_function.get(),
                                             sol.boundary_values.get(),
                                             speed);

    TrilinosWrappers::SparseMatrix system_matrix;
    if (pde == 2)
    {
      // Poisson — use stiffness matrix
      system_matrix.copy_from(stiffness_matrix.get_stiffness_matrix());
    }
    else
    {
      // Heat / Wave — use mass matrix
      MassMatrixOperator<2> mass_matrix(discretization, gpm);
      system_matrix.copy_from(mass_matrix.get_mass_matrix());
    }

    // ── Solve ───────────────────────────────────────────────────
    Solver<2, decltype(sol)> solver(std::move(sol),
                                    discretization,
                                    stiffness_matrix,
                                    system_matrix,
                                    pde,
                                    cfl);
    solver.solve();
    solution = solver.get_solution();                              
    solution.update_ghost_values();

    // ── L2 error ────────────────────────────────────────────────
    L2ErrorOperator<2> l2_error(discretization,
                             solver.get_analytical_solution(),
                             solution);
    const double error_L2 = l2_error.get_l2_error(solver.get_final_time());
    std::cout << "L2 error: " << error_L2<< std::endl;
  };

  switch (pde)
  {
    case 0:
      std::cout << "[main] Test case: Wave Equation" << std::endl;
      run(Combined::Wave::make_solution<2>(S));
      break;

    case 1:
      std::cout << "[main] Test case: Heat Equation" << std::endl;
      run(Combined::Heat::make_solution<2>(S));
      break;

    case 2:
      std::cout << "[main] Test case: Poisson Equation" << std::endl;
      run(Combined::Poisson::make_solution<2>(S));
      break;

    default:
      std::cerr << "[main] ERROR: Unknown PDE" << std::endl;
      return 1;
  }

  std::cout << "Done." << std::endl;
  return 0;
}