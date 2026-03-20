#ifndef POISSON_SOLUTIONS_HPP
#define POISSON_SOLUTIONS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <memory>
#include <cmath>

namespace Combined
{
namespace Poisson         
{
using namespace dealii;

// ═══════════════════════════════════════════════════════
// SolutionSet struct — holds all function types together
// ═══════════════════════════════════════════════════════
template <int dim>
struct SolutionSet
{
    std::unique_ptr<Function<dim>> analytical_solution;
    std::unique_ptr<Function<dim>> rhs_function;
    std::unique_ptr<Function<dim>> boundary_values;
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 1: sin(x)sin(y)
// ═══════════════════════════════════════════════════════

template <int dim>
class AnalyticalSolution0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return std::sin(p[0]) * std::sin(p[1]);
    }
};

template <int dim>
class RHSFunction0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return -2.0 * std::sin(p[0]) * std::sin(p[1]);
    }
};

template <int dim>
class BoundaryValues0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]);
    }
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 2: (1 - 2/dim*(|x|^2 - 1))
// ═══════════════════════════════════════════════════════

template <int dim>
class AnalyticalSolution1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return (1. - 2. / dim * (p.norm_square() - 1.));
    }
};

template <int dim>
class RHSFunction1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return 4.0;
    }
};

template <int dim>
class BoundaryValues1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return (1. - 2. / dim * (p.norm_square() - 1.));
    }
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 3: Bessel function J0(alpha*r)
// ═══════════════════════════════════════════════════════

template <int dim>
class AnalyticalSolution2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm());
    }
};

template <int dim>
class RHSFunction2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return (std::pow(alpha,2)) * std::cyl_bessel_j(0, alpha * p.norm());
    }
};

template <int dim>
class BoundaryValues2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm());
    }
};


// ═══════════════════════════════════════════════════════
//  FACTORY — one call returns all functions for a choice
// ═══════════════════════════════════════════════════════

template <int dim>
SolutionSet<dim> make_solution(const int choice)
{
    SolutionSet<dim> s;
    switch (choice)
    {
        case 0:
            s.analytical_solution = std::make_unique<AnalyticalSolution0<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction0<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues0<dim>>();
            break;
        case 1:
            s.analytical_solution = std::make_unique<AnalyticalSolution1<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction1<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues1<dim>>();
            break;
        case 2:
            s.analytical_solution = std::make_unique<AnalyticalSolution2<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction2<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues2<dim>>();
            break;
        default:
            AssertThrow(false, ExcMessage("Unknown solution choice: "
                                          + std::to_string(choice)));
    }
    return s;
}
} // namespace Poisson
} // namespace Combined

#endif // POISSON_SOLUTIONS_HPP