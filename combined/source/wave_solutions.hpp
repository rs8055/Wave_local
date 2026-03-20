#ifndef WAVE_SOLUTIONS_HPP
#define WAVE_SOLUTIONS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <memory>
#include <cmath>

namespace Combined
{
namespace Wave         
{
using namespace dealii;

// ═══════════════════════════════════════════════════════
// SolutionSet struct — holds all function types together
// ═══════════════════════════════════════════════════════
template <int dim>
struct SolutionSet
{
    // std::unique_ptr<Function<dim>> speed;
    std::unique_ptr<Function<dim>> analytical_solution;
    std::unique_ptr<Function<dim>> rhs_function;
    std::unique_ptr<Function<dim>> boundary_values;
    std::unique_ptr<Function<dim>> initial_data;
    std::unique_ptr<Function<dim>> derivative_function;
    double initial_time;
    double final_time;
    double speed;
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 1: sin(x)sin(y)cos(sqrt(2)t)
// ═══════════════════════════════════════════════════════

// template <int dim>
// class Speed0 : public Function<dim>
// {
// public:
//     double value(const Point<dim> &p,
//                  const unsigned int component = 0) const override
//     {
//         (void)component;
//         return 1.0;
//     }
// };

template <int dim>
class AnalyticalSolution0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]) * std::cos(std::sqrt(2.0) * t);
    }
};

template <int dim>
class RHSFunction0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)p; (void)component;
        return 0.0;
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
        return std::sin(p[0]) * std::sin(p[1]) * std::cos(std::sqrt(2.0) * t);
    }
};

template <int dim>
class InitialData0 : public Function<dim>
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
class DerivativeFunction0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        // return -std::sqrt(2.0) * std::sin(p[0]) * std::sin(p[1])
        //                         * std::sin(std::sqrt(2.0) * t);
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════
//  SOLUTION 2: Bessel function J0(alpha*r)cos(alpha*t)
// ═══════════════════════════════════════════════════════

// template <int dim>
// class Speed1 : public Function<dim>
// {
// public:
//     double value(const Point<dim> &p,
//                  const unsigned int component = 0) const override
//     {
//         (void)component;
//         return 1.0;
//     }
// };

template <int dim>
class AnalyticalSolution1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(alpha * t);
    }
};

template <int dim>
class RHSFunction1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)p; (void)component;
        return 0.0;
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
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(alpha * t);
    }
};

template <int dim>
class InitialData1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm());
    }
};

template <int dim>
class DerivativeFunction1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        // return -alpha * std::cyl_bessel_j(0, alpha * p.norm())
        //               * std::sin(alpha * t);
        return 0.0;
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
            s.initial_data        = std::make_unique<InitialData0<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction0<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = 2.0 * M_PI / std::sqrt(2.0);
            // s.speed               = std::make_unique<Speed0<dim>>();
            s.speed               = 1.0;
            break;
        case 1:
            {
            const double alpha = 2.4048255577;
            s.analytical_solution = std::make_unique<AnalyticalSolution1<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction1<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues1<dim>>();
            s.initial_data        = std::make_unique<InitialData1<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction1<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = 2.0 * M_PI / alpha;
            // s.speed               = std::make_unique<Speed1<dim>>();
            s.speed               = 1.0;
            break;
            }
        default:
            AssertThrow(false, ExcMessage("Unknown solution choice: "
                                          + std::to_string(choice)));
    }
    return s;
}
} // namespace Wave
} // namespace Combined

#endif // WAVE_SOLUTIONS_HPP