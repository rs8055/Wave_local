#ifndef HEAT_SOLUTIONS_HPP
#define HEAT_SOLUTIONS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <memory>
#include <cmath>

namespace Combined
{
namespace Heat         
{
using namespace dealii;

// ═══════════════════════════════════════════════════════
// SolutionSet struct — holds all function types together
// ═══════════════════════════════════════════════════════
template <int dim>
struct SolutionSet
{
    std::unique_ptr<Function<dim>> speed;
    std::unique_ptr<Function<dim>> speed_other;
    std::unique_ptr<Function<dim>> analytical_solution;
    std::unique_ptr<Function<dim>> level_set_function;
    std::unique_ptr<Function<dim>> rhs_function;
    std::unique_ptr<Function<dim>> interface_boundary_condition;
    std::unique_ptr<Function<dim>> outer_boundary_condition;
    std::unique_ptr<Function<dim>> interface_gradient_function;
    std::unique_ptr<Function<dim>> outer_gradient_function;
    std::unique_ptr<Function<dim>> initial_data;
    double initial_time;
    double final_time;
    // double speed;
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 0: sin(x)sin(y)exp(-2t)
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1.0;
    }
};

template <int dim>
class SpeedOther0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1.0;
    }
};

template <int dim>
class AnalyticalSolution0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]) * std::exp(-2 * t);
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
class InterfaceBoundaryCondition0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]) * std::exp(-2 * t);
    }
};

template <int dim>
class OuterBoundaryCondition0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]) * std::exp(-2 * t);
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


// ═══════════════════════════════════════════════════════
//  SOLUTION 1: (1 - 2/dim*(|x|^2 - 1)) * exp(-t)
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1.0;
    }
};

template <int dim>
class AnalyticalSolution1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return (1. - 2. / dim * (p.norm_square() - 1.)) * std::exp(-t);
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
        return (3. + 2./dim * (p.norm_square() - 1.)) * std::exp(-t);
    }
};

template <int dim>
class InterfaceBoundaryCondition1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return (1. - 2. / dim * (p.norm_square() - 1.)) * std::exp(-t);
    }
};

template <int dim>
class OuterBoundaryCondition1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();
        return (1. - 2. / dim * (p.norm_square() - 1.)) * std::exp(-t);
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
        return (1. - 2. / dim * (p.norm_square() - 1.));
    }
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 2: Bessel function J0(alpha*r)exp(-2*t)
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1.0;
    }
};

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
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::exp(-2 * t);
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
        return (std::pow(alpha,2)-2.0) * std::cyl_bessel_j(0, alpha * p.norm()) * std::exp(-2 * t);
    }
};

template <int dim>
class InterfaceBoundaryCondition2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::exp(-2 * t);
    }
};

template <int dim>
class OuterBoundaryCondition2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::exp(-2 * t);
    }
};

template <int dim>
class InitialData2 : public Function<dim>
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

// ═══════════════════════════════════════════════════════
//  SOLUTION 3: x^9y^8exp(-t)
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1.0;
    }
};

template <int dim>
class AnalyticalSolution3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return std::pow(p[0],9) * std::pow(p[1],8) * std::exp(-t);
    }
};

template <int dim>
class RHSFunction3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return -std::pow(p[0], 7.0) * std::pow(p[1], 6.0) * std::exp(-t) *
                    (std::pow(p[0], 2.0) * std::pow(p[1], 2.0) +
                     72 * std::pow(p[1], 2.0) + 56 * std::pow(p[0], 2.0));
    }
};

template <int dim>
class InterfaceBoundaryCondition3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return std::pow(p[0],9) * std::pow(p[1],8) * std::exp(-t);
    }
};

template <int dim>
class OuterBoundaryCondition3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return std::pow(p[0],9) * std::pow(p[1],8) * std::exp(-t);
    }
};

template <int dim>
class InitialData3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return std::pow(p[0],9) * std::pow(p[1],8);
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
            s.analytical_solution          = std::make_unique<AnalyticalSolution0<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction0<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition0<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition0<dim>>();
            s.initial_data                 = std::make_unique<InitialData0<dim>>();
            s.initial_time                 = 0.0;
            s.final_time                   = 1.0;
            s.speed                        = std::make_unique<Speed0<dim>>();
            s.speed_other                  = std::make_unique<SpeedOther0<dim>>();
            break;
        case 1:
            s.analytical_solution          = std::make_unique<AnalyticalSolution1<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction1<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition1<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition1<dim>>();
            s.initial_data                 = std::make_unique<InitialData1<dim>>();
            s.initial_time                 = 0.0;
            s.final_time                   = 1.0;
            s.speed                        = std::make_unique<Speed1<dim>>();
            break;
        case 2:
            s.analytical_solution          = std::make_unique<AnalyticalSolution2<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction2<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition2<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition2<dim>>();
            s.initial_data                 = std::make_unique<InitialData2<dim>>();
            s.initial_time                 = 0.0;
            s.final_time                   = 1.0;
            s.speed                        = std::make_unique<Speed2<dim>>();
            break;
        case 3:
            s.analytical_solution          = std::make_unique<AnalyticalSolution3<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction3<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition3<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition3<dim>>();
            s.initial_data                 = std::make_unique<InitialData3<dim>>();
            s.initial_time                 = 0.0;
            s.final_time                   = 1.0;
            s.speed                        = std::make_unique<Speed3<dim>>();
            break;
        default:
            AssertThrow(false, ExcMessage("Unknown solution choice: "
                                          + std::to_string(choice)));
    }
    return s;
}
} // namespace Heat
} // namespace Combined

#endif // HEAT_SOLUTIONS_HPP