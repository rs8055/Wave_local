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
    std::unique_ptr<Function<dim>> speed;
    std::unique_ptr<Function<dim>> analytical_solution;
    std::unique_ptr<Function<dim>> rhs_function;
    std::unique_ptr<Function<dim>> boundary_values;
    std::unique_ptr<Function<dim>> initial_data;
    std::unique_ptr<Function<dim>> derivative_function;
    double initial_time;
    double final_time;
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 0: sin(x)sin(y)cos(sqrt(2)t)
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
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════
//  SOLUTION 1: Bessel function J0(alpha*r)cos(alpha*t)
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
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════
//  SOLUTION 2: Bessel function J0(alpha*r)cos(2*alpha*t), with non-unity speed
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 4.0;
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
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(2 * alpha * t);
    }
};

template <int dim>
class RHSFunction2 : public Function<dim>
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
class BoundaryValues2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(2 * alpha * t);
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

template <int dim>
class DerivativeFunction2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  SOLUTION 3: sin(x)sin(y)cos(2t), with variable speed
// ═══════════════════════════════════════════════════════════════════════════

template <int dim>
class Speed3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1+p[0]*p[1];
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
        return std::sin(p[0]) * std::sin(p[1]) * std::cos((2.0) * t);
    }
};

template <int dim>
class RHSFunction3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)p; (void)component;
        const double t = this->get_time();
        return std::cos(2.0*t) * (
    (-4.0 + 2.0*(1+p[0]*p[1])) * std::sin(p[0]) * std::sin(p[1])
  - p[1] * std::cos(p[0]) * std::sin(p[1])
  - p[0] * std::sin(p[0]) * std::cos(p[1]));
    }
};

template <int dim>
class BoundaryValues3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return std::sin(p[0]) * std::sin(p[1]) * std::cos((2.0) * t);
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
        const double alpha = 2.4048255577;
        return std::sin(p[0]) * std::sin(p[1]);
    }
};

template <int dim>
class DerivativeFunction3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════
//  SOLUTION 4: Bessel function J0(alpha*r)cos(2*alpha*t), with non-unity speed
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 1+std::sin(p[0]);
    }
};

template <int dim>
class AnalyticalSolution4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(2 * alpha * t);
    }
};

template <int dim>
class RHSFunction4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;

        const double t     = this->get_time();
        const double alpha = 2.4048255577;

        const double r = p.norm();

        const double J0 = std::cyl_bessel_j(0, alpha * r);
        const double J1 = std::cyl_bessel_j(1, alpha * r);

        const double x_over_r = (r > 1e-12 ? p[0] / r : 0.0);

        const double term1 =
            alpha * alpha * (-3.0 + std::sin(p[0])) * J0;

        const double term2 =
            alpha * J1 * x_over_r * std::cos(p[0]);

        return (term1 + term2) * std::cos(2 * alpha * t);
    }
};

template <int dim>
class BoundaryValues4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
        const double alpha = 2.4048255577;
        return std::cyl_bessel_j(0, alpha * p.norm()) * std::cos(2 * alpha * t);
    }
};

template <int dim>
class InitialData4 : public Function<dim>
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
class DerivativeFunction4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t     = this->get_time();
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
            s.speed               = std::make_unique<Speed0<dim>>();
            break;
        case 1:
            {
            const double alpha    = 2.4048255577;
            s.analytical_solution = std::make_unique<AnalyticalSolution1<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction1<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues1<dim>>();
            s.initial_data        = std::make_unique<InitialData1<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction1<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = 2.0 * M_PI / alpha;
            s.speed               = std::make_unique<Speed1<dim>>();
            break;
            }
        case 2:
            {
            const double alpha    = 2.4048255577;
            s.analytical_solution = std::make_unique<AnalyticalSolution2<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction2<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues2<dim>>();
            s.initial_data        = std::make_unique<InitialData2<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction2<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = M_PI / alpha;
            s.speed               = std::make_unique<Speed2<dim>>();
            break;
            }
        case 3:
            {
            s.analytical_solution = std::make_unique<AnalyticalSolution3<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction3<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues3<dim>>();
            s.initial_data        = std::make_unique<InitialData3<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction3<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = M_PI;
            s.speed               = std::make_unique<Speed3<dim>>();
            break;
            }
        case 4:
            {
            const double alpha    = 2.4048255577;
            s.analytical_solution = std::make_unique<AnalyticalSolution4<dim>>();
            s.rhs_function        = std::make_unique<RHSFunction4<dim>>();
            s.boundary_values     = std::make_unique<BoundaryValues4<dim>>();
            s.initial_data        = std::make_unique<InitialData4<dim>>();
            s.derivative_function = std::make_unique<DerivativeFunction4<dim>>();
            s.initial_time        = 0.0;
            s.final_time          = M_PI / alpha;
            s.speed               = std::make_unique<Speed4<dim>>();
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