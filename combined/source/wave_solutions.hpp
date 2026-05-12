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
    std::vector<std::unique_ptr<Function<dim>>> speed;          //Store the function as c^2, i.e. if your speed is 2, write 4 in speed.
    std::unique_ptr<Function<dim>> analytical_solution;
    std::vector<std::unique_ptr<Function<dim>>> level_set_functions;
    std::unique_ptr<Function<dim>> rhs_function;
    std::unique_ptr<Function<dim>> interface_boundary_condition;
    std::unique_ptr<Function<dim>> outer_boundary_condition;
    std::unique_ptr<Function<dim>> interface_gradient_function;
    std::unique_ptr<Function<dim>> outer_gradient_function;
    std::vector<std::unique_ptr<Function<dim>>> initial_data;
    std::vector<std::unique_ptr<Function<dim>>> derivative_initial_data;
    double initial_time;
    double final_time;
};

// ═══════════════════════════════════════════════════════
//  Level Set functions
// ═══════════════════════════════════════════════════════
// ── Case 1: Straight line interface x=0 ─────────────────────────
template <int dim>
class StraightLineInterface : public Function<dim>
  {
  public:
      StraightLineInterface(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
          switch (domain_index)
          {
              case 0: return p[0];       
              case 1: return -(p[0]);    
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

// ── Case 2: Aligned line interface x+y=0 ─────────────────────────
template <int dim>
  class AlignedInterface : public Function<dim>
  {
  public:
      AlignedInterface(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
          switch (domain_index)
          {
              case 0: return p[0] + p[1];       
              case 1: return -(p[0] + p[1]);    
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

  // ── Case 3: Multiple Interface ─────────────────────────
template <int dim>
  class MultipleInterface1 : public Function<dim>
  {
  public:
      MultipleInterface1(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
          switch (domain_index)
          {
              case 0: return p[0]+0.5-1e-6;       
              case 1: return std::max(-p[0]-0.5+1e-6, p[0]-0.5-1e-6); 
              case 2: return 0.5-p[0]+1e-6;    
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

    // ── Case 4: Multiple Interface ─────────────────────────
template <int dim>
  class MultipleInterface2 : public Function<dim>
  {
  public:
      MultipleInterface2(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
          switch (domain_index)
          {
              case 0: return p[0]+p[1]+1;       
              case 1: return std::max(-p[0]-p[1]-1, p[0]+p[1]-1); 
              case 2: return -p[0]-p[1]+1;    
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };


// ── Case 5: Ellipse interface ────────────────────────────────────
template <int dim>
class EllipseInterface : public Function<dim>
{
public:
  EllipseInterface(const double a = 1.0, const double b = 0.5)
    : Function<dim>(), a(a), b(b) {}

  double value(const Point<dim> &p,
               const unsigned int = 0) const override
  {
    return p[0]*p[0]/(a*a) + p[1]*p[1]/(b*b) - 1.0;
  }
private:
  double a, b;
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
  class InterfaceGradientSolution0 : public Function<dim>
  {
  public:
    Tensor<1,dim> gradient(const Point<dim>  &point,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double t = this->get_time();

        Tensor<1, dim> grad;

        grad[0] = std::cos(point[0]) * std::sin(point[1]) * std::cos(std::sqrt(2.0) * t);
        grad[1] = std::sin(point[0]) * std::cos(point[1]) * std::cos(std::sqrt(2.0) * t);

        if (dim == 3)
        grad[2] = 0.0;

        return grad;
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
        return std::sin(p[0]) * std::sin(p[1]) * std::cos(std::sqrt(2.0) * t);
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
class DerivativeInitialData0 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        // const double t = this->get_time();
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
class SpeedOther1 : public Function<dim>
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
class InterfaceBoundaryCondition1 : public Function<dim>
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
class OuterBoundaryCondition1 : public Function<dim>
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
class DerivativeInitialData1 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        // const double t     = this->get_time();
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
class SpeedOther2 : public Function<dim>
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
class InterfaceBoundaryCondition2 : public Function<dim>
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
class OuterBoundaryCondition2 : public Function<dim>
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
class DerivativeInitialData2 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        // const double t     = this->get_time();
        return 0.0;
    }
};


// ═══════════════════════════════════════════════════════
//  SOLUTION 3: Bessel function J0(alpha*r)cos(2*alpha*t), with non-unity speed
// ═══════════════════════════════════════════════════════

template <int dim>
class Speed3 : public Function<dim>
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
class SpeedOther3 : public Function<dim>
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
class AnalyticalSolution3 : public Function<dim>
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
class RHSFunction3 : public Function<dim>
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
class InterfaceBoundaryCondition3 : public Function<dim>
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
class OuterBoundaryCondition3 : public Function<dim>
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
class InitialData3 : public Function<dim>
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
class DerivativeInitialData3 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        // const double t     = this->get_time();
        return 0.0;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  SOLUTION 4: Interface Problem with constant but different speed both sides
// ═══════════════════════════════════════════════════════════════════════════

template <int dim>
class Speed4 : public Function<dim>
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
class SpeedOther4 : public Function<dim>
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
class AnalyticalSolution4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        const double c = (p[0] < 0) ? 1 : 4;
        return (1.0/c) * p[0] * std::sin(M_PI * p[1]) * std::cos(t);
      }
};

template <int dim>
class RHSFunction4 : public Function<dim>
{
public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
      {
        const double t = this->get_time();
        const double c = (p[0] < 0) ? 1 : 4;
        return (c*M_PI*M_PI - 1.0)/c * p[0] 
               * std::sin(M_PI * p[1]) * std::cos(t);
      }
};

template <int dim>
class InterfaceBoundaryCondition4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        const double c = (p[0] < 0) ? 1 : 4;
        return (1.0/c) * p[0] * std::sin(M_PI * p[1]) * std::cos(t);
        // return 0.0;
      }
};

template <int dim>
class OuterBoundaryCondition4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        const double c = (p[0] < 0) ? 1 : 4;
        return (1.0/c) * p[0] * std::sin(M_PI * p[1]) * std::cos(t);
      }
};

template <int dim>
class InitialData4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        // const double t = this->get_time();
        const double c = 1;
        return (1.0/c) * p[0] * std::sin(M_PI * p[1]);
      }
};

template <int dim>
class InitialDataOther4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double c = 4;
        return (1.0/c) * p[0] * std::sin(M_PI * p[1]);
      }
};

template <int dim>
class DerivativeInitialData4 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 0.0;
    }
};


// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SOLUTION 5: Interface Problem with constant but different speed both sides from SIAM Journal of Sci. Comp.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

template <int dim>
class Speed5 : public Function<dim>
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
class SpeedOther5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 2.0;
    }
};

template <int dim>
class AnalyticalSolution5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + std::cos(-p[1] - t);
        }
        else
        {
            return (2) * std::cos((0.5) *(p[0] - p[1]) -  t);
        }
      }
};

template <int dim>
class RHSFunction5 : public Function<dim>
{
public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
      {
        // const double t = this->get_time();
        return 0.0;
      }
};

template <int dim>
class InterfaceBoundaryCondition5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + std::cos(-p[1] - t);
        }
        else
        {
            return (2) * std::cos((0.5) *(p[0] - p[1]) -  t);
        }
      }
};

template <int dim>
class OuterBoundaryCondition5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + std::cos(-p[1] - t);
        }
        else
        {
            return (2) * std::cos((0.5) *(p[0] - p[1]) -  t);
        }
      }
};

template <int dim>
class InitialData5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        return std::cos(p[0]) + std::cos(-p[1]);
      }
};

template <int dim>
class InitialDataOther5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        return (2) * std::cos((0.5) *(p[0] - p[1]));
      }
};

template <int dim>
class DerivativeInitialData5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return std::sin(p[0]) + std::sin(-p[1]);
    }
};

template <int dim>
class DerivativeInitialDataOther5 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return (2) * std::sin((0.5) *(p[0] - p[1]));
    }
};

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SOLUTION 6: Interface Problem with constant but different speed both sides from SIAM Journal of Sci. Comp.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

template <int dim>
class Speed6 : public Function<dim>
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
class SpeedOther6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 0.25;
    }
};

template <int dim>
class AnalyticalSolution6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        // const double k_1 = std::sqrt(7.0/2.0);
        const double k_2 = (1-(0.25)*std::sqrt(7))/(1+(0.25)*std::sqrt(7));
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + k_2 * std::cos(-p[1] - t);
        }
        else
        {
            return (2.0/(1+(0.25)*std::sqrt(7))) * std::cos(((std::sqrt(7)+1.0)/2) * p[0] + ((std::sqrt(7)-1.0)/2) * p[1] -  t);
        }
      }
};

template <int dim>
class RHSFunction6 : public Function<dim>
{
public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
      {
        // const double t = this->get_time();
        return 0.0;
      }
};

template <int dim>
class InterfaceBoundaryCondition6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        // const double k_1 = std::sqrt(7.0/2.0);
        const double k_2 = (1-(0.25)*std::sqrt(7))/(1+(0.25)*std::sqrt(7));
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + k_2 * std::cos(-p[1] - t);
        }
        else
        {
            return (2.0/(1+(0.25)*std::sqrt(7))) * std::cos(((std::sqrt(7)+1.0)/2) * p[0] + ((std::sqrt(7)-1.0)/2) * p[1] -  t);
        }
      }
};

template <int dim>
class OuterBoundaryCondition6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        // const double k_1 = std::sqrt(7.0/2.0);
        const double k_2 = (1-(0.25)*std::sqrt(7))/(1+(0.25)*std::sqrt(7));
        if(p[0]+p[1]<0)
        {
            return std::cos(p[0] - t) + k_2 * std::cos(-p[1] - t);
        }
        else
        {
            return (2.0/(1+(0.25)*std::sqrt(7))) * std::cos(((std::sqrt(7)+1.0)/2) * p[0] + ((std::sqrt(7)-1.0)/2) * p[1] -  t);
        }
      }
};

template <int dim>
class InitialData6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double k_2 = (1-(0.25)*std::sqrt(7))/(1+(0.25)*std::sqrt(7));
        return std::cos(p[0]) + k_2 * std::cos(-p[1]);
      }
};

template <int dim>
class InitialDataOther6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        return (2.0/(1+(0.25)*std::sqrt(7))) * std::cos(((std::sqrt(7)+1.0)/2) * p[0] + ((std::sqrt(7)-1.0)/2) * p[1]);
      }
};

template <int dim>
class DerivativeInitialData6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        const double k_2 = (1-(0.25)*std::sqrt(7))/(1+(0.25)*std::sqrt(7));
        return std::sin(p[0]) + k_2 * std::sin(-p[1]);
    }
};

template <int dim>
class DerivativeInitialDataOther6 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return (2.0/(1+(0.25)*std::sqrt(7))) * std::sin(((std::sqrt(7)+1.0)/2) * p[0] + ((std::sqrt(7)-1.0)/2) * p[1]);
    }
};

// ═════════════════════════════════════════════════════════════════════════
//  SOLUTION 7: Multiple Interface Problem with constant but different speed
// ═════════════════════════════════════════════════════════════════════════
template <int dim>
  class Speed7: public Function<dim>
  {
  public:
      Speed7(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
        switch (domain_index)
          {
              case 0: return 1.0;       
              case 1: return 0.1; 
              case 2: return 1.0; 
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

template <int dim>
class AnalyticalSolution7 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<-1)
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t);
        }
        else if ((p[0] + p[1] >= -1) && (p[0] + p[1] <= 1))
        {
            return (std::cos(std::sqrt(10)*(p[0]+p[1])))*std::cos(std::sqrt(2)*t); 
        }
        else
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t); 
        }
      }
};

template <int dim>
class RHSFunction7 : public Function<dim>
{
public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
      {
        // const double t = this->get_time();
        return 0.0;
      }
};

template <int dim>
class InterfaceBoundaryCondition7 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<-1)
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t);
        }
        else if ((p[0] + p[1] >= -1) && (p[0] + p[1] <= 1))
        {
            return (std::cos(std::sqrt(10)*(p[0]+p[1])))*std::cos(std::sqrt(2)*t); 
        }
        else
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t); 
        }
      }
};

template <int dim>
class OuterBoundaryCondition7 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
      {
        const double t = this->get_time();
        if(p[0]+p[1]<-1)
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t);
        }
        else if ((p[0] + p[1] >= -1) && (p[0] + p[1] <= 1))
        {
            return (std::cos(std::sqrt(10)*(p[0]+p[1])))*std::cos(std::sqrt(2)*t); 
        }
        else
        {
            return ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t); 
        }
      }
};

template <int dim>
  class InitialData7 : public Function<dim>
  {
  public:
      InitialData7(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
          switch (domain_index)
          {
              case 0: return (std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]);       
              case 1: return std::cos(std::sqrt(10)*(p[0]+p[1])); 
              case 2: return (std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]);   
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

template <int dim>
class DerivativeInitialData7 : public Function<dim>
{
public:
    double value(const Point<dim> &p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        return 0.0;
    }
};


// ═════════════════════════════════════════════════════════════════════════
//  SOLUTION 8: Multiple Interface Problem with constant but different speed
// ═════════════════════════════════════════════════════════════════════════
template <int dim>
  class Speed8: public Function<dim>
  {
  public:
      Speed8(const unsigned int domain_index)
          : Function<dim>(1), domain_index(domain_index) {}

      double value(const Point<dim> &p, unsigned int = 0) const override
      {
        switch (domain_index)
          {
              case 0: return 0.25;   
              case 1: return 1.0; 
              // add more cases here
              default: AssertThrow(false, ExcMessage("Unknown domain index"));
                      return 0.0;
          }
      }

  private:
      const unsigned int domain_index;
  };

template <int dim>
class AnalyticalSolution8 : public Function<dim>
{
public:
  AnalyticalSolution8() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const double t = this->get_time();
    const std::complex<double> I(0.0, 1.0);
    const double omega = 2.0 * M_PI;
    const double c = 1.0, c_p = 0.5;
    const double g = omega / c, gp = omega / c_p;

    const double r = std::max(p.norm(), 1e-12);
    const double theta = std::atan2(p[1], p[0]);

    std::complex<double> u_spatial = 0;

    // Summation of modes n=0 to 100
    for (int n = 0; n <= 100; ++n)
    {
        std::complex<double> An, Bn;
        calculate_coeffs(n, g, gp, c, c_p, An, Bn);
        
        std::complex<double> factor = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);
        
        if (r > 1.0) {
            u_spatial += (factor * std::cyl_bessel_j(n, g * r) + An * hankel2(n, g * r)) * std::cos(n * theta);
        } else {
            u_spatial += Bn * std::cyl_bessel_j(n, gp * r) * std::cos(n * theta);
        }
    }
    return (u_spatial * std::exp(I * omega * t)).real();
  }

private:
  std::complex<double> hankel2(int n, double z) const {
    return {std::cyl_bessel_j(n, z), -std::cyl_neumann(n, z)};
  }

  void calculate_coeffs(int n, double g, double gp, double c, double cp, 
                        std::complex<double> &An, std::complex<double> &Bn) const {
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> pre = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);

    double jn_g = std::cyl_bessel_j(n, g), jn_gp = std::cyl_bessel_j(n, gp);
    std::complex<double> hn_g = hankel2(n, g);

    // Derivatives via recurrence: J'n(x) = 0.5*(Jn-1 - Jn+1)
    auto get_dJ = [](int order, double x) {
        if (order == 0) return -std::cyl_bessel_j(1, x);
        return 0.5 * (std::cyl_bessel_j(order - 1, x) - std::cyl_bessel_j(order + 1, x));
    };

    auto get_dH = [this](int order, double x) {
        if (order == 0) return -this->hankel2(1, x);
        return 0.5 * (this->hankel2(order - 1, x) - this->hankel2(order + 1, x));
    };

    std::complex<double> djn_g  = get_dJ(n, g);
    std::complex<double> djn_gp = get_dJ(n, gp);
    std::complex<double> dhn_g  = get_dH(n, g);

    An = pre * (cp*cp*gp*djn_gp*jn_g - c*c*g*djn_g*jn_gp) / (c*c*g*dhn_g*jn_gp - cp*cp*gp*djn_gp*hn_g);
    Bn = (pre * jn_g + An * hn_g) / jn_gp;
  }
};

template <int dim>
class RHSFunction8 : public Function<dim>
{
public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
      {
        // const double t = this->get_time();
        return 0.0;
      }
};

template <int dim>
class InitialData8 : public Function<dim>
{
public:
  InitialData8() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const std::complex<double> I(0.0, 1.0);
    const double omega = 2.0 * M_PI;
    const double c = 1.0, c_p = 0.5;
    const double g = omega / c, gp = omega / c_p;

    const double r = std::max(p.norm(), 1e-12);
    const double theta = std::atan2(p[1], p[0]);

    std::complex<double> u_spatial = 0;

    // Summation of modes n=0 to 100
    for (int n = 0; n <= 100; ++n)
    {
        std::complex<double> An, Bn;
        calculate_coeffs(n, g, gp, c, c_p, An, Bn);
        
        u_spatial += Bn * std::cyl_bessel_j(n, gp * r) * std::cos(n * theta);
    }
    return (u_spatial).real();
  }

private:
  std::complex<double> hankel2(int n, double z) const {
    return {std::cyl_bessel_j(n, z), -std::cyl_neumann(n, z)};
  }

  void calculate_coeffs(int n, double g, double gp, double c, double cp, 
                        std::complex<double> &An, std::complex<double> &Bn) const {
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> pre = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);

    double jn_g = std::cyl_bessel_j(n, g), jn_gp = std::cyl_bessel_j(n, gp);
    std::complex<double> hn_g = hankel2(n, g);

    // Derivatives via recurrence: J'n(x) = 0.5*(Jn-1 - Jn+1)
    auto get_dJ = [](int order, double x) {
        if (order == 0) return -std::cyl_bessel_j(1, x);
        return 0.5 * (std::cyl_bessel_j(order - 1, x) - std::cyl_bessel_j(order + 1, x));
    };

    auto get_dH = [this](int order, double x) {
        if (order == 0) return -this->hankel2(1, x);
        return 0.5 * (this->hankel2(order - 1, x) - this->hankel2(order + 1, x));
    };

    std::complex<double> djn_g  = get_dJ(n, g);
    std::complex<double> djn_gp = get_dJ(n, gp);
    std::complex<double> dhn_g  = get_dH(n, g);

    An = pre * (cp*cp*gp*djn_gp*jn_g - c*c*g*djn_g*jn_gp) / (c*c*g*dhn_g*jn_gp - cp*cp*gp*djn_gp*hn_g);
    Bn = (pre * jn_g + An * hn_g) / jn_gp;
  }
};

template <int dim>
class InitialDataOther8 : public Function<dim>
{
public:
  InitialDataOther8() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const std::complex<double> I(0.0, 1.0);
    const double omega = 2.0 * M_PI;
    const double c = 1.0, c_p = 0.5;
    const double g = omega / c, gp = omega / c_p;

    const double r = std::max(p.norm(), 1e-12);
    const double theta = std::atan2(p[1], p[0]);

    std::complex<double> u_spatial = 0;

    // Summation of modes n=0 to 100
    for (int n = 0; n <= 100; ++n)
    {
        std::complex<double> An, Bn;
        calculate_coeffs(n, g, gp, c, c_p, An, Bn);
        
        std::complex<double> factor = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);
        u_spatial += (factor * std::cyl_bessel_j(n, g * r) + An * hankel2(n, g * r)) * std::cos(n * theta);
    }
    return (u_spatial).real();
  }

private:
  std::complex<double> hankel2(int n, double z) const {
    return {std::cyl_bessel_j(n, z), -std::cyl_neumann(n, z)};
  }

  void calculate_coeffs(int n, double g, double gp, double c, double cp, 
                        std::complex<double> &An, std::complex<double> &Bn) const {
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> pre = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);

    double jn_g = std::cyl_bessel_j(n, g), jn_gp = std::cyl_bessel_j(n, gp);
    std::complex<double> hn_g = hankel2(n, g);

    // Derivatives via recurrence: J'n(x) = 0.5*(Jn-1 - Jn+1)
    auto get_dJ = [](int order, double x) {
        if (order == 0) return -std::cyl_bessel_j(1, x);
        return 0.5 * (std::cyl_bessel_j(order - 1, x) - std::cyl_bessel_j(order + 1, x));
    };

    auto get_dH = [this](int order, double x) {
        if (order == 0) return -this->hankel2(1, x);
        return 0.5 * (this->hankel2(order - 1, x) - this->hankel2(order + 1, x));
    };

    std::complex<double> djn_g  = get_dJ(n, g);
    std::complex<double> djn_gp = get_dJ(n, gp);
    std::complex<double> dhn_g  = get_dH(n, g);

    An = pre * (cp*cp*gp*djn_gp*jn_g - c*c*g*djn_g*jn_gp) / (c*c*g*dhn_g*jn_gp - cp*cp*gp*djn_gp*hn_g);
    Bn = (pre * jn_g + An * hn_g) / jn_gp;
  }
};

template <int dim>
class DerivativeInitialData8 : public Function<dim>
{
public:
  DerivativeInitialData8() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const std::complex<double> I(0.0, 1.0);
    const double omega = 2.0 * M_PI;
    const double c = 1.0, c_p = 0.5;
    const double g = omega / c, gp = omega / c_p;

    const double r = std::max(p.norm(), 1e-12);
    const double theta = std::atan2(p[1], p[0]);

    std::complex<double> u_spatial = 0;

    // Summation of modes n=0 to 100
    for (int n = 0; n <= 100; ++n)
    {
        std::complex<double> An, Bn;
        calculate_coeffs(n, g, gp, c, c_p, An, Bn);
        
        u_spatial += Bn * std::cyl_bessel_j(n, gp * r) * std::cos(n * theta);
    }
    return (I * omega * u_spatial).real();
  }

private:
  std::complex<double> hankel2(int n, double z) const {
    return {std::cyl_bessel_j(n, z), -std::cyl_neumann(n, z)};
  }

  void calculate_coeffs(int n, double g, double gp, double c, double cp, 
                        std::complex<double> &An, std::complex<double> &Bn) const {
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> pre = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);

    double jn_g = std::cyl_bessel_j(n, g), jn_gp = std::cyl_bessel_j(n, gp);
    std::complex<double> hn_g = hankel2(n, g);

    // Derivatives via recurrence: J'n(x) = 0.5*(Jn-1 - Jn+1)
    auto get_dJ = [](int order, double x) {
        if (order == 0) return -std::cyl_bessel_j(1, x);
        return 0.5 * (std::cyl_bessel_j(order - 1, x) - std::cyl_bessel_j(order + 1, x));
    };

    auto get_dH = [this](int order, double x) {
        if (order == 0) return -this->hankel2(1, x);
        return 0.5 * (this->hankel2(order - 1, x) - this->hankel2(order + 1, x));
    };

    std::complex<double> djn_g  = get_dJ(n, g);
    std::complex<double> djn_gp = get_dJ(n, gp);
    std::complex<double> dhn_g  = get_dH(n, g);

    An = pre * (cp*cp*gp*djn_gp*jn_g - c*c*g*djn_g*jn_gp) / (c*c*g*dhn_g*jn_gp - cp*cp*gp*djn_gp*hn_g);
    Bn = (pre * jn_g + An * hn_g) / jn_gp;
  }
};

template <int dim>
class DerivativeInitialDataOther8 : public Function<dim>
{
public:
  DerivativeInitialDataOther8() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const std::complex<double> I(0.0, 1.0);
    const double omega = 2.0 * M_PI;
    const double c = 1.0, c_p = 0.5;
    const double g = omega / c, gp = omega / c_p;

    const double r = std::max(p.norm(), 1e-12);
    const double theta = std::atan2(p[1], p[0]);

    std::complex<double> u_spatial = 0;

    // Summation of modes n=0 to 100
    for (int n = 0; n <= 100; ++n)
    {
        std::complex<double> An, Bn;
        calculate_coeffs(n, g, gp, c, c_p, An, Bn);
        
        std::complex<double> factor = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);
        u_spatial += (factor * std::cyl_bessel_j(n, g * r) + An * hankel2(n, g * r)) * std::cos(n * theta);
    }
    return (I * omega * u_spatial).real();
  }

private:
  std::complex<double> hankel2(int n, double z) const {
    return {std::cyl_bessel_j(n, z), -std::cyl_neumann(n, z)};
  }

  void calculate_coeffs(int n, double g, double gp, double c, double cp, 
                        std::complex<double> &An, std::complex<double> &Bn) const {
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> pre = (n == 0) ? 1.0 : 2.0 * std::pow(-I, n);

    double jn_g = std::cyl_bessel_j(n, g), jn_gp = std::cyl_bessel_j(n, gp);
    std::complex<double> hn_g = hankel2(n, g);

    // Derivatives via recurrence: J'n(x) = 0.5*(Jn-1 - Jn+1)
    auto get_dJ = [](int order, double x) {
        if (order == 0) return -std::cyl_bessel_j(1, x);
        return 0.5 * (std::cyl_bessel_j(order - 1, x) - std::cyl_bessel_j(order + 1, x));
    };

    auto get_dH = [this](int order, double x) {
        if (order == 0) return -this->hankel2(1, x);
        return 0.5 * (this->hankel2(order - 1, x) - this->hankel2(order + 1, x));
    };

    std::complex<double> djn_g  = get_dJ(n, g);
    std::complex<double> djn_gp = get_dJ(n, gp);
    std::complex<double> dhn_g  = get_dH(n, g);

    An = pre * (cp*cp*gp*djn_gp*jn_g - c*c*g*djn_g*jn_gp) / (c*c*g*dhn_g*jn_gp - cp*cp*gp*djn_gp*hn_g);
    Bn = (pre * jn_g + An * hn_g) / jn_gp;
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
            s.level_set_functions.push_back(std::make_unique<MultipleInterface1<dim>>(0));
            s.level_set_functions.push_back(std::make_unique<MultipleInterface1<dim>>(1));
            s.level_set_functions.push_back(std::make_unique<MultipleInterface1<dim>>(2));
            s.interface_gradient_function  = std::make_unique<InterfaceGradientSolution0<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction0<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition0<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition0<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData0<dim>>());
            s.initial_data.push_back(std::make_unique<InitialData0<dim>>());
            s.initial_data.push_back(std::make_unique<InitialData0<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData0<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData0<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData0<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0 * M_PI / std::sqrt(2.0);
            s.speed.push_back(std::make_unique<Speed0<dim>>());
            s.speed.push_back(std::make_unique<Speed0<dim>>());
            s.speed.push_back(std::make_unique<Speed0<dim>>());
            break;
        case 1:
            {
            const double alpha             = 2.4048255577;
            s.analytical_solution          = std::make_unique<AnalyticalSolution1<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction1<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition1<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition1<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData1<dim>>());
            s.initial_data.push_back(std::make_unique<InitialData1<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData1<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData1<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0 * M_PI / alpha;
            s.speed.push_back(std::make_unique<Speed1<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther1<dim>>());
            break;
            }
        case 2:
            {
            const double alpha             = 2.4048255577;
            s.analytical_solution          = std::make_unique<AnalyticalSolution2<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction2<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition2<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition2<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData2<dim>>());
            s.initial_data.push_back(std::make_unique<InitialData2<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData2<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData2<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = M_PI / alpha;
            s.speed.push_back(std::make_unique<Speed2<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther2<dim>>());
            break;
            }
        case 3:
            {
            const double alpha              = 2.4048255577;
            s.analytical_solution           = std::make_unique<AnalyticalSolution3<dim>>();
            s.rhs_function                  = std::make_unique<RHSFunction3<dim>>();
            s.interface_boundary_condition  = std::make_unique<InterfaceBoundaryCondition3<dim>>();
            s.outer_boundary_condition      = std::make_unique<OuterBoundaryCondition3<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData3<dim>>());
            s.initial_data.push_back(std::make_unique<InitialData3<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData3<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData3<dim>>());
            s.initial_time                  = 0.0;
            s.final_time                    = M_PI / alpha;
            s.speed.push_back(std::make_unique<Speed3<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther3<dim>>());
            break;
            }
        case 4:
            s.analytical_solution          = std::make_unique<AnalyticalSolution4<dim>>();
            s.level_set_functions.push_back(std::make_unique<StraightLineInterface<dim>>(0));
            s.level_set_functions.push_back(std::make_unique<StraightLineInterface<dim>>(1));
            s.rhs_function                 = std::make_unique<RHSFunction4<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition4<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition4<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData4<dim>>());
            s.initial_data.push_back(std::make_unique<InitialDataOther4<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData4<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData4<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0 * M_PI ;
            s.speed.push_back(std::make_unique<Speed4<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther4<dim>>());
            break;
        case 5:
            s.analytical_solution          = std::make_unique<AnalyticalSolution5<dim>>();
            s.level_set_functions.push_back(std::make_unique<AlignedInterface<dim>>(0));
            s.level_set_functions.push_back(std::make_unique<AlignedInterface<dim>>(1));
            s.rhs_function                 = std::make_unique<RHSFunction5<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition5<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition5<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData5<dim>>());
            s.initial_data.push_back(std::make_unique<InitialDataOther5<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData5<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialDataOther5<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0;
            s.speed.push_back(std::make_unique<Speed5<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther5<dim>>());
            break;
        case 6:
            s.analytical_solution          = std::make_unique<AnalyticalSolution6<dim>>();
            s.level_set_functions.push_back(std::make_unique<AlignedInterface<dim>>(0));
            s.level_set_functions.push_back(std::make_unique<AlignedInterface<dim>>(1));
            s.rhs_function                 = std::make_unique<RHSFunction6<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition6<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition6<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData6<dim>>());
            s.initial_data.push_back(std::make_unique<InitialDataOther6<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData6<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialDataOther6<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0;
            s.speed.push_back(std::make_unique<Speed6<dim>>());
            s.speed.push_back(std::make_unique<SpeedOther6<dim>>());
            break;
        case 7:
            s.analytical_solution          = std::make_unique<AnalyticalSolution7<dim>>();
            s.level_set_functions.push_back(std::make_unique<MultipleInterface2<dim>>(0));
            s.level_set_functions.push_back(std::make_unique<MultipleInterface2<dim>>(1));
            s.level_set_functions.push_back(std::make_unique<MultipleInterface2<dim>>(2));
            s.rhs_function                 = std::make_unique<RHSFunction7<dim>>();
            s.interface_boundary_condition = std::make_unique<InterfaceBoundaryCondition7<dim>>();
            s.outer_boundary_condition     = std::make_unique<OuterBoundaryCondition7<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData7<dim>>(0));
            s.initial_data.push_back(std::make_unique<InitialData7<dim>>(1));
            s.initial_data.push_back(std::make_unique<InitialData7<dim>>(2));
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData7<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData7<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData7<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0 * M_PI / std::sqrt(2.0);
            s.speed.push_back(std::make_unique<Speed7<dim>>(0));
            s.speed.push_back(std::make_unique<Speed7<dim>>(1));
            s.speed.push_back(std::make_unique<Speed7<dim>>(2));
            break;
        case 8:
            s.analytical_solution          = std::make_unique<AnalyticalSolution8<dim>>();
            s.rhs_function                 = std::make_unique<RHSFunction8<dim>>();
            s.interface_boundary_condition = std::make_unique<AnalyticalSolution8<dim>>();
            s.outer_boundary_condition     = std::make_unique<AnalyticalSolution8<dim>>();
            s.initial_data.push_back(std::make_unique<InitialData8<dim>>());
            s.initial_data.push_back(std::make_unique<InitialDataOther8<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialData8<dim>>());
            s.derivative_initial_data.push_back(std::make_unique<DerivativeInitialDataOther8<dim>>());
            s.initial_time                 = 0.0;
            s.final_time                   = 2.0;
            s.speed.push_back(std::make_unique<Speed8<dim>>(0));
            s.speed.push_back(std::make_unique<Speed8<dim>>(1));
            break;
        default:
            AssertThrow(false, ExcMessage("Unknown solution choice: "
                                          + std::to_string(choice)));
    }
    return s;
}
} // namespace Wave
} // namespace Combined

#endif // WAVE_SOLUTIONS_HPP