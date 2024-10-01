#include "ITURP_1057.h"
#include <cmath>
#include <limits>


ITURP_1057::ITURP_1057()
{

}

ITURP_1057::~ITURP_1057()
{

}

// ITU-R P.1057, Section 3
// inverse complementary cumulative distribution function
double ITURP_1057::Qinv(double p)
{
constexpr double a0 = 2.506628277459239;
constexpr double a1 = -30.66479806614716;
constexpr double a2 = 138.3577518672690;
constexpr double a3 = -275.9285104469687;
constexpr double a4 = 220.9460984245205;
constexpr double a5 = -39.69683028665376;
constexpr double b1 = -13.28068155288572;
constexpr double b2 = 66.80131188771972;
constexpr double b3 = -155.6989798598866;
constexpr double b4 = 161.5858368580409;
constexpr double b5 = -54.47609879822406;
constexpr double c0 = 2.938163982698783;
constexpr double c1 = 4.374664141464968;
constexpr double c2 = -2.549732539343734;
constexpr double c3 = -2.400758277161838;
constexpr double c4 = -0.3223964580411365;
constexpr double c5 = -0.007784894002430293;
constexpr double d1 = 3.754408661907416;
constexpr double d2 = 2.445134137142996;
constexpr double d3 = 0.3224671290700398;
constexpr double d4 = 0.007784695709041462;
double Uinv = std::numeric_limits<double>::quiet_NaN();
double t;
double pcopy = p;

    if( p > 0.5 && p < 1 )
        p = 1.0 - p;

    if( p > 0 && p <= 0.02425 )
    {
        t = sqrt(-2.0*log(p));
        Uinv = (c0 + c1*t + c2*t*t + c3*t*t*t + c4*t*t*t*t + c5*t*t*t*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t + d4*t*t*t*t);
    }
    else if( p > 0.02425 && p <= 0.5 )
    {
        t = (p-0.5)*(p-0.5);
        Uinv = (p-0.5) * (a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t) / (1.0 + b1*t + b2*t*t + b3*t*t*t + b4*t*t*t*t + b5*t*t*t*t*t);
    }

    if( pcopy < 0.5 )
        return -Uinv;
    else
        return Uinv;
}

// ITU-R P.1057, Section 3
// inverse cumulative distribution function
double ITURP_1057::Finv(double p)
{
    return Qinv(1.0-p);
}