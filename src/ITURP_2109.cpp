#include "ITURP_2109.h"
#include "ITURP_1057.h"
#include <cmath>


ITURP_2109::ITURP_2109()
{

}

ITURP_2109::~ITURP_2109()
{

}

const double ITURP_2109::_BLDG_COEFFS[2][9] = {{12.64, 3.72, 0.96, 9.6, 2.0, 9.1, -3.0, 4.5, -2.0},
                                               {28.19, -3.00, 8.48, 13.5, 3.8, 27.8, -2.9, 9.4, -2.1}};

// ITU-R P.2109-2, Annex 1, Section 3
// f_GHz:               frequency (GHz) [~0.08, 100]
// probability_percent: the probability with which the loss is not exceeded (%) ]0, 100[
// bldgType:            building type (traditional or thermally efficient)
// elevAngle_degrees:   elevation angle of the path at the building façade (degrees above the horizontal) ]-90, 90[
// [return]:            building entry loss (dB)
double ITURP_2109::BuildingEntryLoss(double f_GHz, double probability_percent, BuildingType bldgType, double elevAngle_degrees)
{
int i = bldgType-TRADITIONAL;
double r = _BLDG_COEFFS[i][0];
double s = _BLDG_COEFFS[i][1];
double t = _BLDG_COEFFS[i][2];
double u = _BLDG_COEFFS[i][3];
double v = _BLDG_COEFFS[i][4];
double w = _BLDG_COEFFS[i][5];
double x = _BLDG_COEFFS[i][6];
double y = _BLDG_COEFFS[i][7];
double z = _BLDG_COEFFS[i][8];

double C = -3.0;
double logf = log10(f_GHz);
double Lh = r + s*logf + t*logf*logf;
double Le = 0.212*std::abs(elevAngle_degrees);
double mu1 = Lh + Le;
double mu2 = w + x*logf;
double sigma1 = u + v*logf;
double sigma2 = y + z*logf;
double FinvP = ITURP_1057::Finv(probability_percent/100.0);
double AP = FinvP*sigma1 + mu1;
double BP = FinvP*sigma2 + mu2;

double LomniBEL = 10.0*log10(pow(10.0, 0.1*AP) + pow(10.0, 0.1*BP) + pow(10.0, 0.1*C));

    return LomniBEL;
}
