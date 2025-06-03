/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_2108.h"
#include "ITURP_1057.h"
#include <cmath>
#include <algorithm>


ITURP_2108::ITURP_2108()
{

}

ITURP_2108::~ITURP_2108()
{

}

// ITU-R P.2108-1, Annex 1, Section 3.2
// f_GHz:            frequency (GHz) [0.5, 67.0]
// distance_km:      distance (km) [0.25, +inf.]
// location_percent: percentage of locations (%) ]0, 100.0[
// [return]:         clutter loss (dB)
//
// Notes:
//   Implementation validated using results from 'Clutter and BEL workbook_V1.xlsx' document: 
//   https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx
//
//   From CRC colleague (P.B., May 1st, 2023):
//     It seems to me that the sentences “0.25 km (for the correction to be applied at 
//     only one end of the path) and 1.0 km (for the correction to be applied at both 
//     ends of the path)” have been inserted for information only, not for action; these
//     facts are accounted for automatically in equations (3) to (6) of P.2108-1. There 
//     is no need to think about adding twice this correction in dB for paths longer than
//     1 km.  I agree that these sentences can be misleading.
double ITURP_2108::StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent)
{
double Lctt, Lctt2km;

	Lctt = _StatisticalClutterLossForTerrestrialPath(f_GHz, distance_km, location_percent);
	Lctt2km = _StatisticalClutterLossForTerrestrialPath(f_GHz, 2.0, location_percent);
	return std::min(Lctt, Lctt2km);
}

double ITURP_2108::_StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent)
{
double& f = f_GHz;
double& d = distance_km;
double& p = location_percent;
double log10_f, pow_Ll, pow_Ls;
double Lctt, Ll, Ls, sigma_cb;

	log10_f = log10(f);
	Ls = 32.98 + 23.9*log10(d) + 3.0*log10_f;
	Ll = -2.0*log10(pow(10.0, -5.0*log10_f-12.5)+pow(10.0, -16.5));
	pow_Ll = pow(10.0, -0.2*Ll);
	pow_Ls = pow(10.0, -0.2*Ls);
	sigma_cb = sqrt((16.0*pow_Ll+36.0*pow_Ls) / (pow_Ll+pow_Ls));
	Lctt = -5.0*log10(pow_Ll+pow_Ls) - sigma_cb*ITURP_1057::Qinv(p/100.0);
	return Lctt;
}
