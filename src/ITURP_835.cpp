/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_835.h"
#include <cmath>


ITURP_835::ITURP_835()
{

}

ITURP_835::~ITURP_835()
{

}

// ITU-R P.835-6, Annex 1, Section 1.1
// geometricHeight_km: Geometric height (km)
// [return]:           Mean annual global reference atmosphere pressure (hPa)
double ITURP_835::StandardPressure(double geometricHeight_km)
{
double h = geometricHeight_km;

	if( h < 0.0 )
		h = 0.0;
	if( h > 100.0 )
		h = 100.0;

	if( h < 86.0 )
	{
	double hp = _ToGeopotentialHeight(h);

		if( hp <= 11.0 )
			return 1013.25*pow(288.15/(288.15-6.5*hp), -34.1632/6.5);
		else if( hp <= 20.0 )
			return 226.3226*exp(-34.1632*(hp-11.0)/216.65);
		else if( hp <= 32.0 )
			return 54.74980*pow(216.65/(216.65+(hp-20.0)), 34.1632);
		else if( hp <= 47.0 )
			return 8.680422*pow(228.65/(228.65+2.8*(hp-32.0)), 34.1632/2.8);
		else if( hp <= 51.0 )
			return 1.109106*exp(-34.1632*(hp-47.0)/270.65);
		else if( hp <= 71.0 )
			return 0.6694167*pow(270.65/(270.65-2.8*(hp-51.0)), -34.1632/2.8);
		else
			return 0.03956649*pow(214.65/(214.65-2.0*(hp-71.0)), -34.1632/2.0);
	}
	else
	{
		return exp(95.571899 + -4.011801*h + 6.424731E-2*h*h + -4.789660E-4*h*h*h + 1.340543E-6*h*h*h*h);
	}
}

double ITURP_835::_ToGeopotentialHeight(double geometricHeight_km)
{
	return (6356.766*geometricHeight_km)/(6356.766+geometricHeight_km);
}
