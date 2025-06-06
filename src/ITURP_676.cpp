/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_676.h"
#include <cmath>


ITURP_676::ITURP_676()
{

}

ITURP_676::~ITURP_676()
{

}

// f:        frequency (GHz) [1, 1000]
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// rho:      water vapour density (g/m3)
// [return]: specific gaseous attenuation attributable to dry air (oxygen, pressure-induced 
//           nitrogen and non-resonant Debye attenuation) (dB/km)
double ITURP_676::AttenuationDueToDryAir(double f, double P, double TK, double rho)
{
	double yo = 0.1820*f*_NppOxygen(f, P, TK, rho); // eq. (1), (33)
	return yo;
}

// f:        frequency (GHz) [1, 1000]
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// rho:      water vapour density (g/m3)
// [return]: specific gaseous attenuation attributable to water vapour (dB/km)
double ITURP_676::AttenuationDueToWaterVapour(double f, double P, double TK, double rho)
{
	double yw = 0.1820*f*_NppWaterVapour(f, P, TK, rho); // eq. (1), (36)
	return yw;
}

// ITU-R P.676-13, Annex 1, Section 1 - N''Oxygen
// f:        frequency (GHz) [1, 1000]
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// rho:      water vapour density (g/m3)
// [return]: imaginary part of the complex refractivity attributable to oxygen
double ITURP_676::_NppOxygen(double f, double P, double TK, double rho)
{
static const int NUM_LINES = 44;
double Si, Fi;
double theta = 300.0/TK;
double e = rho*TK/216.7;
double df, fi, delta;
const double* a;
double NppO = 0.0;

double prod1 = 1.0E-7*P*theta*theta*theta;
double prod2 = 1.1*e*theta;
double prod3 = 1.0E-4*(P+e)*pow(theta,0.8);

	for(int i=0 ; i<NUM_LINES ; i++)
	{
		fi = _TABLE1[i][0];
		a = _TABLE1[i];

		Si = a[1] * prod1 * exp(a[2]*(1.0-theta));

		df = a[3]*1.0E-4*((P*pow(theta,0.8-a[4]))+prod2);
		df = sqrt((df*df)+2.25E-6);
		delta = (a[5]+(a[6]*theta))*prod3;

		Fi = (f/fi)*( ((df-(delta*(fi-f)))/(((fi-f)*(fi-f))+(df*df))) + ((df-(delta*(fi+f)))/(((fi+f)*(fi+f))+(df*df))) );
		
		NppO += Si*Fi;
	}
	NppO += _NppD(f, P, TK, rho);

	return NppO;
}

// ITU-R P.676-13, Annex 1, Section 1 - N''Water Vapour
// f:        frequency (GHz) [1, 1000]
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// rho:      water vapour density (g/m3)
// [return]: imaginary part of the complex refractivity attributable to water vapour
double ITURP_676::_NppWaterVapour(double f, double P, double TK, double rho)
{
static const int NUM_LINES = 35;
double Si, Fi;
double theta = 300.0/TK;
double e = rho*TK/216.7;
double df, fi;
const double* b;
double NppWV = 0.0;

double prod1 = 1.0E-1*e*pow(theta, 3.5);

	for(int i=0 ; i<NUM_LINES ; i++)
	{
		fi = _TABLE2[i][0];
		b = _TABLE2[i];

		Si = b[1] * prod1 * exp(b[2]*(1.0-theta));

		df = b[3]*1.0E-4*((P*pow(theta,b[4])) + (b[5]*e*pow(theta,b[6])));
		df = 0.535*df + sqrt((0.217*df*df)+(2.1316E-12*fi*fi/theta));
		Fi = (f/fi)*( (df/(((fi-f)*(fi-f))+(df*df))) + (df/(((fi+f)*(fi+f))+(df*df))) );
		
		NppWV += Si*Fi;
	}

	return NppWV;
}

// ITU-R P.676-13, Annex 1, Section 1 - N''D
// f:        frequency (GHz) [1, 1000]
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// rho:      water vapour density (g/m3)
// [return]: the dry continuum due to pressure-induced nitrogen absorption and the Debye spectrum 
double ITURP_676::_NppD(double f, double P, double TK, double rho)
{
double theta = 300.0/TK;
double e = rho*TK/216.7;
double d;
double NppD;

	d = 5.6E-4*(P+e)*pow(theta, 0.8);
	NppD = f*P*theta*theta*( (6.14E-5/(d*(1.0+((f*f)/(d*d))))) + ((1.4E-12*P*pow(theta,1.5))/(1.0+(1.9E-5*pow(f,1.5)))) );
	return NppD;
}

const double ITURP_676::_TABLE1[44][7]={
	{50.474214,0.975,9.651,6.690,0.0,2.566,6.850},
	{50.987745,2.529,8.653,7.170,0.0,2.246,6.800},
	{51.503360,6.193,7.709,7.640,0.0,1.947,6.729},
	{52.021429,14.320,6.819,8.110,0.0,1.667,6.640},
	{52.542418,31.240,5.983,8.580,0.0,1.388,6.526},
	{53.066934,64.290,5.201,9.060,0.0,1.349,6.206},
	{53.595775,124.600,4.474,9.550,0.0,2.227,5.085},
	{54.130025,227.300,3.800,9.960,0.0,3.170,3.750},
	{54.671180,389.700,3.182,10.370,0.0,3.558,2.654},
	{55.221384,627.100,2.618,10.890,0.0,2.560,2.952},
	{55.783815,945.300,2.109,11.340,0.0,-1.172,6.135},
	{56.264774,543.400,0.014,17.030,0.0,3.525,-0.978},
	{56.363399,1331.800,1.654,11.890,0.0,-2.378,6.547},
	{56.968211,1746.600,1.255,12.230,0.0,-3.545,6.451},
	{57.612486,2120.100,0.910,12.620,0.0,-5.416,6.056},
	{58.323877,2363.700,0.621,12.950,0.0,-1.932,0.436},
	{58.446588,1442.100,0.083,14.910,0.0,6.768,-1.273},
	{59.164204,2379.900,0.387,13.530,0.0,-6.561,2.309},
	{59.590983,2090.700,0.207,14.080,0.0,6.957,-0.776},
	{60.306056,2103.400,0.207,14.150,0.0,-6.395,0.699},
	{60.434778,2438.000,0.386,13.390,0.0,6.342,-2.825},
	{61.150562,2479.500,0.621,12.920,0.0,1.014,-0.584},
	{61.800158,2275.900,0.910,12.630,0.0,5.014,-6.619},
	{62.411220,1915.400,1.255,12.170,0.0,3.029,-6.759},
	{62.486253,1503.000,0.083,15.130,0.0,-4.499,0.844},
	{62.997984,1490.200,1.654,11.740,0.0,1.856,-6.675},
	{63.568526,1078.000,2.108,11.340,0.0,0.658,-6.139},
	{64.127775,728.700,2.617,10.880,0.0,-3.036,-2.895},
	{64.678910,461.300,3.181,10.380,0.0,-3.968,-2.590},
	{65.224078,274.000,3.800,9.960,0.0,-3.528,-3.680},
	{65.764779,153.000,4.473,9.550,0.0,-2.548,-5.002},
	{66.302096,80.400,5.200,9.060,0.0,-1.660,-6.091},
	{66.836834,39.800,5.982,8.580,0.0,-1.680,-6.393},
	{67.369601,18.560,6.818,8.110,0.0,-1.956,-6.475},
	{67.900868,8.172,7.708,7.640,0.0,-2.216,-6.545},
	{68.431006,3.397,8.652,7.170,0.0,-2.492,-6.600},
	{68.960312,1.334,9.650,6.690,0.0,-2.773,-6.650},
	{118.750334,940.300,0.010,16.640,0.0,-0.439,0.079},
	{368.498246,67.400,0.048,16.400,0.0,0.000,0.000},
	{424.763020,637.700,0.044,16.400,0.0,0.000,0.000},
	{487.249273,237.400,0.049,16.000,0.0,0.000,0.000},
	{715.392902,98.100,0.145,16.000,0.0,0.000,0.000},
	{773.839490,572.300,0.141,16.200,0.0,0.000,0.000},
	{834.145546,183.100,0.145,14.700,0.0,0.000,0.000}
};

const double ITURP_676::_TABLE2[35][7]={
	{22.235080,.1079,2.144,26.38,.76,5.087,1.00},
	{67.803960,.0011,8.732,28.58,.69,4.930,.82},
	{119.995940,.0007,8.353,29.48,.70,4.780,.79},
	{183.310087,2.273,.668,29.06,.77,5.022,.85},
	{321.225630,.0470,6.179,24.04,.67,4.398,.54},
	{325.152888,1.514,1.541,28.23,.64,4.893,.74},
	{336.227764,.0010,9.825,26.93,.69,4.740,.61},
	{380.197353,11.67,1.048,28.11,.54,5.063,.89},
	{390.134508,.0045,7.347,21.52,.63,4.810,.55},
	{437.346667,.0632,5.048,18.45,.60,4.230,.48},
	{439.150807,.9098,3.595,20.07,.63,4.483,.52},
	{443.018343,.1920,5.048,15.55,.60,5.083,.50},
	{448.001085,10.41,1.405,25.64,.66,5.028,.67},
	{470.888999,.3254,3.597,21.34,.66,4.506,.65},
	{474.689092,1.260,2.379,23.20,.65,4.804,.64},
	{488.490108,.2529,2.852,25.86,.69,5.201,.72},
	{503.568532,.0372,6.731,16.12,.61,3.980,.43},
	{504.482692,.0124,6.731,16.12,.61,4.010,.45},
	{547.676440,.9785,.158,26.00,.70,4.500,1.00},
	{552.020960,.1840,.158,26.00,.70,4.500,1.00},
	{556.935985,497.0,.159,30.86,.69,4.552,1.00},
	{620.700807,5.015,2.391,24.38,.71,4.856,.68},
	{645.766085,.0067,8.633,18.00,.60,4.000,.50},
	{658.005280,.2732,7.816,32.10,.69,4.140,1.00},
	{752.033113,243.4,.396,30.86,.68,4.352,.84},
	{841.051732,.0134,8.177,15.90,.33,5.760,.45},
	{859.965698,.1325,8.055,30.60,.68,4.090,.84},
	{899.303175,.0547,7.914,29.85,.68,4.530,.90},
	{902.611085,.0386,8.429,28.65,.70,5.100,.95},
	{906.205957,.1836,5.110,24.08,.70,4.700,.53},
	{916.171582,8.400,1.441,26.73,.70,5.150,.78},
	{923.112692,.0079,10.293,29.00,.70,5.000,.80},
	{970.315022,9.009,1.919,25.50,.64,4.940,.67},
	{987.926764,134.6,.257,29.85,.68,4.550,.90},
	{1780.000000,17506.,.952,196.3,2.00,24.15,5.00}
};