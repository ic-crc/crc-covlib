#include "Projections.h"
#include <cmath>


const double Projection::PI = 3.14159265358979323846;
const double Projection::PI_ON_2 = 1.57079632679489661923;
const double Projection::PI_ON_4 = 0.78539816339744830962;
const double Projection::PI_ON_180 = 0.017453292519943295769;

Projection::Projection()
{
}

Projection::~Projection()
{
}


// See https://epsg.io/9802-method
// stdLat1:            latitude of first standard parallel (degrees)
// stdLat2:            latitude of second standard parallel (degrees)
// falseOriginLat:     latitude of false origin (degrees)
// centalLon:          central longitude (degrees)
// falseOriginE:       easting at false origin (meters)
// falseOriginN:       northing at false origin (meters)
// ellpsSemiMajorAxis: semi-major axis of the ellipsoid (meters)
// ellpsRevFlattening: reverse flattening of the ellipsoid
LambertConicConformal2SP::LambertConicConformal2SP(double stdLat1, double stdLat2, double falseOriginLat, double centalLon,
                                                   double falseOriginE, double falseOriginN, double ellpsSemiMajorAxis, double ellpsRevFlattening)
{
	EF = falseOriginE;
	NF = falseOriginN;

	a = ellpsSemiMajorAxis;
	double f = 1/ellpsRevFlattening;
    double b = a*(1-f);
    e = sqrt((a*a - b*b) / (a*a));

	double lat1_rad = stdLat1*PI_ON_180;
	double lat2_rad = stdLat2*PI_ON_180;
	double latF_rad = falseOriginLat*PI_ON_180;
	lon0_rad = centalLon*PI_ON_180;

	double sin_lat1 = sin(lat1_rad);
	double sin_lat2 = sin(lat2_rad);
	double sin_latF = sin(latF_rad);

	double m1 = cos(lat1_rad)/sqrt(1 - e*e*sin_lat1*sin_lat1);
	double m2 = cos(lat2_rad)/sqrt(1 - e*e*sin_lat2*sin_lat2);

	double t1 = tan(PI_ON_4 - lat1_rad/2) / pow((1 - e*sin_lat1)/(1 + e*sin_lat1), e/2);
	double t2 = tan(PI_ON_4 - lat2_rad/2) / pow((1 - e*sin_lat2)/(1 + e*sin_lat2), e/2);
	double tF = tan(PI_ON_4 - latF_rad/2) / pow((1 - e*sin_latF)/(1 + e*sin_latF), e/2);

	n = (log(m1) - log(m2))/(log(t1) - log(t2));

	F = m1/(n*pow(t1, n));

	rF = a*F*pow(tF, n);
}

LambertConicConformal2SP::~LambertConicConformal2SP()
{

}

bool LambertConicConformal2SP::GeographicToProjected(double lat, double lon, double* x, double* y) const
{
	double lat_rad = lat*PI_ON_180;
	double lon_rad = lon*PI_ON_180;
	double sin_lat = sin(lat_rad);

	double t  = tan(PI_ON_4 - lat_rad/2) / pow((1 - e*sin_lat)/(1 + e*sin_lat), e/2);

	double r = a*F*pow(t, n);

	double theta = n*(lon_rad-lon0_rad);

	*x = EF + r*sin(theta);
	*y = NF + rF - r*cos(theta);
	return true;
}

bool LambertConicConformal2SP::ProjectedToGeographic(double x, double y, double* lat, double* lon) const
{
	double r_p = sqrt((x-EF)*(x-EF) + (rF-(y-NF))*(rF-(y-NF)));
	if(n < 0)
		r_p = -r_p;

	double t_p = pow(r_p/(a*F), 1/n);

	double theta_p = atan2(x-EF, rF-(y-NF));

	double lat_rad = PI_ON_2 - 2*atan(t_p);
	for(int i=0 ; i<5 ; i++)
	{
		double sin_lat = sin(lat_rad);
		lat_rad = PI_ON_2 - 2*atan(t_p*pow((1 - e*sin_lat)/(1 + e*sin_lat), e/2));
	}

	double lon_rad = theta_p/n + lon0_rad;

	*lat = lat_rad/PI_ON_180;
	*lon = lon_rad/PI_ON_180;
	return true;
}