#pragma once

class Projection
{
public:
	Projection();
	virtual ~Projection();

	virtual bool GeographicToProjected(double lat, double lon, double* x, double* y) const = 0;
	virtual bool ProjectedToGeographic(double x, double y, double* lat, double* lon) const = 0;

	static const double PI;
	static const double PI_ON_2;
	static const double PI_ON_4;
	static const double PI_ON_180;
};


class LambertConicConformal2SP : public Projection
{
public:
	LambertConicConformal2SP(double stdLat1, double stdLat2, double falseOriginLat, double falseOriginLon,
	                         double falseOriginE, double falseOriginN, double ellpsSemiMajorAxis, double ellpsRevFlattening);
	virtual ~LambertConicConformal2SP();

	virtual bool GeographicToProjected(double lat, double lon, double* x, double* y) const;
	virtual bool ProjectedToGeographic(double x, double y, double* lat, double* lon) const;

protected:
	// see https://epsg.io/9802-method
	double EF;
	double NF;
	double lon0_rad;
	double a; // ellipsoid semi-major axis
	double e; // ellipsoid eccentricity
	double n;
	double F;
	double rF;
};