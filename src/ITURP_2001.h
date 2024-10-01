#pragma once
#include <vector>
#include <utility>
	

class ITURP_2001
{
public:
	ITURP_2001();
	~ITURP_2001();

	static void TroposphericScatterPathSegments(double d, double theta_e, double theta_tpos, double theta_rpos, double hts, double hrs, double txLat, double txLon, double rxLat, double rxLon, double* dtcv, double* drcv, double* phi_cvn, double* phi_cve, double* hcv, double* phi_tcvn, double* phi_tcve, double* phi_rcvn, double* phi_rcve);
	static void GreatCircleIntermediatePoint(double lat0, double lon0, double lat1, double lon1, double distKm, double& intermLat, double& intermLon);
	static void GreatCircleIntermediatePoints(double lat0, double lon0, double lat1, double lon1, std::vector<double>& distKmProfile, std::vector<std::pair<double,double>>& outLatLonProfile);
	static void GreatCircleDistance(double lat0, double lon0, double lat1, double lon1, double& distKm);
	static double Wavelength(double f);

	static const double Re;
	static const double c;
	static const double PI;
	static const double PI_ON_180;
};