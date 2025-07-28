/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_2001.h"
#include <cmath>
#include <algorithm>


// Rec. ITU-R P.2001-4, Section 2.3 of Annex
const double ITURP_2001::Re = 6371.0; // Average Earth radius (km)
const double ITURP_2001::c = 2.998E8; // Speed of propagation (m/s)

const double ITURP_2001::PI = 3.14159265358979323846;
const double ITURP_2001::PI_ON_180 = 0.017453292519943295769;

ITURP_2001::ITURP_2001()
{

}

ITURP_2001::~ITURP_2001()
{

}

// Rec. ITU-R P.2001-4, Sectio0n 3.9 of Annex
// d:          path length (km)
// theta_e:    angle subtended by d km at centre of spherical Earth (rad)
// theta_tpos: horizon elevation angle relative to the local horizontal as viewed from the transmitter
//             and limited to be positive (not less than zero) (mrad)
// theta_rpos: horizon elevation angle relative to the local horizontal as viewed from the receiver
//             and limited to be positive (not less than zero) (mrad)
// hts:        transmitter height above mean sea level (m)
// hrs:        receiver height above mean sea level (m)
// txLat:      latitude of transmitter (degrees) [-90, 90]
// txLon:      longitude of transmitter (degrees) [-180, 180]
// rxLat:      latitude of receiver (degrees) [-90, 90]
// rxLon:      longitude of receiver (degrees) [-180, 180]
// [out] dtcv:     horizontal path length from transmitter to common volume (km)
// [out] drcv:     horizontal path length from common volume to receiver (km)
// [out] phi_cvn:  latitude of the common volume (degrees) [-90, 90]
// [out] phi_cve:  longitude of the common volume (degrees) [-180, 180]
// [out] hcv:      height of the troposcatter common volume (m amsl)
// [out] phi_tcvn: mid-point's latitude of the path segment from transmitter to common volume (degrees) [-90, 90]
// [out] phi_tcve: mid-point's longitude of the path segment from transmitter to common volume (degrees) [-180, 180]
// [out] phi_rcvn: mid-point's latitude of the path segment from receiver to common volume (degrees) [-90, 90]
// [out] phi_rcve: mid-point's longitude of the path segment from receiver to common volume (degrees) [-180, 180]
void ITURP_2001::TroposphericScatterPathSegments(double d, double theta_e, double theta_tpos, double theta_rpos, double hts, double hrs,
                                                 double txLat, double txLon, double rxLat, double rxLon,
                                                 double* dtcv, double* drcv, double* phi_cvn, double* phi_cve, double* hcv,
                                                 double* phi_tcvn, double* phi_tcve, double* phi_rcvn, double* phi_rcve)
{
double d_tcv;
double d_rcv;

	d_tcv = (d*tan(0.001*theta_rpos + 0.5*theta_e) - 0.001*(hts-hrs)) / (tan(0.001*theta_tpos + 0.5*theta_e) + tan(0.001*theta_rpos + 0.5*theta_e));
	d_tcv = std::max(0.0, d_tcv);
	d_tcv = std::min(d, d_tcv);
	d_rcv = d - d_tcv;

	if( dtcv != nullptr )
		*dtcv = d_tcv;

	if( drcv != nullptr )
		*drcv = d_rcv;	

	if( phi_cvn != nullptr && phi_cve != nullptr )
		GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, d_tcv, *phi_cvn, *phi_cve);

	if( hcv != nullptr )
	{
		double ae = d / theta_e; // eq. (14), Section 3.5 of Annex
		*hcv = hts + 1000.0*d_tcv*tan(0.001*theta_tpos) + (1000.0*d_tcv*d_tcv/(2.0*ae));
	}

	if( phi_tcvn != nullptr && phi_tcve != nullptr )
		GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, 0.5*d_tcv, *phi_tcvn, *phi_tcve);

	if( phi_rcvn != nullptr && phi_rcve != nullptr )
		GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, d-(0.5*d_rcv), *phi_rcvn, *phi_rcve);	
}

// Rec. ITU-R P.2001-4, Attachment H
// lat0:            latitude of first end of path (degrees) [-90, 90]
// lon0:            longitude of first end of path (degrees) [-180, 180]
// lat1:            latitude of second end of path (degrees) [-90, 90]
// lon1:            longitude of second end of path (degrees) [-180, 180]
// distKm:          distance of an intermediate point from the first end of path (km)
// [out] intermLat: latitude of the intermediate point along the great-circle path (degrees) [-90, 90]
// [out] intermLon: longitude of the intermediate point along the great-circle path(degrees) [-180, 180]
void ITURP_2001::GreatCircleIntermediatePoint(double lat0, double lon0, double lat1, double lon1, double distKm, double& intermLat, double& intermLon)
{
double phi_tn = lat0*PI_ON_180;
double sin_phi_tn = sin(phi_tn);
double cos_phi_tn = cos(phi_tn);
double phi_te = lon0*PI_ON_180;
double phi_rn = lat1*PI_ON_180;
double sin_phi_rn = sin(phi_rn);
double cos_phi_rn = cos(phi_rn);
double phi_re = lon1*PI_ON_180;
double phi_pnt = distKm/Re;
double sin_phi_pnt = sin(phi_pnt);
double cos_phi_pnt = cos(phi_pnt);
double phi_pntn, phi_pnte;
double x1, y1, x2, y2;
double Bt2r, delta_lon, r, s;

	delta_lon = phi_re-phi_te;
	r = sin_phi_tn*sin_phi_rn+cos_phi_tn*cos_phi_rn*cos(delta_lon);
	x1 = sin_phi_rn-r*sin_phi_tn;
	y1 = cos_phi_tn*cos_phi_rn*sin(delta_lon);
	if( fabs(x1) < 1E-9 && fabs(y1) < 1E-9 )
		Bt2r = phi_re;
	else
		Bt2r = atan2(y1, x1);

	s = sin_phi_tn*cos_phi_pnt+cos_phi_tn*sin_phi_pnt*cos(Bt2r);
	phi_pntn = asin(s);
	x2 = cos_phi_pnt-s*sin_phi_tn;
	y2 = cos_phi_tn*sin_phi_pnt*sin(Bt2r);
	if( fabs(x2) < 1E-9 && fabs(y2) < 1E-9 )
		phi_pnte = Bt2r;
	else
		phi_pnte = phi_te+atan2(y2, x2);

	intermLat = phi_pntn/PI_ON_180;
	intermLon = phi_pnte/PI_ON_180;
	if( intermLon < -180)
		intermLon += 360;
	if( intermLon > 180)
		intermLon -= 360;
}

// Rec. ITU-R P.2001-4, Attachment H
// lat0:                   latitude of first end of path (degrees) [-90, 90]
// lon0:                   longitude of first end of path (degrees) [-180, 180]
// lat1:                   latitude of second end of path (degrees) [-90, 90]
// lon1:                   longitude of second end of path (degrees) [-180, 180]
// distKmProfile:          distances of intermediate points from the first end of path (km)
// [out] outLatLonProfile: latitude/longitude of the intermediate points along the great-circle path (degrees) [-90, 90]
void ITURP_2001::GreatCircleIntermediatePoints(double lat0, double lon0, double lat1, double lon1, std::vector<double>& distKmProfile, 
                                               std::vector<std::pair<double,double>>& outLatLonProfile)
{
double phi_tn = lat0*PI_ON_180;
double sin_phi_tn = sin(phi_tn);
double cos_phi_tn = cos(phi_tn);
double phi_te = lon0*PI_ON_180;
double phi_rn = lat1*PI_ON_180;
double sin_phi_rn = sin(phi_rn);
double cos_phi_rn = cos(phi_rn);
double phi_re = lon1*PI_ON_180;
double phi_pnt, phi_pntn, phi_pnte;
double cos_phi_pnt, sin_phi_pnt;
double x1, y1, x2, y2;
double Bt2r, delta_lon, r, s, cos_Bt2r, sin_Bt2r;
size_t numPts = distKmProfile.size();
std::pair<double,double> intermPt;

	delta_lon = phi_re-phi_te;
	r = sin_phi_tn*sin_phi_rn+cos_phi_tn*cos_phi_rn*cos(delta_lon);
	x1 = sin_phi_rn-r*sin_phi_tn;
	y1 = cos(phi_tn)*cos_phi_rn*sin(delta_lon);
	if( fabs(x1) < 1E-9 && fabs(y1) < 1E-9 )
		Bt2r = phi_re;
	else
		Bt2r = atan2(y1, x1);

	cos_Bt2r = cos(Bt2r);
	sin_Bt2r = sin(Bt2r);

	outLatLonProfile.clear();
	outLatLonProfile.reserve(numPts);
	for(size_t i=0 ; i<numPts ; i++)
	{
		phi_pnt = distKmProfile[i]/Re;

		cos_phi_pnt = cos(phi_pnt);
		sin_phi_pnt = sin(phi_pnt);
		s = sin_phi_tn*cos_phi_pnt+cos_phi_tn*sin_phi_pnt*cos_Bt2r;
		phi_pntn = asin(s);
		x2 = cos_phi_pnt-s*sin_phi_tn;
		y2 = cos_phi_tn*sin_phi_pnt*sin_Bt2r;
		if( fabs(x2) < 1E-9 && fabs(y2) < 1E-9 )
			phi_pnte = Bt2r;
		else
			phi_pnte = phi_te+atan2(y2, x2);

		intermPt.first = phi_pntn/PI_ON_180; // latitude
		intermPt.second = phi_pnte/PI_ON_180; // longitude
		if( intermPt.second < -180)
			intermPt.second += 360;
		if( intermPt.second > 180)
			intermPt.second -= 360;
		outLatLonProfile.push_back(intermPt);
	}
}

// Rec. ITU-R P.2001-4, Attachment H
// lat0:         latitude of first end of path (degrees) [-90, 90]
// lon0:         longitude of first end of path (degrees) [-180, 180]
// lat1:         latitude of second end of path (degrees) [-90, 90]
// lon1:         longitude of second end of path (degrees) [-180, 180]
// [out] distKm: great-circle path length (km)
void ITURP_2001::GreatCircleDistance(double lat0, double lon0, double lat1, double lon1, double& distKm)
{
	lat0 = lat0*PI_ON_180;
	lon0 = lon0*PI_ON_180;
	lat1 = lat1*PI_ON_180;
	lon1 = lon1*PI_ON_180;
	double x = (sin(lat0)*sin(lat1)) + (cos(lat0)*cos(lat1)*cos(lon1-lon0));
	x = std::min(x, 1.0); // to avoid domain error in acos() in some cases where (lat0, lon0) == (lat1, lon1)
	distKm = acos(x) * Re;
}

// Rec. ITU-R P.2001-4, Section 3.6 of Annex
// f:        frequency (GHz)
// [return]: wavelength (m)
double ITURP_2001::Wavelength(double f)
{
	return 1.0E-9*c/f;
}