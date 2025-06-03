/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

// Implementation of ITU-R P.1812-6
// Does not include building entry loss (Annex1, Section 4.8)
// Does not include the alternative method to calculate the spherical earth diffraction loss (Attachment 3 to Annex 1)
#include "ITURP_1812.h"
#include "ITURP_2001.h"
#include "ITURP_DigitalMaps.h"
#include <cmath>


ITURP_1812::ITURP_1812()
{
	_wa = 100;
}
	
ITURP_1812::~ITURP_1812()
{

}

void ITURP_1812::SetDefaultRepresentativeHeight(ClutterCategory clutterCategory, double representativeHeight)
{
	_SetDefaultRepresentativeHeight(clutterCategory, representativeHeight);
}
	
double ITURP_1812::GetDefaultRepresentativeHeight(ClutterCategory clutterCategory) const
{
	return _GetDefaultRepresentativeHeight(clutterCategory);
}

void ITURP_1812::SetPredictionResolution(double resolution_meters)
{
	if( resolution_meters < 0 )
		return;
	_wa = resolution_meters;
}

double ITURP_1812::GetPredictionResolution() const
{
	return _wa;
}

// f:        frequency (GHz) [0.03, 6.0]
// p:        percentage of average year for which the calculated basic transmission loss is not exceeded (%) [1.0, 50.0]
// pL:       percentage of locations for which the calculated basic transmission loss is not exceeded (%) [1.0, 99.0]
// latt:     latitude of transmitter (degrees) [-80, 80]
// lont:     longitude of transmitter, positive = East of Greenwich (degrees) [-180, 180]
// latr:     latitude of receiver (degrees) [-80, 80]
// lonr:     longitude of receiver, positive = East of Greenwich (degrees) [-180, 180]
// htg:      transmit antenna centre height above ground level (m) [1, 3000]
// hrg:      receive antenna centre height above ground level (m) [1, 3000]
// vpol:     true to indicate vertical polarization, false to indicate horizontal polarization
// n:        number of profile points
// d:        distance from transmitter profile of n points (km)
// h:        terrain height profile of n points (m above mean sea level)
// cc:       clutter category profile of n points (if set to nullptr, OPEN_RURAL is assumed everywhere).
//           cc is ignored if either or both rch and sh are specified (i.e. if rch and/or sh are not set to nullptr).
// rch:      representative clutter height (m) profile of n points.
//           rch is ignored if sh is specified (i.e. if sh is not set to nullptr)
// sh:       surface height (m) profile of n points.
// rcz:      radio climatic zone profile of n points (if set to nullptr, INLAND is assumed everywhere)
// dct, dcr: If at least 75% of the path is over zone B (SEA) two further parameters are required, dct, dcr, giving the distance of
//           the transmitter and the receiver from the coast (km), respectively, in the direction of the other terminal.
//           For a terminal on a ship or sea platform the distance is zero. (If set to ITURP_452_1812::AUTO, values are automatically
//           estimated using the radio climate zone (rcz) and distance (d) profiles).
// deltaN:   average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere (if set to ITURP_452_1812::AUTO, its
//           value is automatically obtained from one of the Recommendation's digital map) (N-units/km)
// N0:       sea-level surface refractivity (if set to ITURP_452_1812::AUTO, its value is automatically obtained from one of the
//           Recommendation's digital map) (N-units)
// [return]: basic transmission loss (dB)
double ITURP_1812::BasicTransmissionloss(double f, double p, double pL, double latt, double lont, double latr, double lonr,
                                         double htg, double hrg, bool vpol, unsigned int n, double* d, double *h,
                                         ClutterCategory* cc, double* rch, double* sh, RadioClimaticZone* rcz,
                                         double dct, double dcr, double deltaN, double N0)
{
std::vector<double> gVector;
double* g; // g: terrain height (m amsl) + representative clutter height (m) profile of n points
std::vector<double> lopsVector;
double* lops; // lops: length of path sections profile of n points (km)
double hts; // hts: transmit antenna centre height above mean sea level (m)
double hrs; // hrs: receive antenna centre height above mean sea level (m)
double dn; // dn: great-circle path distance (km)
double centreLat; // centreLat: latitude of the path's centre point (where the path is the great-circle between the transmit and receive antennas) (degrees) 
double centreLon; // centreLon: longitude of the path's centre point (where the path is the great-circle between the transmit and receive antennas) (degrees)
double B0; // B0: the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
double aB; // aB: effective Earth radius exceeded for B0 time (km)
double ae; // ae: median effective Earth radius (km)
double omega; // omega: the fraction of the total path over water
double R; // R: height of representative clutter at the receiver/mobile location (m)
double dtm; // longest continuous land (inland + coastal) section of the great-circle path (km)
double dlm; // longest continuous inland section of the great-circle path (km)

double dlt; // dlt: transmitting antenna horizon distance (km)
double dlr; // dlr: receiving antenna horizon distance (km)
double thetat; // thetat: transmit horizon elevation angle (mrad)
double thetar; // thetar: receive horizon elevation angle (mrad)
double theta; // theta: path angular distance (mrad)
double hstd; // hstd: modified smooth-Earth surface at the transmitting station location for the diffraction model (m amsl)
double hsrd; // hsrd: modified smooth-Earth surface at the receiving station location for the diffraction model (m amsl)
double hte; // hte: effective height of the transmitting antenna for the ducting/layer-reflection model (m)
double hre; // hre: effective height of the receiving antenna for the ducting/layer-reflection model (m)
double hm; // hm: terrain roughness (m)

double Lbfs; // Lbfs: basic transmission loss due to free-space propagation (dB)
double Lb0p; // Lb0p: basic transmission loss not exceeded for time percentage, p%, due to LoS propagation (dB)
double Lb0B; // Lb0B: basic transmission loss not exceeded for time percentage, B0%, due to LoS propagation (dB)
double Ldp; // Ldp: diffraction loss not exceeded for p% time (dB)
double Lbd50; // Lbd50: median basic transmission loss associated with diffraction (dB)
double Lbd; // Lbd: basic transmission loss associated with diffraction not exceeded for p% time (dB)
double Lbs; // Lbs: basic transmission loss due to troposcatter (dB)
double Lba; // Lba: basic transmission loss associated with ducting/layer-reflection not exceeded for p% time (dB)
double Lbc; // Lbc: basic transmission loss not exceeded for p% time and 50% locations (dB)
double Lb; // Lb: basic transmission loss not exceeded for p% time and pL% locations (dB)

	_gProfile(n, h, cc, rch, sh, gVector, R);
	g = gVector.data();
	_LengthOfPathSectionsProfile(n, d, lopsVector);
	lops = lopsVector.data();
	hts = h[0] + htg;
	hrs = h[n-1] + hrg;
	dn = d[n-1];
	ITURP_2001::GreatCircleIntermediatePoint(latt, lont, latr, lonr, dn/2.0, centreLat, centreLon);
	_LongestContinuousLand(dn, n, rcz, lops, dtm, dlm);
	B0 = _B0(centreLat, dtm, dlm);
	if( _IsAUTO(deltaN) )
		deltaN = ITURP_DigitalMaps::DN50(centreLat, centreLon);
	if( _IsAUTO(N0) )
		N0 = ITURP_DigitalMaps::N050(centreLat, centreLon);
	_EffectiveEarthRadius(deltaN, ae, aB);
	omega = _OverWaterPercent(dn, n, rcz, lops);
	if( omega >= 0.75 ) // otherwise dct and dcr are not used
	{
		if( _IsAUTO(dct) )
			dct = _Dct(n, rcz, lops);
		if( _IsAUTO(dcr) )
			dcr = _Dcr(n, rcz, lops);
	}

	_PathProfileParameters(f, htg, hrg, n, d, h, ae, dlt, dlr, thetat, thetar, theta, hstd, hsrd, hte, hre, hm);

	_LoSPropagationLoss(f, p, B0, hts, hrs, dn, dlt, dlr, Lbfs, Lb0p, Lb0B);
	_DiffractionLoss(f, p, B0, hts, hrs, vpol, ae, aB, n, d, g, omega, hstd, hsrd, Lbfs, Lb0p, Ldp, Lbd50, Lbd);
	Lbs = _TroposcatterLoss(f, p, dn, theta, N0);
	Lba = _DuctingLayerReflectionLoss(f, p, ae, B0, dlm, dn, hts, hrs, dlt, dlr, thetat, thetar, hte, hre, hm, omega, dct, dcr);
	Lbc = _50PLocBasicTransmissionLoss(p, B0, dn, theta, omega, Lb0p, Lb0B, Ldp, Lbd50, Lba, Lbd, Lbs);
	Lb = _BasicTransmissionLoss(f, pL, hrg, R, Lb0p, Lbc);

	return Lb;
}

// ITU-R P.1812-6/7, Annex 1, Section 4.10
// f:        frequency (GHz) [0.03, 6.0]
// Lb:       basic transmission loss not exceeded for p% time and pL% locations calculated by BasicTransmissionloss()
// [return]: field strength normalized to 1 kW effective radiated power exceeded for p% time and pL% locations (dBuV/m)
double ITURP_1812::FieldStrength(double f, double Lb)
{
	return 199.36 + 20.0*log10(f) - Lb;
}

// ITU-R P.1812-6, Annex 1, Section 4.4
// f:        frequency (GHz) [0.03, 6.0]
// p:        percentage of average year for which the calculated basic transmission loss is not exceeded (%) [1.0, 50.0]
// dn:       great-circle path distance between transmit and receive antennas (km)
// theta:    path angular distance (mrad)
// N0:       sea-level surface refractivity (N-units)
// [return]: Lbs, basic transmission loss due to troposcatter (dB)
double ITURP_1812::_TroposcatterLoss(double f, double p, double dn, double theta, double N0)
{
double Lf, Lbs;

	Lf = 25.0*log10(f) - 2.5*log10(f/2.0)*log10(f/2.0);
	Lbs = 190.1 + Lf + 20.0*log10(dn) + 0.573*theta - 0.15*N0 - 10.125*pow(log10(50.0/p), 0.7);

	return Lbs;
}

// ITU-R P.1812-7, Annex 1, Section 4.4
// Note1: currently not used since the ITU website mentions: "users are advised to revert to using the
//        troposcatter model from Recommendation ITU-R P.1812-6 for the calculation of the Lbs parameter
//        (eq. (44))."
// Note2: also in Draft version of ITU-R P.452-17 (R19-SG03-C-0127!R1!MSW-E.docx), Annex 1, Section 4.3,
//        but was removed in official release of ITU-R P.452-18.
// f:        frequency (GHz)
// p:        percentage of average year for which the calculated signal level is exceeded (%) [0.001, 50.0]
// latt:     latitude of transmitting station (degrees) [-90, 90]
// lont:     longitude of transmitting station, positive = East of Greenwich (degrees) [-180, 180]
// latr:     latitude of receiving station (degrees) [-90, 90]
// lonr:     longitude of receiving station, positive = East of Greenwich (degrees) [-180, 180]
// hts:      transmitting antenna centre height above mean sea level (m)
// hrs:      receiving antenna centre height above mean sea level (m)
// Gt:       transmitting antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// Gr:       receiving antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// n:        number of profile points
// d:        distance from transmitter profile of n points (km)
// h:        terrain height profile of n points (m above mean sea level)
// ae:       median effective Earth radius (km)
// thetat:   transmitter horizon elevation angle (mrad)
// thetar:   receiver horizon elevation angle (mrad)
// [return]: Lbs, basic transmission loss due to troposcatter (dB)
double ITURP_1812::_TroposcatterLoss(double f, double p, double latt, double lont, double latr, double lonr, double hts, double hrs,
                                     double Gt, double Gr, unsigned int n, double* d, double* h, double ae, double thetat, double thetar)
{
double Lbs;        // basic transmission loss due to troposcatter (dB)
double theta_e;    // angle subtended by d km at centre of spherical Earth (rad)
double theta_tpos; // horizon elevation angle relative to the local horizontal as viewed from the transmitter
                   // and limited to be positive (not less than zero) (mrad)
double theta_rpos; // horizon elevation angle relative to the local horizontal as viewed from the receiver
                   // and limited to be positive (not less than zero) (mrad)
double dtcv;       // horizontal path length from transmitter to common volume (km)
double phi_cvn;    // latitude of the troposcatter common volume (degrees)
double phi_cve;    // longitude of the troposcatter common volume (degrees)
double N0;         // average annual sea-level surface refractivity for the common volume (N-units)
double deltaN;     // radio-refractivity lapse rate for the common volume (N-units/km)
double theta;      // scatter angle (mrad)
double Lc;         // aperture-to-medium coupling loss (dB)
double hb = 7.35;  // global mean of the scale height (km)
double dn = d[n-1];

	// Step 1
	theta_e = dn/ae; // ITUR-P.2001-4, Section 3.5 of Annex, eq. (14)
	theta_tpos = std::max(0.0, thetat);
	theta_rpos = std::max(0.0, thetar);
	ITURP_2001::TroposphericScatterPathSegments(dn, theta_e, theta_tpos, theta_rpos, hts, hrs, latt, lont, latr, lonr,
												&dtcv, nullptr, &phi_cvn, &phi_cve, nullptr, nullptr, nullptr, nullptr, nullptr);
	N0 = ITURP_DigitalMaps::N050(phi_cvn, phi_cve);
	deltaN = ITURP_DigitalMaps::DN50(phi_cvn, phi_cve);

	// Step 2
	theta = (1000.0*dn/ae) + thetat + thetar;

	// Step 3
	Lc = 0.07*exp(0.055*(Gt+Gr));

	// Step 4
	double beta = (dn/(2.0*ae)) + (thetar/1000.0) + ((hrs-hts)/(1000.0*dn));
	double dsinB = dn*sin(beta);
	double sint = sin(theta/1000.0);
	double sintt = sin(thetat/1000.0);
	double h0 = (hts/1000.0) + ((dsinB/sint)*((dsinB/(2.0*ae*sint))+sintt));
	double Yp;
	if( p < 50.0 )
		Yp = 0.035*N0*exp(-h0/hb)*pow(-log10(p/50), 0.67);
	else
		Yp = -0.035*N0*exp(-h0/hb)*pow(-log10((100.0-p)/50), 0.67);
	double hs = _TerrainHeight(n, d, h, dtcv) / 1000.0;
	double F = 0.18*N0*exp(-hs/hb) - 0.23*deltaN;
	Lbs = F + 22.0*log10(f*1000.0) + 35.0*log10(theta) + 17.0*log10(dn) + Lc - Yp;

	return Lbs;
}

// ITU-R P.1812-6/7, Annex 1, Section 4.6
// p:        percentage of average year for which the calculated basic transmission loss is not exceeded (%) [1.0, 50.0]
// B0:       the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
// dn:       great-circle path distance between transmit and receive antennas (km)
// theta:    path angular distance (mrad)
// omega:    fraction of the total path over water
// Lb0p:     notional LoS basic transmission loss not exceeded for p% time (dB)
// Lb0B:     notional LoS basic transmission loss not exceeded for B0% time (dB)
// Ldp:      diffraction loss not exceeded for p% time (dB)
// Lbd50:    median basic transmission loss associated with diffraction (dB)
// Lba:      ducting/layer-reflection basic transmission loss not exceeded for for p% time (dB)
// Lbd:      basic transmission loss for diffraction not exceeded for p% time (dB)
// Lbs:      basic transmission loss due to troposcatter (dB)
// [return]: Lbc, basic transmission loss not exceeded for p% time and 50% locations (dB)
double ITURP_1812::_50PLocBasicTransmissionLoss(double p, double B0, double dn, double theta, double omega, double Lb0p, double Lb0B, double Ldp, double Lbd50,
                                                double Lba, double Lbd, double Lbs)
{
double Fj, Lbc;

	Fj = 1.0-0.5*(1.0+tanh(3.0*0.8*(theta-0.3)/0.3));
	Lbc  = ITURP_452_1812_common::_50PLocBasicTransmissionLoss(p, B0, dn, Fj, omega, Lb0p, Lb0B, Ldp, Lbd50, Lba, Lbd, Lbs);

	return Lbc;
}

// ITU-R P.1812-6/7, Annex 1, Sections 4.7 & 4.9
// f:            frequency (GHz)
// pL:           percentage of locations for which the calculated signal level is exceeded (%) [1.0, 99.0]
// hrg:          receive antenna centre height above ground level (m)
// R:            height of representative clutter at the receiver/mobile location (m)
// Lb0p:         basic transmission loss not exceeded for p% time and 50% locations associated with LoS with short term enhancements (dB)
// Lbc:          basic transmission loss not exceeded for p% of time and 50% locations,including the effects of terminal clutter losses (dB)
// [return]: Lb: basic transmission loss not exceeded for p% time and pL% locations (dB)
double ITURP_1812::_BasicTransmissionLoss(double f, double pL, double hrg, double R, double Lb0p, double Lbc)
{
double sigmaL, uh, Lloc, sigmaloc, Lb;

	sigmaL = (0.024*f+0.52)*pow(_wa, 0.28);
	if( hrg < R )
		uh = 1.0;
	else if( hrg < R+10.0 )
		uh = 1.0-(hrg-R)/10.0;
	else
		uh = 0.0;

	// assumes outdoors
	Lloc = 0.0;
	sigmaloc = uh*sigmaL;

	Lb = std::max(Lb0p, Lbc+Lloc-_I(pL/100.0)*sigmaloc);
	return Lb;
}

// ITU-R P.1812-6/7, Annex 1 Section 3.2 eq.(1c)
// n:             number of profile points
// h:             terrain height profile of n points (m above mean sea level)
// cc:            clutter category profile of n points
// rch:           representative clutter height profile of n points (m)
// sh:            surface height profile of n points (m)
// [out] gVector: vector of terrain height (m amsl) + representative clutter height (m) of n elements
// [out] R:       height of representative clutter at the receiver/mobile location (m)
void ITURP_1812::_gProfile(unsigned int n, double* h, ClutterCategory* cc, double* rch, double* sh, std::vector<double>& gVector, double& R)
{
	gVector.reserve(n);
	gVector.push_back(h[0]);
	if( sh != nullptr )
	{
		for(unsigned int i=1 ; i<n-1 ; i++)
			gVector.push_back( sh[i] );	

		R = sh[n-1] - h[n-1];	
	}
	else if( rch != nullptr )
	{
		for(unsigned int i=1 ; i<n-1 ; i++)
			gVector.push_back( h[i] + rch[i] );

		R = rch[n-1];
	}
	else if( cc != nullptr )
	{
		for(unsigned int i=1 ; i<n-1 ; i++)
			gVector.push_back( h[i] + _defaultRepClutterHeights[cc[i]] );

		R = _defaultRepClutterHeights[cc[n-1]];
	}
	else
	{
	double openRuralRepHeight = _defaultRepClutterHeights[ClutterCategory::OPEN_RURAL];

		for(unsigned int i=1 ; i<n-1 ; i++)
			gVector.push_back( h[i] + openRuralRepHeight );

		R = openRuralRepHeight;
	}

	gVector.push_back(h[n-1]);
}

// n:        number of profile points
// d:        distance from transmitter profile of n points (km)
// h:        terrain height profile of n points (m above mean sea level)
// distKm:   distance from transmitter (km)
// [return]: terrain height at distKm (m above mean sea level)
double ITURP_1812::_TerrainHeight(unsigned int n, double* d, double* h, double distKm)
{
double d0, d1, h0, h1;
double terrainHeight = 0;

	distKm = std::max(0.0, distKm);
	distKm = std::min(d[n-1], distKm);
	for(unsigned int i=0 ; i<n-1 ; i++)
	{
		d0 = d[i];
		d1 = d[i+1];
		if( distKm >= d0 && distKm <= d1 )
		{
			h0 = h[i];
			h1 = h[i+1];
			terrainHeight = (h0*(d1-distKm) + h1*(distKm-d0)) / (d1-d0);
			break;
		}
	}

	return terrainHeight;
}