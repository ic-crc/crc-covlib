/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_452_v18.h"
#include "ITURP_DigitalMaps.h"
#include "ITURP_835.h"
#include "ITURP_2001.h"
#include <cmath>


ITURP_452_v18::ITURP_452_v18()
{

}

ITURP_452_v18::~ITURP_452_v18()
{

}

void ITURP_452_v18::SetDefaultRepresentativeHeight(ClutterCategory clutterCategory, double representativeHeight)
{
	_SetDefaultRepresentativeHeight(clutterCategory, representativeHeight);
}

double ITURP_452_v18::GetDefaultRepresentativeHeight(ClutterCategory clutterCategory) const
{
	return _GetDefaultRepresentativeHeight(clutterCategory);
}

// f:          frequency (GHz) [0.1, 50.0]
// p:          required time percentage(s) for which the calculated basic transmission loss is not exceeded (%) [0.001, 50]
// worstMonth: true to indicate an average "worst month" prediction, false to indicate an average annual prediction
// latt:       latitude of interfering station (degrees) [-90, 90]
// lont:       longitude of interfering, positive = East of Greenwich (degrees) [-180, 180]
// latr:       latitude of interfered-width station (degrees) [-90, 90]
// lonr:       longitude of interfered-width station, positive = East of Greenwich (degrees) [-180, 180]
// htg:        interfering station's antenna centre height above ground level (m) [N/A, N/A]
// hrg:        interfered-with station's antenna centre height above ground level (m) [N/A, N/A]
// Gt:         interfering station's antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// Gr:         interfered-with station's antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// vpol:       true to indicate vertical polarization, false to indicate horizontal polarization
// n:          number of profile points
// d:          distance from interferer profile of n points (km) (distance limit of 10000 km)
// h:          terrain height profile of n points (m above mean sea level)
// cc:         clutter category profile of n points (if set to nullptr, OPEN_RURAL is assumed everywhere).
//             cc is not used if a representative clutter height profile is specified via the rch parameter.
// rch:        representative clutter height (m) profile of n points (if set to nullptr, default representative clutter height values
//             associated with clutter categories in cc are used)
// rcz:        radio climatic zone profile of n points (if set to nullptr, INLAND is assumed everywhere)
// dct, dcr:   If at least 75% of the path is over zone B (SEA) two further parameters are required, dct, dcr, giving the distance of
//             the transmitter and the receiver from the coast (km), respectively, in the direction of the other terminal.
//             For a terminal on a ship or sea platform the distance is zero. (If set to ITURP_452_1812::AUTO, values are automatically
//             estimated using the radio climate zone (rcz) and distance (d) profiles).
// P:          atmospheric pressure (if set to ITURP_452_1812::AUTO, its value is automatically obtained from ITU-R P.835) (hPa)
// TC:         temperature (if set to ITURP_452_1812::AUTO, its value is automatically obtained from ITU-R P.1510) (°C)
// deltaN:     average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere (if set to ITURP_452_1812::AUTO, its
//             value is automatically obtained from one of the Recommendation's digital map) (N-units/km)
// N0:         sea-level surface refractivity (if set to ITURP_452_1812::AUTO, its value is automatically obtained from one of the
//             Recommendation's digital map) (N-units)
// [return]:   basic transmission loss not exceeded for p% time
double ITURP_452_v18::ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double latt, double lont, double latr, double lonr, 
                                                    double htg, double hrg, double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h,
                                                    ClutterCategory* cc, double* rch, RadioClimaticZone* rcz, double dct, double dcr,
                                                    double P, double TC, double deltaN, double N0)
{
std::vector<double> gVector;
double* g; // g: terrain height (m amsl) + representative clutter height (m) profile of n points
std::vector<double> lopsVector;
double* lops; // lops: length of path sections profile of n points (km)
double hts; // hts: interferer antenna centre height above mean sea level (m)
double hrs; // hrs: interfered-with antenna centre height above mean sea level (m)
double dn; // dn: great-circle path distance (km)
double centreLat; // centreLat: latitude of the path centre between the interfering and interfered-with stations (degrees)
double centreLon; // centreLon: longitude of the path centre between the interfering and interfered-with stations (degrees)
double aB; // aB: effective Earth radius exceeded for B0 time (km)
double B0; // B0: the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
double ae; // ae: median effective Earth radius (km)
double omega; // omega: the fraction of the total path over water
double dtm; // dtm: longest continuous land (inland + coastal) section of the great-circle path (km)
double dlm; // dlm: longest continuous inland section of the great-circle path (km)
double TK; // TK: temperature (K)
std::vector<double> dc; // dc: distance from interferer profile in the height-gain model (km)
std::vector<double> hc; // hc: terrain height profile in the height-gain model (m above mean sea level)

double dlt; // dlt: interferer antenna horizon distance (km)
double dlr; // dlr: interfered-with antenna horizon distance (km)
double thetat; // thetat: interferer horizon elevation angle (mrad)
double thetar; // thetar: interfered-with horizon elevation angle (mrad)
double theta; // theta: path angular distance (mrad)
double hstd; // hstd: modified smooth-Earth surface at the interfering station location for the diffraction model (m amsl)
double hsrd; // hsrd: modified smooth-Earth surface at the interfered-with station location for the diffraction model (m amsl)
double hte; // hte: effective height of the interfering antenna for the ducting/layer-reflection model (m)
double hre; // hre: effective height of the interfered-with antenna for the ducting/layer-reflection model (m)
double hm; // hm: terrain roughness (m)

double Lbfsg; // Lbfsg: basic transmission loss due to free-space propagation and attenuation by atmospheric gases (dB)
double Lb0p; // Lb0p: basic transmission loss not exceeded for time percentage, p%, due to LoS propagation (dB)
double Lb0B; // Lb0B: basic transmission loss not exceeded for time percentage, B0%, due to LoS propagation (dB)
double Ldp; // Ldp: diffraction loss not exceeded for p% time (dB)
double Lbd50; // Lbd50: median basic transmission loss associated with diffraction (dB)
double Lbd; // Lbd: basic transmission loss associated with diffraction not exceeded for p% time (dB)
double Lbs; // Lbs: basic transmission loss due to troposcatter (dB)
double Lba; // Lba: the prediction of the basic transmission loss occurring during periods of anomalous propagation (ducting and layer reflection) (dB)
double Lb; // Lbc: final basic transmission loss not exceeded for p% time (dB)

	dn = d[n-1];
	_gProfile(n, d, h, cc, rch, gVector);
	g = gVector.data();
	_LengthOfPathSectionsProfile(n, d, lopsVector);
	lops = lopsVector.data();
	omega = _OverWaterPercent(dn, n, rcz, lops);
	if( omega >= 0.75 ) // otherwise dct and dcr are not used
	{
		if( _IsAUTO(dct) )
			dct = _Dct(n, rcz, lops);
		if( _IsAUTO(dcr) )
			dcr = _Dcr(n, rcz, lops);
	}
	_LongestContinuousLand(dn, n, rcz, lops, dtm, dlm);
    ITURP_2001::GreatCircleIntermediatePoint(latt, lont, latr, lonr, dn/2.0, centreLat, centreLon);
	B0 = _B0(centreLat, dtm, dlm);

	if( worstMonth )
		p = _AnnualEquivalentTimePercent(p, centreLat, omega);
	if( _IsAUTO(P) )
	{
		P = 1013.25;
		if( n > 0 )
			P = ITURP_835::StandardPressure(h[(int)(n/2.0)]/1000.0);
	}
	if( _IsAUTO(TC) )
		TK = ITURP_DigitalMaps::T_Annual(centreLat, centreLon);
	else
		TK = TC + 273.15;
	if( _IsAUTO(deltaN) )
		deltaN = ITURP_DigitalMaps::DN50(centreLat, centreLon);
	if( _IsAUTO(N0) )
		N0 = ITURP_DigitalMaps::N050(centreLat, centreLon);
	_EffectiveEarthRadius(deltaN, ae, aB);

	hts = h[0] + htg;
	hrs = h[n-1] + hrg;

	_PathProfileParameters(f, htg, hrg, n, d, h, ae, dlt, dlr, thetat, thetar, theta, hstd, hsrd, hte, hre, hm);

	_LoSPropagationLoss(f, p, B0, hts, hrs, dn, dlt, dlr, omega, P, TK, /*out*/Lbfsg, /*out*/Lb0p, /*out*/Lb0B);
	_DiffractionLoss(f, p, B0, hts, hrs, vpol, ae, aB, n, d, g, omega, hstd, hsrd, Lbfsg, Lb0p, /*out*/Ldp, /*out*/Lbd50, /*out*/Lbd);
	Lbs = _TroposcatterLoss(f, p, Gt, Gr, dn, theta, N0, P, TK);
	//Lbs = _TroposcatterLoss(f, p, latt, lont, latr, lonr, hts, hrs, Gt, Gr, n, d, h, ae, thetat, thetar);
	Lbs = std::max(Lbs, Lbfsg);
	Lba = _DuctingLayerReflectionLoss(f, p, ae, B0, dlm, dn, hts, hrs, dlt, dlr, thetat, thetar, hte, hre, hm, omega, dct, dcr, P, TK);
	Lb =  _OverallPrediction(p, B0, dn, omega, hts, hrs, ae, n, d, h, Lb0p, Lb0B, Ldp, Lbd50, Lba, Lbd, Lbs);

	return Lb;
}

// ITU-R P.452-18, Annex 1 Section 4.2.2.1 eq.(6e)
// n:             number of profile points
// d:             distance from interferer profile of n points (km)
// h:             terrain height profile of n points (m above mean sea level)
// cc:            clutter category profile of n points
// rch:           representative clutter height profile of n points (m)
// [out] gVector: vector of terrain height (m amsl) + representative clutter height (m) of n elements
void ITURP_452_v18::_gProfile(unsigned int n, double* d, double* h, ClutterCategory* cc, double* rch, std::vector<double>& gVector)
{
double dn = d[n-1];
double graceDistKm = 0.050;
double openRuralRepHeight = _defaultRepClutterHeights[ClutterCategory::OPEN_RURAL];
double gi;

	gVector.reserve(n);
	gVector.push_back(h[0]);
	for(unsigned int i=1 ; i<n-1 ; i++)
	{
		gi = h[i];
		if( !(d[i] < graceDistKm || d[i] > dn-graceDistKm) )
		{
			if( rch != nullptr )
				gi += rch[i];
			else if( cc != nullptr )
				gi += _defaultRepClutterHeights[cc[i]];
			else
				gi += openRuralRepHeight;
		}
		gVector.push_back(gi);
	}
	gVector.push_back(h[n-1]);
}
