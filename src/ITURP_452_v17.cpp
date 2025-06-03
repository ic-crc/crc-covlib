/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP_452_v17.h"
#include "ITURP_2001.h"
#include "ITURP_DigitalMaps.h"
#include "ITURP_835.h"
#include <cmath>
#include <vector>
#include <cstring>


ITURP_452_v17::ITURP_452_v17()
{

}

ITURP_452_v17::~ITURP_452_v17()
{

}

// f:          frequency (GHz) [0.1, 50.0]
// p:          required time percentage(s) for which the calculated basic transmission loss is not exceeded (%) [0.001, 50]
// worstMonth: true to indicate an average "worst month" prediction, false to indicate an average annual prediction
// lat:        latitude of the path centre between the interfering and interfered-with stations (degrees) [-90, 90]
// lon:        longitude of the path centre between the interfering and interfered-with stations (degrees) [-180, 180]
// htg:        interfering station's antenna centre height above ground level (m) [N/A, N/A]
// hrg:        interfered-with station's antenna centre height above ground level (m) [N/A, N/A]
// Gt:         interfering station's antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// Gr:         interfered-with station's antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// vpol:       true to indicate vertical polarization, false to indicate horizontal polarization
// n:          number of profile points
// d:          distance from interferer profile of n points (km) (distance limit of 10000 km)
// h:          terrain height profile of n points (m above mean sea level)
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
// dkt:        distance from nominal clutter point to the interfering station's antenna (km)
// dkr:        distance from nominal clutter point to the interfered-with station's antenna (km)
// hat:        nominal clutter height above local ground level at the interfering station's location (m)
// har:        nominal clutter height above local ground level at the interfered-with station's location (m)
// [return]:   basic transmission loss not exceeded for p% time
double ITURP_452_v17::ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double lat, double lon,double htg, double hrg, 
                                                    double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h, RadioClimaticZone* rcz,
                                                    double dct, double dcr, double P, double TC, double deltaN, double N0,
                                                    double dkt, double dkr, double hat, double har)
{
std::vector<double> lopsVector;
double* lops; // lops: length of path sections profile of n points (km)
double hts; // hts: interferer antenna centre height above mean sea level (m)
double hrs; // hrs: interfered-with antenna centre height above mean sea level (m)
double dn; // dn: great-circle path distance (km)
double aB; // aB: effective Earth radius exceeded for B0 time (km)
double B0; // B0: the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
double ae; // ae: median effective Earth radius (km)
double omega; // omega: the fraction of the total path over water
double dtm; // longest continuous land (inland + coastal) section of the great-circle path (km)
double dlm; // longest continuous inland section of the great-circle path (km)
double TK; // temperature (K)
double htgc; // interfering station's antenna centre height above ground level in the height-gain model (m)
double hrgc; // interfered-with station's antenna centre height above ground level in the height-gain model (m)
std::vector<double> dc; // distance from interferer profile in the height-gain model (km)
std::vector<double> hc; // terrain height profile in the height-gain model (m above mean sea level)

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
double Aht; // Aht: the additional loss at the interfering antenna due to protection from local clutter (dB)
double Ahr; // Ahr: the additional loss at the interfered-with antenna due to protection from local clutter (dB)
double Lb; // Lbc: final basic transmission loss not exceeded for p% time (dB)

	dn = d[n-1];
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
	B0 = _B0(lat, dtm, dlm);
	if( worstMonth )
		p = _AnnualEquivalentTimePercent(p, lat, omega);
	if( _IsAUTO(P) )
	{
		P = 1013.25;
		if( n > 0 )
			P = ITURP_835::StandardPressure(h[(int)(n/2.0)]/1000.0);
	}
	if( _IsAUTO(TC) )
		TK = ITURP_DigitalMaps::T_Annual(lat, lon);
	else
		TK = TC + 273.15;
	if( _IsAUTO(deltaN) )
		deltaN = ITURP_DigitalMaps::DN50(lat, lon);
	if( _IsAUTO(N0) )
		N0 = ITURP_DigitalMaps::N050(lat, lon);
	_EffectiveEarthRadius(deltaN, ae, aB);

	Aht = Ahr = 0;
	if( hat > 0 || har > 0 )
	{
		_AdditionalClutterLosses(f, htg, hrg, n, d, h, dkt, dkr, hat, har, /*out*/Aht, /*out*/Ahr, /*out*/htgc, /*out*/hrgc, /*out*/dc, /*out*/ hc);
		htg = htgc;
		hrg = hrgc;
		n = dc.size();
		d = dc.data();
		h = hc.data();
		dn = d[n-1];
	}

	hts = h[0] + htg;
	hrs = h[n-1] + hrg;

	_PathProfileParameters(f, htg, hrg, n, d, h, ae, dlt, dlr, thetat, thetar, theta, hstd, hsrd, hte, hre, hm);

	_LoSPropagationLoss(f, p, B0, hts, hrs, dn, dlt, dlr, omega, P, TK, /*out*/Lbfsg, /*out*/Lb0p, /*out*/Lb0B);
	_DiffractionLoss(f, p, B0, hts, hrs, vpol, ae, aB, n, d, h, omega, hstd, hsrd, Lbfsg, Lb0p, /*out*/Ldp, /*out*/Lbd50, /*out*/Lbd);
	Lbs = _TroposcatterLoss(f, p, Gt, Gr, dn, theta, N0, P, TK);
	Lba = _DuctingLayerReflectionLoss(f, p, ae, B0, dlm, dn, hts, hrs, dlt, dlr, thetat, thetar, hte, hre, hm, omega, dct, dcr, P, TK);
	Lb =  Aht + Ahr + _OverallPrediction(p, B0, dn, omega, hts, hrs, ae, n, d, h, Lb0p, Lb0B, Ldp, Lbd50, Lba, Lbd, Lbs);

	return Lb;
}

// See ClearAirBasicTransmissionLoss() above for more details
// latt:       latitude of interfering station (degrees) [-90, 90]
// lont:       longitude of interfering, positive = East of Greenwich (degrees) [-180, 180]
// latr:       latitude of interfered-width station (degrees) [-90, 90]
// lonr:       longitude of interfered-width station, positive = East of Greenwich (degrees) [-180, 180]
double ITURP_452_v17::ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double latt, double lont, double latr, double lonr,
                                                    double htg, double hrg, double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h,
                                                    RadioClimaticZone* rcz, double dct, double dcr, double P, double TC, double deltaN, double N0,
                                                    double dkt, double dkr, double hat, double har)
{
double centreLat, centreLon;

	ITURP_2001::GreatCircleIntermediatePoint(latt, lont, latr, lonr, d[n-1]/2.0, centreLat, centreLon);
	return ClearAirBasicTransmissionLoss(f, p, worstMonth, centreLat, centreLon, htg, hrg, Gt, Gr, vpol, n, d, h, rcz, dct, dcr, P, TC, deltaN, N0, dkt, dkr, hat, har);
}

// ITU-R P.452-17, Annex 1, Section 4.5.3
// f:        frequency (GHz)
// h:        antenna height above local ground level (m)
// dk:       distance from nominal clutter point to the antenna (km)
// ha:       nominal clutter height above local ground level (m)
// [return]: additional loss due to protection from local clutter (dB)
double ITURP_452_v17::_AdditionalClutterLosses(double f, double h, double dk, double ha)
{
double Ah = 0;
double Ffc;

	if( ha > 0 )
	{
		Ffc = 0.25+0.375*(1.0+tanh(7.5*(f-0.5)));
		Ah = 10.25*Ffc*exp(-dk)*(1.0-tanh(6.0*(h/ha-0.625)))-0.33;
	}
	return Ah;
}

// ITU-R P.452-17, Annex 1, Section 4.5
// f:          frequency (GHz)
// htg:        interfering station's antenna centre height above ground level (m)
// hrg:        interfered-with station's antenna centre height above ground level (m)
// n:          number of profile points
// d:          distance from interferer profile of n points (km)
// h:          terrain height profile of n points (m above mean sea level)
// dkt:        distance from nominal clutter point to the interfering station's antenna (km)
// dkr:        distance from nominal clutter point to the interfered-with station's antenna (km)
// hat:        nominal clutter height above local ground level at the interfering station's location (m)
// har:        nominal clutter height above local ground level at the interfered-with station's location (m)
// [out] Aht:  additional loss due to protection from local clutter at the interfering station's location (dB)
// [out] Ahr:  additional loss due to protection from local clutter at the interfered-with station's location (dB)
// [out] htgc: interfering station's antenna centre height above ground level in the height-gain model (m)
// [out] hrgc: interfered-with station's antenna centre height above ground level in the height-gain model (m)
// [out] dc:   distance from interferer profile in the height-gain model (km)
// [out] hc:   terrain height profile in the height-gain model (m above mean sea level)
void ITURP_452_v17::_AdditionalClutterLosses(double f, double htg, double hrg, unsigned int n, double* d, double* h, double dkt, double dkr, double hat, double har, 
                                             double& Aht, double& Ahr, double& htgc, double& hrgc, std::vector<double>& dc, std::vector<double>& hc)
{
// Note: The recommendation mentions the path profiles to use should be from the nominal clutter locations
//       instead of from the stations' locations.
unsigned int index1 = 0;
unsigned int index2 = n-1;

	Aht = 0;
	Ahr = 0;
	htgc = htg;
	hrgc = hrg;

	if( hat > htg )
	{
		Aht = _AdditionalClutterLosses(f, htg, dkt, hat);

		index1 = n-1;
		for(unsigned int i=0 ; i<n ; i++)
		{
			if( d[i] >= dkt )
			{
				index1 = i;
				break;
			}
		}
	}

	if( har > hrg )
	{
		Ahr = _AdditionalClutterLosses(f, hrg, dkr, har);

		index2 = 0;
		for(int i=((int)n)-1 ; i>=0 ; i--)
		{
			if( d[i] <= d[n-1]-dkr )
			{
				index2 = (unsigned int)i;
				break;	
			}
		}
	}

	if( ((int)index2) - ((int)index1) < 3 ) // at least two points between the clutter at Tx and Rx sides
		Aht = Ahr = 0;

	if( Aht > 0 || Ahr > 0 )
	{
	double delta, nc;

		if( Aht > 0 )
			htgc = hat;
		if( Ahr > 0 )
			hrgc = har;

		nc = index2 - index1 + 1;

		hc.resize(nc);
		std::memcpy(hc.data(), &(h[index1]), nc*sizeof(double));

		delta = d[index1];
		dc.resize(nc);
		std::memcpy(dc.data(), &(d[index1]), nc*sizeof(double));
		for(unsigned int i=0 ; i<nc ; i++)
			dc[i] = dc[i] - delta;
	}
	else
	{
		dc.resize(n);
		std::memcpy(dc.data(), d, n*sizeof(double));
		hc.resize(n);
		std::memcpy(hc.data(), h, n*sizeof(double));
	}
}
