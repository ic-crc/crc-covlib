/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

// Implements calculations that are common to both ITU-R P.452-17 and ITU-R P.452-18
#include "ITURP_452_common.h"
#include "ITURP_2001.h"
#include "ITURP_676.h"
#include <cmath>
#include <limits>


ITURP_452_common::ITURP_452_common()
{

}

ITURP_452_common::~ITURP_452_common()
{

}

// ITU-R P.452-17/18, Annex 1, Section 3.2.1 (Step 2)
// pw:       worst-month time percentage for which particular values of basic transmission loss are not exceeded
// lat:      latitude of the path's centre point (degrees) 
// omega:    the fraction of the total path over water
// [return]: p, the annual equivalent time percentage of the worst-month time percentage
double ITURP_452_common::_AnnualEquivalentTimePercent(double pw, double lat, double omega)
{
double p;
double absLatDeg = fabs(lat);
double latRad = lat*ITURP_2001::PI_ON_180;
double GL;

	if( absLatDeg > 45 )
		GL = sqrt(1.1-pow(fabs(cos(2*latRad)), 0.7));
	else
		GL = sqrt(1.1+pow(fabs(cos(2*latRad)), 0.7));
	p = pow(10.0, (log10(pw)+log10(GL)-(0.186*omega)-0.444)/(0.816+(0.078*omega)));
	p = std::max(p, pw/12.0);

	return p; 
}

// ITU-R P.452-17/18, Annex 1, Section 4.1
// f:        frequency (GHz)
// hts:      transmit antenna centre height above mean sea level (m)
// hrs:      receive antenna centre height above mean sea level (m)
// dn:       great-circle path distance (km)
// omega:    the fraction of the total path over water
// rho:      water vapour density (g/m3)
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// [return]: Ag, the total gaseous absorption (dB)
double ITURP_452_common::_TotalGaseousAbsorption(double f, double hts, double hrs, double dn, double rho, double P, double TK)
{
double Ag;
double dfs, dh;
double yo, yw;

	dh = (hts-hrs)/1000.0;
	dfs = sqrt(dn*dn + dh*dh);
	yo = ITURP_676::AttenuationDueToDryAir(f, P, TK, rho);
	yw = ITURP_676::AttenuationDueToWaterVapour(f, P, TK, rho);
	Ag = (yo+yw)*dfs;

	return Ag;
}

// ITU-R P.452-17/18, Annex 1, Section 4.1
// f:           frequency (GHz)
// p:           percentage of average year for which the calculated signal level is exceeded (%)
// B0:          the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
// hts:         transmit antenna centre height above mean sea level (m)
// hrs:         receive antenna centre height above mean sea level (m)
// dn:          great-circle path distance (km)
// dlt:         transmitting antenna horizon distance (km)
// dlr:         receiving antenna horizon distance (km)
// omega:       the fraction of the total path over water
// P:           atmospheric pressure (hPa)
// TK:          temperature (K)
// [out] Lbfsg: basic transmission loss due to free-space propagation and attenuation by atmospheric gases (dB)
// [out] Lb0p:  basic transmission loss not exceeded for time percentage, p%, due to LoS propagation (dB)
// [out] Lb0B:  basic transmission loss not exceeded for time percentage, B0%, due to LoS propagation (dB)
// [out] Ag:    total gaseous absorption (dB)
void ITURP_452_common::_LoSPropagationLoss(double f, double p, double B0, double hts, double hrs, double dn, double dlt, double dlr,
                                           double omega, double P, double TK, double& Lbfsg, double& Lb0p, double& Lb0B)
{
double Lbfs;
double rho = 7.5+(2.5*omega);
double Ag;

	ITURP_452_1812_common::_LoSPropagationLoss(f, p, B0, hts, hrs, dn, dlt, dlr, Lbfs, Lb0p, Lb0B);
	Ag = _TotalGaseousAbsorption(f, hts, hrs, dn, rho, P, TK);
	Lbfsg = Lbfs + Ag;
	Lb0p += Ag;
	Lb0B += Ag;
}

// ITU-R P.452-17/18, Annex 1, Section 4.3
// f:        frequency (GHz)
// p:        percentage of average year for which the calculated signal level is exceeded (%) [0.001, 50.0]
// Gt:       transmitter antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// Gr:       receiver antenna gain in the direction of the horizon along the great-circle interference path (dBi)
// dn:       great-circle path distance between transmit and receive antennas (km)
// theta:    path angular distance (mrad)
// N0:       sea-level surface refractivity (N-units)
// P:        atmospheric pressure (hPa)
// TK:       temperature (K)
// [return]: Lbs, basic transmission loss due to troposcatter (dB)
double ITURP_452_common::_TroposcatterLoss(double f, double p, double Gt, double Gr, double dn, double theta, double N0, double P, double TK)
{
double Lf, Lbs, Lc, Ag;

	Lf = 25.0*log10(f) - 2.5*log10(f/2.0)*log10(f/2.0);
	Lc = 0.051*exp(0.055*(Gt+Gr));

	// Note: The official ITU reference implementation only uses hts and hrs in Ag's calculation
	//       for the line-of-sight propagation losses. It is not that clear from the recommendation
	//       document itself whether hts and hrs should be used here ; in any case, the impact on the
	//       overall result is negligible.
	Ag = _TotalGaseousAbsorption(f, 0 /*hts*/, 0 /*hrs*/, dn, 3.0, P, TK);

	Lbs = 190.0 + Lf + 20.0*log10(dn) + 0.573*theta - 0.15*N0 + Lc + Ag - 10.1*pow(-log10(p/50.0), 0.7);

	return Lbs;
}

// ITU-R P.452-17/18, Annex 1, Section 4.4
// f:            frequency (GHz)
// p:            percentage of average year for which the calculated signal level is exceeded (%)
// ae:           median effective Earth radius (km)
// B0:           the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
// dlm:          longest continuous inland section of the great-circle path (km)
// dn:           great-circle path distance (km)
// hts:          transmit antenna centre height above mean sea level (m)
// hrs:          receive antenna centre height above mean sea level (m)
// dlt:          transmitting antenna horizon distance (km)
// dlr:          receiving antenna horizon distance (km)
// thetat:       transmit horizon elevation angle (mrad)
// thetar:       receive horizon elevation angle (mrad)
// hte:          effective height of the transmitting antenna for the ducting/layer-reflection model (m)
// hre:          effective height of the receiving antenna for the ducting/layer-reflection model (m)
// hm:           terrain roughness (m)
// omega:        fraction of the total path over water
// dct:          distance of the transmitter from the coast in the direction of the receiver. For a terminal on a ship or sea platform the distance is zero (km)
// dcr:          distance of the receiver from the coast in the direction of the transmitter. For a terminal on a ship or sea platform the distance is zero (km)
// P:            atmospheric pressure (hPa)
// TK:           temperature (K)
// [return] Lba: basic transmission loss associated with ducting/layer-reflection not exceeded for p% time (dB)
double ITURP_452_common::_DuctingLayerReflectionLoss(double f, double p, double ae, double B0, double dlm, double dn, double hts, double hrs,
                                                     double dlt, double dlr, double thetat, double thetar, double hte, double hre, double hm,
                                                     double omega, double dct, double dcr, double P, double TK)
{
double Ag;
double rho = 7.5+(2.5*omega);
double Lba;

	// Note: The official ITU reference implementation only uses hts and hrs in Ag's calculation
	//       for the line-of-sight propagation losses. It is not that clear from the recommendation
	//       document itself whether hts and hrs should be used here ; in any case, the impact on the
	//       overall result is negligible.
	Ag = _TotalGaseousAbsorption(f, 0 /*hts*/, 0 /*hrs*/, dn, rho, P, TK);

	Lba = Ag + ITURP_452_1812_common::_DuctingLayerReflectionLoss(f, p, ae, B0, dlm, dn, hts, hrs, dlt, dlr, thetat, thetar, hte, hre, hm, omega, dct, dcr);

	return Lba;
}

// ITU-R P.452-17, Annex 1, Section 4.6 (excluding Aht and Ahr (clutter losses))
// ITU-R P.452-18, Annex 1, Section 4.5
// p:        percentage of average year for which the calculated signal level is exceeded (%) [1.0, 50.0]
// B0:       the time percentage for which refractivity lapse-rates exceeding 100 N-units/km can be expected in the first 100 m of the lower atmosphere (%)
// dn:       great-circle path distance between transmit and receive antennas (km)
// omega:    fraction of the total path over water
// htc:      transmitting (interferer) antenna height (m)
// hrc:      receiving (interfered-with) antenna height (m)
// ae:       median effective Earth radius (km)
// n:        number of profile points
// d:        distance from transmitter profile of n points (km)
// h:        terrain height profile of n points (m amsl)
// Lb0p:     notional LoS basic transmission loss not exceeded for p% time (dB)
// Lb0B:     notional LoS basic transmission loss not exceeded for B0% time (dB)
// Ldp:      diffraction loss not exceeded for p% time (dB)
// Lbd50:    median basic transmission loss associated with diffraction (dB)
// Lba:      ducting/layer-reflection basic transmission loss not exceeded for for p% time (dB)
// Lbd:      basic transmission loss for diffraction not exceeded for p% time (dB)
// Lbs:      basic transmission loss due to troposcatter (dB)
// [return]: Lbc, basic transmission loss not exceeded for p% time and 50% locations (dB)
double ITURP_452_common::_OverallPrediction(double p, double B0, double dn, double omega, double htc, double hrc, double ae, 
                                            unsigned int n, double* d, double* h, double Lb0p, double Lb0B, double Ldp,
                                            double Lbd50, double Lba, double Lbd, double Lbs)
{
double Fj, Stim, Str;
double Ce = 1.0/ae;
double Lb;

	Stim = std::numeric_limits<double>::lowest();
	for(unsigned int i=1 ; i<n-1 ; i++)
		Stim = std::max(Stim, (h[i] + 500.0*Ce*d[i]*(dn-d[i])-htc)/d[i]);
	Str = (hrc-htc)/dn;
	Fj = 1.0-0.5*(1.0+tanh(3.0*0.8*(Stim-Str)/0.3));

	Lb = ITURP_452_1812_common::_50PLocBasicTransmissionLoss(p, B0, dn, Fj, omega, Lb0p, Lb0B, Ldp, Lbd50, Lba, Lbd, Lbs);

	return Lb;
}