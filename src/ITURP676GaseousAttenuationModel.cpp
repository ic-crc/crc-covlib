/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP676GaseousAttenuationModel.h"
#include "ITURP_676.h"
#include "ITURP_835.h"
#include "ITURP_DigitalMaps.h"
#include "ITURP_2001.h"
#include "CRC-COVLIB.h"
#include <cmath>

using namespace Crc::Covlib;


ITURP676GaseousAttenuationModel::ITURP676GaseousAttenuationModel()
{
	pIsActive = false;
	pAtmPressure_hPa = AUTOMATIC;
	pTemperature_K = AUTOMATIC;
	pWaterVapourDensity_gm3 = AUTOMATIC;
}

ITURP676GaseousAttenuationModel::~ITURP676GaseousAttenuationModel()
{

}

void ITURP676GaseousAttenuationModel::SetActiveState(bool active)
{
    pIsActive = active;
}

bool ITURP676GaseousAttenuationModel::IsActive() const
{
    return pIsActive;
}

void ITURP676GaseousAttenuationModel::SetAtmosphericPressure(double pressure_hPa)
{
	if( pIsAutomatic(pressure_hPa) || pressure_hPa > 0 )
		pAtmPressure_hPa = pressure_hPa;
}

double ITURP676GaseousAttenuationModel::GetAtmosphericPressure() const
{
	return pAtmPressure_hPa;
}

void ITURP676GaseousAttenuationModel::SetTemperature(double temperature_K)
{
	if( pIsAutomatic(temperature_K) || temperature_K > 0 )
		pTemperature_K = temperature_K;
}

double ITURP676GaseousAttenuationModel::GetTemperature() const
{
	return pTemperature_K;
}

void ITURP676GaseousAttenuationModel::SetWaterVapourDensity(double density_gm3)
{
	if( pIsAutomatic(density_gm3) || pWaterVapourDensity_gm3 > 0 )
		pWaterVapourDensity_gm3 = density_gm3;
}

double ITURP676GaseousAttenuationModel::GetWaterVapourDensity()
{
	return pWaterVapourDensity_gm3;
}

double ITURP676GaseousAttenuationModel::GaseousAttenuationPerKm(double frequency_GHz, double atmPressure_hPa, double temperature_K, double waterVapourDensity_gm3)
{
double att_dBperKm;

	if( frequency_GHz <= 0 || frequency_GHz > 1000 || atmPressure_hPa < 0 || temperature_K <= 0 || waterVapourDensity_gm3 < 0 )
		return 0;

	att_dBperKm = ITURP_676::AttenuationDueToDryAir(frequency_GHz, atmPressure_hPa, temperature_K, waterVapourDensity_gm3);
	att_dBperKm += ITURP_676::AttenuationDueToWaterVapour(frequency_GHz, atmPressure_hPa, temperature_K, waterVapourDensity_gm3);

	return att_dBperKm;
}

double ITURP676GaseousAttenuationModel::CalcGaseousAttenuation(double frequency_GHz, double txLat, double txLon, double rxLat, double rxLon,
                                                               double txRcagl_m, double rxRcagl_m, unsigned int sizeProfiles, double* distKmProfile,
                                                               double* elevProfile) const
{
double midPathLat, midPathLon;
double pathLength_km, hdiff_km, distFs_km;
double P_hPa = pAtmPressure_hPa;
double T_K = pTemperature_K;
double rho = pWaterVapourDensity_gm3;
double att_dBperKm;
double att_dB;

	if( frequency_GHz <= 0 || frequency_GHz > 1000 )
		return 0;

	pathLength_km = distKmProfile[sizeProfiles-1];

	ITURP_2001::GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, pathLength_km/2.0, midPathLat, midPathLon);

	if( pIsAutomatic(P_hPa) )
	    P_hPa = ITURP_835::StandardPressure(elevProfile[(int)(sizeProfiles/2.0)]/1000.0);
	if( pIsAutomatic(T_K) )
		T_K = ITURP_DigitalMaps::T_Annual(midPathLat, midPathLon);
	if( pIsAutomatic(rho) )
		rho = ITURP_DigitalMaps::Surfwv_50(midPathLat, midPathLon);

	att_dBperKm = ITURP_676::AttenuationDueToDryAir(frequency_GHz, P_hPa, T_K, rho);
	att_dBperKm += ITURP_676::AttenuationDueToWaterVapour(frequency_GHz, P_hPa, T_K, rho);

	// Using free-space distance calculation method from eq.(42) of ITU-R P.2001-5.
	hdiff_km = ((elevProfile[0]+txRcagl_m) - (elevProfile[sizeProfiles-1]+rxRcagl_m)) / 1000.0;
	distFs_km = sqrt((pathLength_km*pathLength_km) + (hdiff_km*hdiff_km));

	att_dB = att_dBperKm*distFs_km;

	return att_dB;
}

bool ITURP676GaseousAttenuationModel::pIsAutomatic(double param)
{
	return std::isnan(param);
}