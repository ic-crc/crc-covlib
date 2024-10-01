#include "ITURP452v17PropagModel.h"
#include <cmath>

using namespace Crc::Covlib;


ITURP452v17PropagModel::ITURP452v17PropagModel()
{
	pTimePercent = 50;
	pPredictionType = P452_AVERAGE_YEAR;
	pDeltaN = AUTOMATIC;
	pN0 = AUTOMATIC;
	pTemperature_C = AUTOMATIC;
	pPressure_hPa = AUTOMATIC;
	pHeightGainModelModeAtTransmitter = P452_USE_CLUTTER_AT_ENDPOINT;
	pHeightGainModelModeAtReceiver    = P452_USE_CLUTTER_AT_ENDPOINT;
	pClutterProfileUsageLimitKm = 0.100;

	pClutterParams[P452_HGM_HIGH_CROP_FIELDS]                    = {4.0,  0.1};
	pClutterParams[P452_HGM_PARK_LAND]                           = {4.0,  0.1};
	pClutterParams[P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES]     = {4.0,  0.1};
	pClutterParams[P452_HGM_ORCHARD_REGULARLY_SPACED]            = {4.0,  0.1};
	pClutterParams[P452_HGM_SPARSE_HOUSES]                       = {4.0,  0.1};
	pClutterParams[P452_HGM_VILLAGE_CENTRE]                      = {5.0,  0.07};
	pClutterParams[P452_HGM_DECIDUOUS_TREES_IRREGULARLY_SPACED]  = {15.0, 0.05};
	pClutterParams[P452_HGM_DECIDUOUS_TREES_REGULARLY_SPACED]    = {15.0, 0.05};
	pClutterParams[P452_HGM_MIXED_TREE_FOREST]                   = {15.0, 0.05};
	pClutterParams[P452_HGM_CONIFEROUS_TREES_IRREGULARLY_SPACED] = {20.0, 0.05};
	pClutterParams[P452_HGM_CONIFEROUS_TREES_REGULARLY_SPACED]   = {20.0, 0.05};
	pClutterParams[P452_HGM_TROPICAL_RAIN_FOREST]                = {20.0, 0.03};
	pClutterParams[P452_HGM_SUBURBAN]                            = {9.0,  0.025};
	pClutterParams[P452_HGM_DENSE_SUBURBAN]                      = {12.0, 0.02};
	pClutterParams[P452_HGM_URBAN]                               = {20.0, 0.02};
	pClutterParams[P452_HGM_DENSE_URBAN]                         = {25.0, 0.02};
	pClutterParams[P452_HGM_HIGH_RISE_URBAN]                     = {35.0, 0.02};
	pClutterParams[P452_HGM_INDUSTRIAL_ZONE]                     = {20.0, 0.05};
	pClutterParams[P452_HGM_OTHER]                               = {0.0,  0.0};
	pClutterParams[P452_HGM_CUSTOM_AT_TRANSMITTER]               = {0.0,  0.0};
	pClutterParams[P452_HGM_CUSTOM_AT_RECEIVER]                  = {0.0,  0.0};
}

ITURP452v17PropagModel::~ITURP452v17PropagModel()
{

}

PropagationModel ITURP452v17PropagModel::Id()
{
	return ITU_R_P_452_V17;
}

bool ITURP452v17PropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool ITURP452v17PropagModel::IsUsingMappedLandCoverData()
{
	if( pHeightGainModelModeAtTransmitter == P452_USE_CLUTTER_AT_ENDPOINT || pHeightGainModelModeAtTransmitter == P452_USE_CLUTTER_PROFILE )
		return true;
	if( pHeightGainModelModeAtReceiver == P452_USE_CLUTTER_AT_ENDPOINT || pHeightGainModelModeAtReceiver == P452_USE_CLUTTER_PROFILE )
		return true;
	return false;
}

bool ITURP452v17PropagModel::IsUsingItuRadioClimZoneData()
{
	return true;
}

bool ITURP452v17PropagModel::IsUsingSurfaceElevData()
{
	return false;
}

int ITURP452v17PropagModel::DefaultMappedLandCoverValue()
{
	return P452_HGM_OTHER;
}

double ITURP452v17PropagModel::CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m, 
                                            double txAntGain_dBi, double rxAntGain_dBi, Polarization pol, unsigned int sizeProfiles, double* distKmProfile,
                                            double* elevProfile, ITURadioClimaticZone* radioClimaticZoneProfile,
                                            P452HeightGainModelClutterCategory* clutterCatProfile/*=nullptr*/)
{
double dkt = 0;
double dkr = 0;
double hat = 0;
double har = 0;

	if( sizeProfiles < 3 )
		return 0;
	if( distKmProfile[sizeProfiles-1] < 1E-5 )
		return 0;

	if( pHeightGainModelModeAtTransmitter == P452_USE_CUSTOM_AT_CATEGORY )
	{
		dkt = pClutterParams[P452_HGM_CUSTOM_AT_TRANSMITTER].nominalDist_km;
		hat = pClutterParams[P452_HGM_CUSTOM_AT_TRANSMITTER].nominalHeight_m;
	}
	else if( pHeightGainModelModeAtTransmitter == P452_USE_CLUTTER_PROFILE && clutterCatProfile != nullptr )
	{
	ClutterParams clutterParams;

		for(unsigned int i=0 ; i<sizeProfiles ; i++)
		{
			if( distKmProfile[i] <= pClutterProfileUsageLimitKm )
			{
				clutterParams = pClutterParams[clutterCatProfile[i]];
				if( clutterParams.nominalHeight_m > hat )
				{
					hat = clutterParams.nominalHeight_m;
					dkt = clutterParams.nominalDist_km;
				}
			}
			else
				break;
		}
	}
	else if( pHeightGainModelModeAtTransmitter == P452_USE_CLUTTER_AT_ENDPOINT && clutterCatProfile != nullptr && sizeProfiles > 0 )
	{
	ClutterParams clutterParams = pClutterParams[clutterCatProfile[0]];

		hat = clutterParams.nominalHeight_m;
		dkt = clutterParams.nominalDist_km;
	}

	if( pHeightGainModelModeAtReceiver == P452_USE_CUSTOM_AT_CATEGORY )
	{
		dkr = pClutterParams[P452_HGM_CUSTOM_AT_RECEIVER].nominalDist_km;
		har = pClutterParams[P452_HGM_CUSTOM_AT_RECEIVER].nominalHeight_m;
	}
	else if( pHeightGainModelModeAtReceiver == P452_USE_CLUTTER_PROFILE && clutterCatProfile != nullptr )
	{
	ClutterParams clutterParams;
	double distFromRxKm;

		for(int i=((int)sizeProfiles)-1 ; i>=0 ; i--)
		{
			distFromRxKm = distKmProfile[sizeProfiles-1] - distKmProfile[i];
			if( distFromRxKm <= pClutterProfileUsageLimitKm )
			{
				clutterParams = pClutterParams[clutterCatProfile[i]];
				if( clutterParams.nominalHeight_m > har )
				{
					har = clutterParams.nominalHeight_m;
					dkr = clutterParams.nominalDist_km;
				}
			}
			else
				break;
		}
	}
	else if( pHeightGainModelModeAtReceiver == P452_USE_CLUTTER_AT_ENDPOINT && clutterCatProfile != nullptr && sizeProfiles > 0 )
	{
	ClutterParams clutterParams = pClutterParams[clutterCatProfile[sizeProfiles-1]];

		har = clutterParams.nominalHeight_m;
		dkr = clutterParams.nominalDist_km;
	}

	static_assert(std::isnan(Crc::Covlib::AUTOMATIC) && std::isnan(ITURP_452_1812_common::AUTO), "");

	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_COASTAL_LAND == (int)ITURP_452_1812_common::RadioClimaticZone::COASTAL_LAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_INLAND == (int)ITURP_452_1812_common::RadioClimaticZone::INLAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_SEA == (int)ITURP_452_1812_common::RadioClimaticZone::SEA, "");

	return ClearAirBasicTransmissionLoss(freq_Ghz, pTimePercent, pPredictionType==P452_WORST_MONTH, txLat, txLon, rxLat, rxLon, txRcagl_m, rxRcagl_m,
	                                     txAntGain_dBi, rxAntGain_dBi, pol==VERTICAL_POL, sizeProfiles, distKmProfile, elevProfile,
	                                     (RadioClimaticZone*)radioClimaticZoneProfile, /*dct*/AUTO, /*dcr*/AUTO, pPressure_hPa, pTemperature_C,
	                                     pDeltaN, pN0,  dkt, dkr, hat, har);
}

void ITURP452v17PropagModel::SetTimePercentage(double percent)
{
	if( percent < 0.001 || percent > 50.0 )
		return;
	pTimePercent = percent;
}
	
double ITURP452v17PropagModel::GetTimePercentage() const
{
	return pTimePercent;
}

void ITURP452v17PropagModel::SetPredictionType(P452PredictionType predictionType)
{
	if( predictionType != P452_AVERAGE_YEAR && predictionType != P452_WORST_MONTH )
		return;
	pPredictionType = predictionType;
}

P452PredictionType ITURP452v17PropagModel::GetPredictionType() const
{
	return pPredictionType;
}

void ITURP452v17PropagModel::SetAverageRadioRefractivityLapseRate(double deltaN)
{
	if( pIsAutomatic(deltaN) || deltaN > 0 )
		pDeltaN = deltaN;
}
	
double ITURP452v17PropagModel::GetAverageRadioRefractivityLapseRate() const
{
	return pDeltaN;
}

void ITURP452v17PropagModel::SetSeaLevelSurfaceRefractivity(double N0)
{
	if( pIsAutomatic(N0) || N0 > 0 )
		pN0 = N0;
}
	
double ITURP452v17PropagModel::GetSeaLevelSurfaceRefractivity() const
{
	return pN0;
}

void ITURP452v17PropagModel::SetAirTemperature(double temperature_C)
{
	if( pIsAutomatic(temperature_C) || temperature_C >= -273.15 )
		pTemperature_C = temperature_C;
}
	
double ITURP452v17PropagModel::GetAirTemperature() const
{
	return pTemperature_C;
}

void ITURP452v17PropagModel::SetAirPressure(double pressure_hPa)
{
	if( pIsAutomatic(pressure_hPa) || pressure_hPa > 0 )
		pPressure_hPa = pressure_hPa;
}
	
double ITURP452v17PropagModel::GetAirPressure() const
{
	return pPressure_hPa;
}

void ITURP452v17PropagModel::SetNominalClutterParams(P452HeightGainModelClutterCategory clutterCategory, double nominalHeight_m, double nominalDist_km)
{
	pClutterParams[clutterCategory] = {nominalHeight_m, nominalDist_km};
}

void ITURP452v17PropagModel::SetNominalClutterParams(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory, ClutterParams nominalClutterParams)
{
	pClutterParams[clutterCategory] = nominalClutterParams;
}

ITURP452v17PropagModel::ClutterParams ITURP452v17PropagModel::GetNominalClutterParams(P452HeightGainModelClutterCategory clutterCategory) const
{
	if( pClutterParams.count(clutterCategory) == 1 )
		return pClutterParams.at(clutterCategory);
	return {0.0, 0.0};
}

void ITURP452v17PropagModel::SetTransmitterHeightGainModelMode(P452HeightGainModelMode mode)
{
	if( mode < P452_NO_SHIELDING || mode > P452_USE_CLUTTER_AT_ENDPOINT )
		return;
	pHeightGainModelModeAtTransmitter = mode;
}
	
P452HeightGainModelMode ITURP452v17PropagModel::GetTransmitterHeightGainModelMode() const
{
	return pHeightGainModelModeAtTransmitter;
}

void ITURP452v17PropagModel::SetReceiverHeightGainModelMode(P452HeightGainModelMode mode)
{
	if( mode < P452_NO_SHIELDING || mode > P452_USE_CLUTTER_AT_ENDPOINT )
		return;
	pHeightGainModelModeAtReceiver = mode;
}
	
P452HeightGainModelMode ITURP452v17PropagModel::GetReceiverHeightGainModelMode() const
{
	return pHeightGainModelModeAtReceiver;
}

double ITURP452v17PropagModel::GetClutterProfileUsageLimitKm() const
{
	return pClutterProfileUsageLimitKm;
}

bool ITURP452v17PropagModel::pIsAutomatic(double param)
{
	return std::isnan(param);
}