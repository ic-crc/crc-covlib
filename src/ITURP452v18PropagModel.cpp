/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP452v18PropagModel.h"
#include <cmath>

using namespace Crc::Covlib;


ITURP452v18PropagModel::ITURP452v18PropagModel()
{
	pTimePercent = 50;
	pPredictionType = P452_AVERAGE_YEAR;
	pDeltaN = AUTOMATIC;
	pN0 = AUTOMATIC;
	pTemperature_C = AUTOMATIC;
	pPressure_hPa = AUTOMATIC;
	pMappingType = P452_MAP_TO_CLUTTER_CATEGORY;
	pSurfaceProfileMethod = P452_ADD_REPR_CLUTTER_HEIGHT;
}

ITURP452v18PropagModel::~ITURP452v18PropagModel()
{

}

PropagationModel ITURP452v18PropagModel::Id()
{
	return ITU_R_P_452_V18;
}
bool ITURP452v18PropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool ITURP452v18PropagModel::IsUsingMappedLandCoverData()
{
	return (pSurfaceProfileMethod == P452_ADD_REPR_CLUTTER_HEIGHT);
}

bool ITURP452v18PropagModel::IsUsingItuRadioClimZoneData()
{
	return true;
}

bool ITURP452v18PropagModel::IsUsingSurfaceElevData()
{
	return (pSurfaceProfileMethod == P452_EXPERIMENTAL_USE_OF_SURFACE_ELEV_DATA);
}

int ITURP452v18PropagModel::DefaultMappedLandCoverValue()
{
	if( pMappingType == P452_MAP_TO_CLUTTER_CATEGORY )
		return P452_OPEN_RURAL;
	else // P452_MAP_TO_REPR_CLUTTER_HEIGHT
		return 0; // 0 meters
}

double ITURP452v18PropagModel::CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m, 
	                                        double txAntGain_dBi, double rxAntGain_dBi, Polarization pol, unsigned int sizeProfiles,
	                                        double* distKmProfile, double* elevProfile, int* mappedLandCoverProfile, double* surfaceHeightProfile,
	                                        ITURadioClimaticZone* radioClimaticZoneProfile)
{
std::vector<double> reprClutterHeightVector;
double* reprClutterHeightProfile = nullptr;

	if( sizeProfiles < 3 )
		return 0;
	if( distKmProfile[sizeProfiles-1] < 1E-5 )
		return 0;

	if( IsUsingMappedLandCoverData() )
	{
		GetReprClutterHeightProfile(sizeProfiles, mappedLandCoverProfile, &reprClutterHeightVector);
		reprClutterHeightProfile = reprClutterHeightVector.data();
	}

	if( IsUsingSurfaceElevData() )
	{
		reprClutterHeightVector.resize(sizeProfiles);
		for(unsigned int i=0 ; i<sizeProfiles ; i++)
			reprClutterHeightVector[i] = surfaceHeightProfile[i]-elevProfile[i];
		reprClutterHeightProfile = reprClutterHeightVector.data();
	}

	static_assert(std::isnan(Crc::Covlib::AUTOMATIC) && std::isnan(ITURP_452_1812_common::AUTO), "");

	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_COASTAL_LAND == (int)ITURP_452_1812_common::RadioClimaticZone::COASTAL_LAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_INLAND == (int)ITURP_452_1812_common::RadioClimaticZone::INLAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_SEA == (int)ITURP_452_1812_common::RadioClimaticZone::SEA, "");

	return ClearAirBasicTransmissionLoss(freq_Ghz, pTimePercent, pPredictionType==P452_WORST_MONTH, txLat, txLon, rxLat, rxLon,
	                                     txRcagl_m, rxRcagl_m, txAntGain_dBi, rxAntGain_dBi, pol==VERTICAL_POL, sizeProfiles, distKmProfile, elevProfile,
	                                     nullptr, reprClutterHeightProfile, (RadioClimaticZone*)radioClimaticZoneProfile,
	                                     /*dct*/AUTO, /*dcr*/AUTO, pPressure_hPa, pTemperature_C, pDeltaN, pN0);
}

void ITURP452v18PropagModel::GetReprClutterHeightProfile(unsigned int sizeProfile, int* mappedLandCoverProfile, std::vector<double>* reprClutterHeightProfile)
{
	reprClutterHeightProfile->resize(sizeProfile);
	if( pMappingType == P452_MAP_TO_CLUTTER_CATEGORY )
	{
		for(unsigned int i=0 ; i<sizeProfile ; i++)
			(*reprClutterHeightProfile)[i] = GetClutterCategoryReprHeight((P452ClutterCategory)mappedLandCoverProfile[i]);
	}
	else // P452_MAP_TO_REPR_CLUTTER_HEIGHT
	{
		for(unsigned int i=0 ; i<sizeProfile ; i++)
			(*reprClutterHeightProfile)[i] = (double)mappedLandCoverProfile[i];
	}
}

void ITURP452v18PropagModel::SetTimePercentage(double percent)
{
	if( percent < 0.001 || percent > 50.0 )
		return;
	pTimePercent = percent;
}
	
double ITURP452v18PropagModel::GetTimePercentage() const
{
	return pTimePercent;
}

void ITURP452v18PropagModel::SetPredictionType(P452PredictionType predictionType)
{
	if( predictionType != P452_AVERAGE_YEAR && predictionType != P452_WORST_MONTH )
		return;
	pPredictionType = predictionType;
}

P452PredictionType ITURP452v18PropagModel::GetPredictionType() const
{
	return pPredictionType;
}

void ITURP452v18PropagModel::SetAverageRadioRefractivityLapseRate(double deltaN)
{
	if( pIsAutomatic(deltaN) || deltaN > 0 )
		pDeltaN = deltaN;
}
	
double ITURP452v18PropagModel::GetAverageRadioRefractivityLapseRate() const
{
	return pDeltaN;
}

void ITURP452v18PropagModel::SetSeaLevelSurfaceRefractivity(double N0)
{
	if( pIsAutomatic(N0) || N0 > 0 )
		pN0 = N0;
}
	
double ITURP452v18PropagModel::GetSeaLevelSurfaceRefractivity() const
{
	return pN0;
}

void ITURP452v18PropagModel::SetAirTemperature(double temperature_C)
{
	if( pIsAutomatic(temperature_C) || temperature_C >= -273.15 )
		pTemperature_C = temperature_C;
}
	
double ITURP452v18PropagModel::GetAirTemperature() const
{
	return pTemperature_C;
}

void ITURP452v18PropagModel::SetAirPressure(double pressure_hPa)
{
	if( pIsAutomatic(pressure_hPa) || pressure_hPa > 0 )
		pPressure_hPa = pressure_hPa;
}
	
double ITURP452v18PropagModel::GetAirPressure() const
{
	return pPressure_hPa;
}

void ITURP452v18PropagModel::SetClutterCategoryReprHeight(Crc::Covlib::P452ClutterCategory clutterCategory, double representativeHeight_m)
{
static_assert((int)ITURP_452_1812_common::ClutterCategory::WATER_SEA == (int)Crc::Covlib::P452ClutterCategory::P452_WATER_SEA, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::OPEN_RURAL == (int)Crc::Covlib::P452ClutterCategory::P452_OPEN_RURAL, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::SUBURBAN == (int)Crc::Covlib::P452ClutterCategory::P452_SUBURBAN, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::URBAN_TREES_FOREST == (int)Crc::Covlib::P452ClutterCategory::P452_URBAN_TREES_FOREST, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::DENSE_URBAN == (int)Crc::Covlib::P452ClutterCategory::P452_DENSE_URBAN, "");

	ITURP_452_v18::SetDefaultRepresentativeHeight((ITURP_452_1812_common::ClutterCategory) clutterCategory, representativeHeight_m);
}

double ITURP452v18PropagModel::GetClutterCategoryReprHeight(P452ClutterCategory clutterCategory) const
{
	return ITURP_452_v18::GetDefaultRepresentativeHeight((ITURP_452_1812_common::ClutterCategory) clutterCategory);
}

void ITURP452v18PropagModel::SetLandCoverMappingType(P452LandCoverMappingType mappingType)
{
	if( mappingType == P452_MAP_TO_CLUTTER_CATEGORY || mappingType == P452_MAP_TO_REPR_CLUTTER_HEIGHT )
		pMappingType = mappingType;
}
	
P452LandCoverMappingType ITURP452v18PropagModel::GetLandCoverMappingType() const
{
	return pMappingType;
}

void ITURP452v18PropagModel::SetSurfaceProfileMethod(P452SurfaceProfileMethod method)
{
	if( method == P452_ADD_REPR_CLUTTER_HEIGHT || method == P452_EXPERIMENTAL_USE_OF_SURFACE_ELEV_DATA )
		pSurfaceProfileMethod = method;
}

P452SurfaceProfileMethod ITURP452v18PropagModel::GetSurfaceProfileMethod() const
{
	return pSurfaceProfileMethod;
}

bool ITURP452v18PropagModel::pIsAutomatic(double param)
{
	return std::isnan(param);
}