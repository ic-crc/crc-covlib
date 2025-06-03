/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "LongleyRicePropagModel.h"
#include "ntia_itm/include/itm.h"
#include <vector>
#include <cstring>

using namespace Crc::Covlib;


const double LongleyRicePropagModel::MIN_PERCENT =  0.1;
const double LongleyRicePropagModel::MAX_PERCENT = 99.9;

LongleyRicePropagModel::LongleyRicePropagModel()
{
	pSurfaceRefractivity = 301;
	pGroundDielectricConst = 15;
	pGroundConductivity = 0.005;
	pClimaticZone = LR_CONTINENTAL_TEMPERATE;
	pActivePercentageSet = LR_TIME_LOCATION_SITUATION;
	pTimePercent = 50;
	pLocationPercent = 50;
	pSituationPercent = 50;
	pConfidencePercent = 50;
	pReliabilityPercent = 50;
	pModeOfVariability = 12;
}
	
LongleyRicePropagModel::~LongleyRicePropagModel()
{

}

PropagationModel LongleyRicePropagModel::Id()
{
	return LONGLEY_RICE;
}

bool LongleyRicePropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool LongleyRicePropagModel::IsUsingMappedLandCoverData()
{
	return false;
}

bool LongleyRicePropagModel::IsUsingItuRadioClimZoneData()
{
	return false;
}

bool LongleyRicePropagModel::IsUsingSurfaceElevData()
{
	return false;
}

int LongleyRicePropagModel::DefaultMappedLandCoverValue()
{
	return -1;
}

double LongleyRicePropagModel::CalcPathLoss(double freq_mHz, double txHeight_meters, double rxHeight_meters, Crc::Covlib::Polarization pol,
                                            unsigned int sizeProfile, double deltaDist_km, double* elevProfile)
{
double dbloss;
std::vector<double> lrElevProfile;
long warnings;

	if( sizeProfile < 2 )
		return 0;
	if( deltaDist_km < 0.00001 )
		return 0;

	lrElevProfile.resize(sizeProfile+2);
	lrElevProfile[0] = sizeProfile-1;
	lrElevProfile[1] = deltaDist_km*1000.0;
	std::memcpy(&(lrElevProfile[2]), elevProfile, sizeProfile*sizeof(double));

	if( pActivePercentageSet == LR_TIME_LOCATION_SITUATION )
	{
		ITM_P2P_TLS(txHeight_meters, rxHeight_meters, lrElevProfile.data(), pClimaticZone, pSurfaceRefractivity, freq_mHz,
		            pol, pGroundDielectricConst, pGroundConductivity, pModeOfVariability, pTimePercent, pLocationPercent, pSituationPercent,
		            &dbloss, &warnings);
	}
	else
	{
		ITM_P2P_CR(txHeight_meters, rxHeight_meters, lrElevProfile.data(), pClimaticZone, pSurfaceRefractivity, freq_mHz,
		           pol, pGroundDielectricConst, pGroundConductivity, pModeOfVariability, pConfidencePercent, pReliabilityPercent,
		           &dbloss, &warnings);
	}

	return dbloss;
}

void LongleyRicePropagModel::SetSurfaceRefractivity(double refractivity_NUnits)
{
	if( refractivity_NUnits < 250 || refractivity_NUnits > 400 )
		return;
	pSurfaceRefractivity = refractivity_NUnits;
}

double LongleyRicePropagModel::GetSurfaceRefractivity() const
{
	return pSurfaceRefractivity;
}

void LongleyRicePropagModel::SetGroundDielectricConst(double dielectricConst)
{
	if( dielectricConst < 4 || dielectricConst > 81 )
		return;
	pGroundDielectricConst = dielectricConst;
}

double LongleyRicePropagModel::GetGroundDielectricConst() const
{
	return pGroundDielectricConst;
}

void LongleyRicePropagModel::SetGroundConductivity(double groundConduct_Sm)
{
	if( groundConduct_Sm < 0.001 || groundConduct_Sm > 5 )
		return;
	pGroundConductivity = groundConduct_Sm;
}

double LongleyRicePropagModel::GetGroundConductivity() const
{
	return pGroundConductivity;
}

void LongleyRicePropagModel::SetClimaticZone(LRClimaticZone climaticZone)
{
	if( climaticZone < LR_EQUATORIAL || climaticZone > LR_MARITIME_TEMPERATE_OVER_SEA )
		return;
	pClimaticZone = climaticZone;
}

LRClimaticZone LongleyRicePropagModel::GetClimaticZone() const
{
	return pClimaticZone;
}

void LongleyRicePropagModel::SetActivePercentageSet(LRPercentageSet percentageSet)
{
	if( percentageSet == LR_TIME_LOCATION_SITUATION || percentageSet == LR_CONFIDENCE_RELIABILITY )
		pActivePercentageSet = percentageSet;
}

LRPercentageSet LongleyRicePropagModel::GetActivePercentageSet() const
{
	return pActivePercentageSet;
}

void LongleyRicePropagModel::SetTimePercentage(double percent)
{
	if( percent < MIN_PERCENT || percent > MAX_PERCENT )
		return;
	pTimePercent = percent;
}
	
double LongleyRicePropagModel::GetTimePercentage() const
{
	return pTimePercent;
}

void LongleyRicePropagModel::SetLocationPercentage(double percent)
{
	if( percent < MIN_PERCENT || percent > MAX_PERCENT )
		return;
	pLocationPercent = percent;
}
	
double LongleyRicePropagModel::GetLocationPercentage() const
{
	return pLocationPercent;
}

void LongleyRicePropagModel::SetSituationPercentage(double percent)
{
	if( percent < MIN_PERCENT || percent > MAX_PERCENT )
		return;
	pSituationPercent = percent;
}

double LongleyRicePropagModel::GetSituationPercentage() const
{
	return pSituationPercent;
}

void LongleyRicePropagModel::SetConfidencePercentage(double percent)
{
	if( percent < MIN_PERCENT || percent > MAX_PERCENT )
		return;
	pConfidencePercent = percent;
}
	
double LongleyRicePropagModel::GetConfidencePercentage() const
{
	return pConfidencePercent;
}

void LongleyRicePropagModel::SetReliabilityPercentage(double percent)
{
	if( percent < MIN_PERCENT || percent > MAX_PERCENT )
		return;
	pReliabilityPercent = percent;
}
	
double LongleyRicePropagModel::GetReliabilityPercentage() const
{
	return pReliabilityPercent;
}

void LongleyRicePropagModel::SetModeOfVariability(int mode)
{
	if( (mode >= 0 && mode <= 3) || (mode >= 10 && mode <= 13) || (mode >= 20 && mode <= 23) || (mode >= 30 && mode <= 33) )
		pModeOfVariability = mode;
}
	
int LongleyRicePropagModel::GetModeOfVariability() const
{
	return pModeOfVariability;
}
