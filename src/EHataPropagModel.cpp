/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "EHataPropagModel.h"
#include "ntia_ehata/include/ehata.h"
#include <vector>
#include <cstring>

using namespace Crc::Covlib;


EHataPropagModel::EHataPropagModel()
{
	pClutterEnvironment = EHATA_URBAN;
	pReliabilityPercent = 50;
}

EHataPropagModel::~EHataPropagModel()
{

}

PropagationModel EHataPropagModel::Id()
{
	return EXTENDED_HATA;
}

bool EHataPropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool EHataPropagModel::IsUsingMappedLandCoverData()
{
	return false;
}

bool EHataPropagModel::IsUsingItuRadioClimZoneData()
{
	return false;
}

bool EHataPropagModel::IsUsingSurfaceElevData()
{
	return false;
}

int EHataPropagModel::DefaultMappedLandCoverValue()
{
	return -1;
}

double EHataPropagModel::CalcPathLoss(double freq_mHz, double txHeight_meters, double rxHeight_meters,
                                      unsigned int sizeProfile, double deltaDist_km, double* elevProfile)
{
double dbloss;
std::vector<double> ehataElevProfile;

	if( sizeProfile < 2 )
		return 0;
	if( deltaDist_km < 0.00001 )
		return 0;

	ehataElevProfile.resize(sizeProfile+2);
	ehataElevProfile[0] = sizeProfile-1;
	ehataElevProfile[1] = deltaDist_km*1000.0;
	std::memcpy(&(ehataElevProfile[2]), elevProfile, sizeProfile*sizeof(double));

	ExtendedHata(ehataElevProfile.data(), freq_mHz, txHeight_meters, rxHeight_meters, pClutterEnvironment,
	             pReliabilityPercent/100.0, &dbloss);

	return dbloss;
}

void EHataPropagModel::SetClutterEnvironment(EHataClutterEnvironment clutterEnvironment)
{
	if( clutterEnvironment == EHATA_URBAN || clutterEnvironment == EHATA_SUBURBAN || clutterEnvironment == EHATA_RURAL )
		pClutterEnvironment = clutterEnvironment;
}

EHataClutterEnvironment EHataPropagModel::GetClutterEnvironment() const
{
	return pClutterEnvironment;
}

void EHataPropagModel::SetReliabilityPercentage(double percent)
{
	if( percent > 0 && percent < 100 )
		pReliabilityPercent = percent;
}
	
double EHataPropagModel::GetReliabilityPercentage() const
{
	return pReliabilityPercent;
}