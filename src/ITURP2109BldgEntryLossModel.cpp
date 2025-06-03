/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "ITURP2109BldgEntryLossModel.h"
#include "ITURP_2109.h"

using namespace Crc::Covlib;


ITURP2109BldgEntryLossModel::ITURP2109BldgEntryLossModel()
{
	pIsActive = false;
	pProbability = 50;
	pDefaultBuildingType = P2109_TRADITIONAL;
}

ITURP2109BldgEntryLossModel::~ITURP2109BldgEntryLossModel()
{

}

void ITURP2109BldgEntryLossModel::SetActiveState(bool active)
{
	pIsActive = active;
}

bool ITURP2109BldgEntryLossModel::IsActive() const
{
	return pIsActive;
}

void ITURP2109BldgEntryLossModel::SetProbability(double percent)
{
	if( percent < 0.000001 || percent > 99.999999 )
		return;
	pProbability = percent;
}

double ITURP2109BldgEntryLossModel::GetProbability() const
{
	return pProbability;
}

void ITURP2109BldgEntryLossModel::SetDefaultBuildingType(P2109BuildingType buildingType)
{
static_assert((int)ITURP_2109::BuildingType::TRADITIONAL == (int)Crc::Covlib::P2109BuildingType::P2109_TRADITIONAL, "");
static_assert((int)ITURP_2109::BuildingType::THERMALLY_EFFICIENT == (int)Crc::Covlib::P2109BuildingType::P2109_THERMALLY_EFFICIENT, "");

	if( buildingType != P2109_TRADITIONAL && buildingType != P2109_THERMALLY_EFFICIENT )
		return;
	pDefaultBuildingType = buildingType;
}
	
P2109BuildingType ITURP2109BldgEntryLossModel::GetDefaultBuildingType() const
{
	return pDefaultBuildingType;
}

double ITURP2109BldgEntryLossModel::CalcBuildingEntryLoss(double frequency_GHz, double elevAngle_degrees) const
{
	if( frequency_GHz >= 0.08 && frequency_GHz <= 100.0 && elevAngle_degrees >= -90.0 && elevAngle_degrees <= 90.0)
		return ITURP_2109::BuildingEntryLoss(frequency_GHz, pProbability, (ITURP_2109::BuildingType)pDefaultBuildingType, elevAngle_degrees);
	else
		return 0;
}
