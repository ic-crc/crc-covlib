/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "LandCoverSource.h"



LandCoverSource::LandCoverSource()
{

}

LandCoverSource::~LandCoverSource()
{
    
}

bool LandCoverSource::GetPropagModelLandCoverClass(double lat, double lon, Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass)
{
bool success;
int landCoverClass;

	success = GetLandCoverClass(lat, lon, &landCoverClass);
	if( success )
		success = GetMapping(landCoverClass, propagModel, propagModelLandCoverClass);
	return success;
}

void LandCoverSource::SetMapping(int landCoverClass, Crc::Covlib::PropagationModel propagModel, int propagModelLandCoverClass)
{
	pMap[std::pair<int, int>(propagModel, landCoverClass)] = propagModelLandCoverClass;
}

bool LandCoverSource::GetMapping(int landCoverClass, Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass) const
{
std::pair<int, int> key(propagModel, landCoverClass);

	if( pMap.count(key) == 1 )
	{
		*propagModelLandCoverClass = pMap.at(key);
		return true;
	}
	else if( pDefaultMap.count(propagModel) == 1 )
	{
		*propagModelLandCoverClass = pDefaultMap.at(propagModel);
		return true;
	}
	return false;
}

void LandCoverSource::SetDefaultMapping(Crc::Covlib::PropagationModel propagModel, int propagModelLandCoverClass)
{
	pDefaultMap[propagModel] = propagModelLandCoverClass;
}

bool LandCoverSource::GetDefaultMapping(Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass) const
{
	if( pDefaultMap.count(propagModel) == 1 )
	{
		*propagModelLandCoverClass = pDefaultMap.at(propagModel);
		return true;
	}
	return false;
}

void LandCoverSource::ClearMappings(Crc::Covlib::PropagationModel propagModel)
{
std::map<std::pair<int, int>, int>::iterator iter;

	iter = pMap.begin();
	while( iter != pMap.end() )
	{
		if( iter->first.first == propagModel )
			iter = pMap.erase(iter);
		else
			++iter;
	}

	pDefaultMap.erase(propagModel);
}