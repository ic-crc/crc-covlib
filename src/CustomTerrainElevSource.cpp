/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "CustomTerrainElevSource.h"



CustomTerrainElevSource::CustomTerrainElevSource()
{

}

CustomTerrainElevSource::~CustomTerrainElevSource()
{
    
}

bool CustomTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestData(lat, lon, terrainElevation);
	else
		return GetInterplData(lat, lon, terrainElevation);
}

void CustomTerrainElevSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}