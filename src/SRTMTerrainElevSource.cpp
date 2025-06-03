/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "SRTMTerrainElevSource.h"


SRTMTerrainElevSource::SRTMTerrainElevSource()
{

}

SRTMTerrainElevSource::~SRTMTerrainElevSource()
{

}

bool SRTMTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT )
		return GetValue(lat, lon, false, terrainElevation);
	else
		return GetValue(lat, lon, true, terrainElevation);
}

void SRTMTerrainElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
