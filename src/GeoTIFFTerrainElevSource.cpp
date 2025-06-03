/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoTIFFTerrainElevSource.h"



GeoTIFFTerrainElevSource::GeoTIFFTerrainElevSource()
{

}

GeoTIFFTerrainElevSource::~GeoTIFFTerrainElevSource()
{
    
}

bool GeoTIFFTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestFltValue(lat, lon, terrainElevation);
	else
		return GetInterplValue(lat, lon, terrainElevation);
}

void GeoTIFFTerrainElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}