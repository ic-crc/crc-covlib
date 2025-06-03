/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoTIFFSurfaceElevSource.h"



GeoTIFFSurfaceElevSource::GeoTIFFSurfaceElevSource()
{

}

GeoTIFFSurfaceElevSource::~GeoTIFFSurfaceElevSource()
{
    
}

bool GeoTIFFSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestFltValue(lat, lon, surfaceElevation);
	else
		return GetInterplValue(lat, lon, surfaceElevation);
}

void GeoTIFFSurfaceElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
