/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "CustomSurfaceElevSource.h"



CustomSurfaceElevSource::CustomSurfaceElevSource()
{

}

CustomSurfaceElevSource::~CustomSurfaceElevSource()
{
    
}

bool CustomSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestData(lat, lon, surfaceElevation);
	else
		return GetInterplData(lat, lon, surfaceElevation);
}

void CustomSurfaceElevSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}