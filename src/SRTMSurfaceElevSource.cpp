/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "SRTMSurfaceElevSource.h"


SRTMSurfaceElevSource::SRTMSurfaceElevSource()
{

}

SRTMSurfaceElevSource::~SRTMSurfaceElevSource()
{

}

bool SRTMSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT )
		return GetValue(lat, lon, false, surfaceElevation);
	else
		return GetValue(lat, lon, true, surfaceElevation);
}

void SRTMSurfaceElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
