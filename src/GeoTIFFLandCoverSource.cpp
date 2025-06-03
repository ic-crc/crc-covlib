/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoTIFFLandCoverSource.h"



GeoTIFFLandCoverSource::GeoTIFFLandCoverSource()
{

}

GeoTIFFLandCoverSource::~GeoTIFFLandCoverSource()
{
    
}

bool GeoTIFFLandCoverSource::GetLandCoverClass(double lat, double lon, int* landCoverClass)
{
	return GetClosestIntValue(lat, lon, landCoverClass);
}

void GeoTIFFLandCoverSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}