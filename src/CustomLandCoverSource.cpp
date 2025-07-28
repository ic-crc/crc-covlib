/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "CustomLandCoverSource.h"



CustomLandCoverSource::CustomLandCoverSource()
{

}

CustomLandCoverSource::~CustomLandCoverSource()
{

}

bool CustomLandCoverSource::GetLandCoverClass(double lat, double lon, int* landCoverClass)
{
short data = 0;
bool success;

	success = GetClosestData(lat, lon, &data);
	*landCoverClass = data;
	return success;
}

void CustomLandCoverSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}