/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "GeoTIFFReader.h"
#include "LandCoverSource.h"

class GeoTIFFLandCoverSource : public GeoTIFFReader, public LandCoverSource
{
public:
	GeoTIFFLandCoverSource();
	virtual ~GeoTIFFLandCoverSource();

	virtual bool GetLandCoverClass(double lat, double lon, int* landCoverClass);

	virtual void ReleaseResources(bool clearCaches);
};