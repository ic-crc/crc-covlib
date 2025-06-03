/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "SRTMHGTReader.h"
#include "SurfaceElevSource.h"


class SRTMSurfaceElevSource : public SRTMHGTReader, public SurfaceElevSource
{
public:
	SRTMSurfaceElevSource();
	virtual ~SRTMSurfaceElevSource();

	virtual bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation);
	virtual void ReleaseResources(bool clearCaches);

};
