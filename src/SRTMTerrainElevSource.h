/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "SRTMHGTReader.h"
#include "TerrainElevSource.h"


class SRTMTerrainElevSource : public SRTMHGTReader, public TerrainElevSource
{
public:
	SRTMTerrainElevSource();
	virtual ~SRTMTerrainElevSource();

	virtual bool GetTerrainElevation(double lat, double lon, float* terrainElevation);
	virtual void ReleaseResources(bool clearCaches);

};
