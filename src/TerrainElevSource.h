/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "TopographicSource.h"

class TerrainElevSource : public TopographicSource
{
public:
	TerrainElevSource();
	virtual ~TerrainElevSource();

	virtual bool GetTerrainElevation(double lat, double lon, float* terrainElevation) = 0;

	enum Interpolation
	{
		CLOSEST_POINT = 1,
		BILINEAR = 2
	};
	void SetInterpolationType(Interpolation i);
	Interpolation GetInterpolationType() const;

protected:
	Interpolation pInterpolationType;
};