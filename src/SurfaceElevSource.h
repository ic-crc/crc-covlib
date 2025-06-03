/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "TopographicSource.h"

class SurfaceElevSource : public TopographicSource
{
public:
	SurfaceElevSource();
	virtual ~SurfaceElevSource();

	virtual bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation) = 0;

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