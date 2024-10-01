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