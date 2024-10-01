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