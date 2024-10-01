#pragma once
#include "GeoDataGridCollection.h"
#include "SurfaceElevSource.h"

class CustomSurfaceElevSource : public GeoDataGridCollection<float>, public SurfaceElevSource
{
public:
	CustomSurfaceElevSource();
	virtual ~CustomSurfaceElevSource();

	virtual bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation);

	virtual void ReleaseResources(bool clearCaches);
};