#pragma once
#include "GeoTIFFReader.h"
#include "SurfaceElevSource.h"

class GeoTIFFSurfaceElevSource : public GeoTIFFReader, public SurfaceElevSource
{
public:
	GeoTIFFSurfaceElevSource();
	virtual ~GeoTIFFSurfaceElevSource();

	virtual bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation);

	virtual void ReleaseResources(bool clearCaches);
};