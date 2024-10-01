#pragma once
#include "GeoTIFFReader.h"
#include "TerrainElevSource.h"

class GeoTIFFTerrainElevSource : public GeoTIFFReader, public TerrainElevSource
{
public:
	GeoTIFFTerrainElevSource();
	virtual ~GeoTIFFTerrainElevSource();

	virtual bool GetTerrainElevation(double lat, double lon, float* terrainElevation);

	virtual void ReleaseResources(bool clearCaches);
};