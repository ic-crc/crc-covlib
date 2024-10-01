#pragma once
#include "GeoDataGridCollection.h"
#include "TerrainElevSource.h"

class CustomTerrainElevSource : public GeoDataGridCollection<float>, public TerrainElevSource
{
public:
	CustomTerrainElevSource();
	virtual ~CustomTerrainElevSource();

	virtual bool GetTerrainElevation(double lat, double lon, float* terrainElevation);
	
	virtual void ReleaseResources(bool clearCaches);
};