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
