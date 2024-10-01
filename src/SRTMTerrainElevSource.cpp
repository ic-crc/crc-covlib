#include "SRTMTerrainElevSource.h"


SRTMTerrainElevSource::SRTMTerrainElevSource()
{

}

SRTMTerrainElevSource::~SRTMTerrainElevSource()
{

}

bool SRTMTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT )
		return GetValue(lat, lon, false, terrainElevation);
	else
		return GetValue(lat, lon, true, terrainElevation);
}

void SRTMTerrainElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
