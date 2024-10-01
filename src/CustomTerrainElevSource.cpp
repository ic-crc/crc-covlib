#include "CustomTerrainElevSource.h"



CustomTerrainElevSource::CustomTerrainElevSource()
{

}

CustomTerrainElevSource::~CustomTerrainElevSource()
{
    
}

bool CustomTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestData(lat, lon, terrainElevation);
	else
		return GetInterplData(lat, lon, terrainElevation);
}

void CustomTerrainElevSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}