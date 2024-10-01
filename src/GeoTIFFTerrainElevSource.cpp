#include "GeoTIFFTerrainElevSource.h"



GeoTIFFTerrainElevSource::GeoTIFFTerrainElevSource()
{

}

GeoTIFFTerrainElevSource::~GeoTIFFTerrainElevSource()
{
    
}

bool GeoTIFFTerrainElevSource::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestFltValue(lat, lon, terrainElevation);
	else
		return GetInterplValue(lat, lon, terrainElevation);
}

void GeoTIFFTerrainElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}