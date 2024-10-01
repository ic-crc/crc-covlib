#include "GeoTIFFSurfaceElevSource.h"



GeoTIFFSurfaceElevSource::GeoTIFFSurfaceElevSource()
{

}

GeoTIFFSurfaceElevSource::~GeoTIFFSurfaceElevSource()
{
    
}

bool GeoTIFFSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestFltValue(lat, lon, surfaceElevation);
	else
		return GetInterplValue(lat, lon, surfaceElevation);
}

void GeoTIFFSurfaceElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
