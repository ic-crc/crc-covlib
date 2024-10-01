#include "CustomSurfaceElevSource.h"



CustomSurfaceElevSource::CustomSurfaceElevSource()
{

}

CustomSurfaceElevSource::~CustomSurfaceElevSource()
{
    
}

bool CustomSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT)
		return GetClosestData(lat, lon, surfaceElevation);
	else
		return GetInterplData(lat, lon, surfaceElevation);
}

void CustomSurfaceElevSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}