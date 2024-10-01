#include "SRTMSurfaceElevSource.h"


SRTMSurfaceElevSource::SRTMSurfaceElevSource()
{

}

SRTMSurfaceElevSource::~SRTMSurfaceElevSource()
{

}

bool SRTMSurfaceElevSource::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pInterpolationType == CLOSEST_POINT )
		return GetValue(lat, lon, false, surfaceElevation);
	else
		return GetValue(lat, lon, true, surfaceElevation);
}

void SRTMSurfaceElevSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}
