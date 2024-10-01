#pragma once
#include "SRTMHGTReader.h"
#include "SurfaceElevSource.h"


class SRTMSurfaceElevSource : public SRTMHGTReader, public SurfaceElevSource
{
public:
	SRTMSurfaceElevSource();
	virtual ~SRTMSurfaceElevSource();

	virtual bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation);
	virtual void ReleaseResources(bool clearCaches);

};
