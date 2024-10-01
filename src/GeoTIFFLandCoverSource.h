#pragma once
#include "GeoTIFFReader.h"
#include "LandCoverSource.h"

class GeoTIFFLandCoverSource : public GeoTIFFReader, public LandCoverSource
{
public:
	GeoTIFFLandCoverSource();
	virtual ~GeoTIFFLandCoverSource();

	virtual bool GetLandCoverClass(double lat, double lon, int* landCoverClass);

	virtual void ReleaseResources(bool clearCaches);
};