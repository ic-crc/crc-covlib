#pragma once
#include "GeoDataGridCollection.h"
#include "LandCoverSource.h"

class CustomLandCoverSource : public GeoDataGridCollection<short>, public LandCoverSource
{
public:
	CustomLandCoverSource();
	virtual ~CustomLandCoverSource();

	virtual bool GetLandCoverClass(double lat, double lon, int* landCoverClass);

	virtual void ReleaseResources(bool clearCaches);
};
