#include "CustomLandCoverSource.h"



CustomLandCoverSource::CustomLandCoverSource()
{

}

CustomLandCoverSource::~CustomLandCoverSource()
{

}

bool CustomLandCoverSource::GetLandCoverClass(double lat, double lon, int* landCoverClass)
{
short data;
bool success;

	success = GetClosestData(lat, lon, &data);
	*landCoverClass = data;
	return success;
}

void CustomLandCoverSource::ReleaseResources([[maybe_unused]] bool clearCaches)
{
	// no resources to release
}