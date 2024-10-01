#include "GeoTIFFLandCoverSource.h"



GeoTIFFLandCoverSource::GeoTIFFLandCoverSource()
{

}

GeoTIFFLandCoverSource::~GeoTIFFLandCoverSource()
{
    
}

bool GeoTIFFLandCoverSource::GetLandCoverClass(double lat, double lon, int* landCoverClass)
{
	return GetClosestIntValue(lat, lon, landCoverClass);
}

void GeoTIFFLandCoverSource::ReleaseResources(bool clearCaches)
{
	CloseAllFiles(clearCaches);
}