#pragma once
#include "TopographicSource.h"
#include "CRC-COVLIB.h"
#include <map>

class LandCoverSource : public TopographicSource
{
public:
	LandCoverSource();
	virtual ~LandCoverSource();

	virtual bool GetLandCoverClass(double lat, double lon, int* landCoverClass) = 0;

	bool GetPropagModelLandCoverClass(double lat, double lon, Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass);

	void SetMapping(int landCoverClass, Crc::Covlib::PropagationModel propagModel, int propagModelLandCoverClass);
	bool GetMapping(int landCoverClass, Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass) const;
	void SetDefaultMapping(Crc::Covlib::PropagationModel propagModel, int propagModelLandCoverClass);
	bool GetDefaultMapping(Crc::Covlib::PropagationModel propagModel, int* propagModelLandCoverClass) const;
	void ClearMappings(Crc::Covlib::PropagationModel propagModel);

protected:
	std::map<std::pair<int,int>, int> pMap;
	std::map<int, int> pDefaultMap;
};