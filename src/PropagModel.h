#pragma once
#include "CRC-COVLIB.h"

class PropagModel
{
public:
	PropagModel() {};
	virtual ~PropagModel() {};

	virtual Crc::Covlib::PropagationModel Id() = 0;
	virtual bool IsUsingTerrainElevData() = 0;
	virtual bool IsUsingMappedLandCoverData() = 0;
	virtual bool IsUsingItuRadioClimZoneData() = 0;
	virtual bool IsUsingSurfaceElevData() = 0;
	virtual int DefaultMappedLandCoverValue() = 0;
};