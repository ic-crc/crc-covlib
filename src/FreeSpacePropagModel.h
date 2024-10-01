#pragma once
#include "PropagModel.h"


class FreeSpacePropagModel : PropagModel
{
public:
	FreeSpacePropagModel();
	virtual ~FreeSpacePropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_MHz, double dist_km);

	double CalcPathLoss(double freq_Mhz, double txRcagl_m, double rxRcagl_m, unsigned int sizeProfiles, double* distKmProfile, double* elevProfile);
};
