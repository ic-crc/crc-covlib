/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "PropagModel.h"


class FreeSpacePropagModel : public PropagModel
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
