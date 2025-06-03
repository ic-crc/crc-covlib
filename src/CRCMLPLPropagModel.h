/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "PropagModel.h"
#include "FreeSpacePropagModel.h"
#include "crc-mlpl/include/crc-mlpl.h"


class CRCMLPLPropagModel : PropagModel
{
public:
	CRCMLPLPropagModel();
	CRCMLPLPropagModel(const CRCMLPLPropagModel& original);
	virtual ~CRCMLPLPropagModel();
	const CRCMLPLPropagModel& operator=(const CRCMLPLPropagModel& original);

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
		unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile);

private:
	double pObstructionDepth(double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
		unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile);
	double pElevationAngleRad(double hamsl_from, double hamsl_to, double distKm, double aeKm);

	FreeSpacePropagModel pFreeSpaceModel;
	IMLPLModel* const pMLPLModel;
};
