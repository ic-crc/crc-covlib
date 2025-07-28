/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "PropagModel.h"
#include "crc-ml/include/crc-ml.h"
#include <array>


class CRCPathObscuraPropagModel : public PropagModel
{
public:
	CRCPathObscuraPropagModel();
	CRCPathObscuraPropagModel(const CRCPathObscuraPropagModel& original);
	virtual ~CRCPathObscuraPropagModel();
	const CRCPathObscuraPropagModel& operator=(const CRCPathObscuraPropagModel& original);

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
		unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile);

private:
	double pMedianEffectiveEarthRadius(double txLat, double txLon, double rxLat, double rxLon, double txRxPathLengthKm);
	double pElevationAngleRad(double hamsl_from, double hamsl_to, double distKm, double aeKm);
	double pFirstFresnelZoneRadius(double freq_MHz, double distToTxKm, double distToRxKm);
	double pFirstFresnelZoneMaxHeight(double freq_MHz, double distToTxKm, double distToRxKm, double txHeight_mamsl, double txToRxElevAngleRad);
	std::array<double, 7> pComputeFeatures(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m,
		double rxRcagl_m, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile);

	IPathObscuraModel* const pPathObscuraModel;
};
