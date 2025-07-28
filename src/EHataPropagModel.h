/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "CRC-COVLIB.h"
#include "PropagModel.h"

class EHataPropagModel : public PropagModel
{
public:
	EHataPropagModel();
	~EHataPropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_mHz, double txHeight_meters, double rxHeight_meters, unsigned int sizeProfile, double deltaDist_km, double* elevProfile);

	void SetClutterEnvironment(Crc::Covlib::EHataClutterEnvironment clutterEnvironment);
	Crc::Covlib::EHataClutterEnvironment GetClutterEnvironment() const;
	void SetReliabilityPercentage(double percent);
	double GetReliabilityPercentage() const;

private:
	Crc::Covlib::EHataClutterEnvironment pClutterEnvironment;
	double pReliabilityPercent;
};