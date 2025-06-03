/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

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