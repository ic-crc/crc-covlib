/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "crc-ml.h"
#include "FrugallyDeepModel.h"


class MLPLModel : public IMLPLModel
{
public:
	MLPLModel(void);
	MLPLModel(const MLPLModel& original);
	virtual ~MLPLModel(void);

	const MLPLModel& operator=(const MLPLModel& original);

	virtual float ExcessLoss(float frequency_MHz, float distance_m, float obstructionDepth_m);
	virtual void Release();

private:
	FrugallyDeepModel frugallyDeepModel_;

	static const char* MLPL_JSON;
};
