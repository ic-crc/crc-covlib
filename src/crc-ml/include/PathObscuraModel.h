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


class PathObscuraModel : public IPathObscuraModel
{
public:
	PathObscuraModel(void);
	PathObscuraModel(const PathObscuraModel& original);
	virtual ~PathObscuraModel(void);

	const PathObscuraModel& operator=(const PathObscuraModel& original);

	virtual const char* Version();
	virtual float PathLoss(float freq_MHz, float dist_m, float obsDepth_m, float intersectDept_m,
	                       int obsBlockCount, float terrainRatio, int aboveFzCount);
	virtual void Release();

private:
	FrugallyDeepModel frugallyDeepModel_;

	static const char* PATH_OBSCURA_V1_0_JSON;
};
