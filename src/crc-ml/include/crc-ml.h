/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once


struct IMLPLModel
{
	virtual void Release() = 0;
	virtual float ExcessLoss(float frequency_MHz, float distance_m, float obstructionDepth_m) = 0;
};

IMLPLModel* NewMLPLModel();


struct IPathObscuraModel
{
	virtual void Release() = 0;
	virtual const char* Version() = 0;
	virtual float PathLoss(float freq_MHz, float dist_m, float obsDepth_m, float intersectDept_m,
	                       int obsBlockCount, float terrainRatio, int aboveFzCount) = 0;
};

IPathObscuraModel* NewPathObscuraModel();
