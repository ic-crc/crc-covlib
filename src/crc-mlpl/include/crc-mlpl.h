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
