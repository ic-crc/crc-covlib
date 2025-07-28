/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "MLPLModel.h"
#include <vector>


MLPLModel::MLPLModel(void) :
	frugallyDeepModel_(MLPLModel::MLPL_JSON)
{

}

MLPLModel::MLPLModel(const MLPLModel& original) :
	// create new FrugallyDeepModel upon copy
	frugallyDeepModel_(MLPLModel::MLPL_JSON)
{
	*this = original;
}

MLPLModel::~MLPLModel(void)
{
}

const MLPLModel& MLPLModel::operator=(const MLPLModel& original)
{
	if( &original != this )
	{
		// nothing to copy
	}

	return *this;
}

void MLPLModel::Release()
{
	delete this;
}

float MLPLModel::ExcessLoss(float frequency_MHz, float distance_m, float obstructionDepth_m)
{
	float means[3] = {1893.075, 7660.8379832, 1466.43825833};
	float stdDevs[3] = {1149.98434381, 5878.52489494, 3168.08398862};

	std::vector<float> scaled_values(3);
	scaled_values[0] = (frequency_MHz - means[0]) / stdDevs[0];
	scaled_values[1] = (distance_m - means[1]) / stdDevs[1];
	scaled_values[2] = (obstructionDepth_m - means[2]) / stdDevs[2];

	return frugallyDeepModel_.predict(scaled_values);
}
