/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "MLPLModel.h"


MLPLModel::MLPLModel(void) :
	fdeepModel_(fdeep::read_model_from_string(MLPLModel::MLPL_JSON, true, fdeep::dev_null_logger))
{

}

MLPLModel::MLPLModel(const MLPLModel& original) :
	// create new fdeep model upon copy
	fdeepModel_(fdeep::read_model_from_string(MLPLModel::MLPL_JSON, true, fdeep::dev_null_logger))
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
	float mean[3] = {1893.075, 7660.8379832, 1466.43825833};
	float stdDev[3] = {1149.98434381, 5878.52489494, 3168.08398862};

	float scaledFreq = (frequency_MHz - mean[0]) / stdDev[0];
	float scaledDist = (distance_m - mean[1]) / stdDev[1];
	float scaledObs = (obstructionDepth_m - mean[2]) / stdDev[2];

	const auto result = fdeepModel_.predict(
		{fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(3)),
		std::vector<float>{scaledFreq, scaledDist, scaledObs})});
    
	return result[0].get(0,0,0,0,0);
}
