/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "FrugallyDeepModel.h"


FrugallyDeepModel::FrugallyDeepModel(const std::string &content) :
	fdeepModel_(fdeep::read_model_from_string(content, true, fdeep::dev_null_logger))
{
	jsonContent_ = content;
}

FrugallyDeepModel::FrugallyDeepModel(const FrugallyDeepModel& original) :
	// create new fdeep model upon copy
	fdeepModel_(fdeep::read_model_from_string(original.jsonContent_, true, fdeep::dev_null_logger))
{
	jsonContent_ = original.jsonContent_;
}

FrugallyDeepModel::~FrugallyDeepModel(void)
{
}

const FrugallyDeepModel& FrugallyDeepModel::operator=(const FrugallyDeepModel& original)
{
	if( &original != this )
	{
		jsonContent_ = original.jsonContent_;
		fdeepModel_ = fdeep::model(fdeep::read_model_from_string(jsonContent_, true, fdeep::dev_null_logger));
	}

	return *this;
}

float FrugallyDeepModel::predict(const std::vector<float>& scaledParams)
{
	const auto result = fdeepModel_.predict(
		{fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(scaledParams.size())), scaledParams)});

	return result[0].get(0,0,0,0,0);
}
