/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include <string>
#include <vector>
#include <fdeep/fdeep.hpp> // https://github.com/Dobiasd/frugally-deep


class FrugallyDeepModel
{
public:
	FrugallyDeepModel(const std::string &content);
	FrugallyDeepModel(const FrugallyDeepModel& original);
	virtual ~FrugallyDeepModel(void);
	const FrugallyDeepModel& operator=(const FrugallyDeepModel& original);

	float predict(const std::vector<float>& scaledParams);

private:
	fdeep::model fdeepModel_;
	std::string jsonContent_;
};