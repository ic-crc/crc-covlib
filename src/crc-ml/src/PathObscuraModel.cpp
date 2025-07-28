/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "PathObscuraModel.h"
#include <vector>


PathObscuraModel::PathObscuraModel(void) :
	frugallyDeepModel_(PathObscuraModel::PATH_OBSCURA_V1_0_JSON)
{

}

PathObscuraModel::PathObscuraModel(const PathObscuraModel& original) :
	// create new FrugallyDeepModel upon copy
	frugallyDeepModel_(PathObscuraModel::PATH_OBSCURA_V1_0_JSON)
{
	*this = original;
}

PathObscuraModel::~PathObscuraModel(void)
{
}

const PathObscuraModel& PathObscuraModel::operator=(const PathObscuraModel& original)
{
	if( &original != this )
	{
		// nothing to copy
	}

	return *this;
}

void PathObscuraModel::Release()
{
	delete this;
}

const char* PathObscuraModel::Version()
{
	return "1.0";
}

/*
freq_MHz: Frequency in MHz
dist_m: 3D distance between antennas, in meters
obsDepth_m: Total obstruction depth, in meters
intersectDept_m: Intersection depth (last_blockage_location – first_blockage_location), in meters
obsBlockCount: Number of obstruction blocks
terrainRatio: Percentage of blockage that is terrain [% value between 0 and 1]
aboveFzCount: Number of local DSM (Digital Surface Model) maxima above the 1st Fresnel Zone
*/
float PathObscuraModel::PathLoss(float freq_MHz, float dist_m, float obsDepth_m,  float intersectDept_m,
	                             int obsBlockCount, float terrainRatio, int aboveFzCount)
{
	float means[7] = {2.55249107e+03, 7.40813118e+03, 1.35072893e+03, 4.77936075e+03,
		              4.12463446e+01, 3.86704441e-01, 8.07793839e+01};
	float stdDevs[7] = {1.80929707e+03, 5.69803071e+03, 2.94513523e+03, 4.95281480e+03,
	                    4.36165200e+01, 3.97736854e-01, 1.71244719e+02};

	float values[7] = {freq_MHz, dist_m, obsDepth_m, intersectDept_m, static_cast<float>(obsBlockCount),
		               terrainRatio, static_cast<float>(aboveFzCount)};
	std::vector<float> scaled_values(7);
	for(int i=0 ; i<7 ; i++)
		scaled_values[i] = (values[i] - means[i]) / stdDevs[i];

	return frugallyDeepModel_.predict(scaled_values);
}
