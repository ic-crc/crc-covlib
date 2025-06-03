/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "FreeSpacePropagModel.h"
#include <cmath>
#include <algorithm>

using namespace Crc::Covlib;


FreeSpacePropagModel::FreeSpacePropagModel()
{

}

FreeSpacePropagModel::~FreeSpacePropagModel()
{

}

PropagationModel FreeSpacePropagModel::Id()
{
	return 	PropagationModel::FREE_SPACE;
}

bool FreeSpacePropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool FreeSpacePropagModel::IsUsingMappedLandCoverData()
{
	return false;
}

bool FreeSpacePropagModel::IsUsingItuRadioClimZoneData()
{
	return false;
}

bool FreeSpacePropagModel::IsUsingSurfaceElevData()
{
	return false;
}

int FreeSpacePropagModel::DefaultMappedLandCoverValue()
{
	return -1;
}

double FreeSpacePropagModel::CalcPathLoss(double freq_MHz, double dist_km)
{
double Lb;

	if( dist_km <= 0 )
		return 0;

	Lb = 32.4 + 20.0*log10(freq_MHz) + 20.0*log10(dist_km); // from ITU-R P.525-4
	Lb = std::max(Lb, 0.0);
	return Lb;
}

double FreeSpacePropagModel::CalcPathLoss(double freq_Mhz, double txRcagl_m, double rxRcagl_m,
	                                      unsigned int sizeProfiles, double* distKmProfile, double* elevProfile)
{
// Using free-space distance calculation method from eq.(42) of ITU-R P.2001-5.
double pathLength_km = distKmProfile[sizeProfiles-1];
double hdiff_km = ((elevProfile[0]+txRcagl_m) - (elevProfile[sizeProfiles-1]+rxRcagl_m)) / 1000.0;
double distFs_km = sqrt((pathLength_km*pathLength_km) + (hdiff_km*hdiff_km));

	return CalcPathLoss(freq_Mhz, distFs_km);
}
