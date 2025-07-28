/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "CRCPathObscuraPropagModel.h"
#include "ITURP_2001.h"
#include "ITURP_DigitalMaps.h"
#include <cmath>


 using namespace Crc::Covlib;


CRCPathObscuraPropagModel::CRCPathObscuraPropagModel() :
	pPathObscuraModel(NewPathObscuraModel())
{

}

CRCPathObscuraPropagModel::CRCPathObscuraPropagModel(const CRCPathObscuraPropagModel& original) :
	pPathObscuraModel(NewPathObscuraModel())
{
	*this = original;
}

CRCPathObscuraPropagModel::~CRCPathObscuraPropagModel()
{
	pPathObscuraModel->Release();
}

const CRCPathObscuraPropagModel& CRCPathObscuraPropagModel::operator=(const CRCPathObscuraPropagModel& original)
{
	if( &original != this )
	{
		*pPathObscuraModel = *(original.pPathObscuraModel);
	}

	return *this;
}

PropagationModel CRCPathObscuraPropagModel::Id()
{
	return 	PropagationModel::CRC_PATH_OBSCURA;
}

bool CRCPathObscuraPropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool CRCPathObscuraPropagModel::IsUsingMappedLandCoverData()
{
	return false;
}

bool CRCPathObscuraPropagModel::IsUsingItuRadioClimZoneData()
{
	return false;
}

bool CRCPathObscuraPropagModel::IsUsingSurfaceElevData()
{
	return true;
}

int CRCPathObscuraPropagModel::DefaultMappedLandCoverValue()
{
	return -1;
}

double CRCPathObscuraPropagModel::CalcPathLoss(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon,
                                               double txRcagl_m, double rxRcagl_m, unsigned int sizeProfiles,
                                               double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile)
{
double pathLoss_dB;
std::array<double, 7> features;

	if( sizeProfiles < 2 || distKmProfile[sizeProfiles-1] < 1E-5 )
		return 0;

	features = pComputeFeatures(freq_MHz, txLat, txLon, rxLat, rxLon, txRcagl_m, rxRcagl_m, sizeProfiles,
	                            distKmProfile, terrainElevProfile, surfaceElevProfile);

	pathLoss_dB = pPathObscuraModel->PathLoss(features[0], features[1], features[2], features[3],
	                                          features[4], features[5], features[6]);

	return pathLoss_dB;
}

double CRCPathObscuraPropagModel::pMedianEffectiveEarthRadius(double txLat, double txLon, double rxLat, double rxLon, double txRxPathLengthKm)
{
double pathCentreLat, pathCentreLon, deltaN;
double aeKm;

	ITURP_2001::GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, txRxPathLengthKm/2.0, pathCentreLat, pathCentreLon);
	deltaN = ITURP_DigitalMaps::DN50(pathCentreLat, pathCentreLon);
	aeKm = 157.0*ITURP_2001::Re / (157.0-deltaN); // see eq.(11) from ITU-R P.2001-5
	return aeKm;
}

// return value: pi/2 = towards sky (zenith), -pi/2 = towards ground (nadir), 0 = parallel to ground.
double CRCPathObscuraPropagModel::pElevationAngleRad(double hamsl_from, double hamsl_to, double distKm, double aeKm)
{
	distKm = std::max(1E-5, distKm); // avoid div by zero
	return atan(((hamsl_to-hamsl_from)/(1000.0*distKm))-(distKm/(2.0*aeKm))); // see eq.(79) from ITU-R P.1812-7
}

double CRCPathObscuraPropagModel::pFirstFresnelZoneRadius(double freq_MHz, double distToTxKm, double distToRxKm)
{
double freq_GHz = freq_MHz/1000;
double pathLengthKm = distToTxKm + distToRxKm;
double radius_m;

	pathLengthKm = std::max(1E-5, pathLengthKm); // avoid div by zero
	radius_m = 17.3*sqrt((distToTxKm*distToRxKm)/(freq_GHz*pathLengthKm)); // see eq.(3) from ITU-R P.530-18
	return radius_m; // in meters
}

// NOTE: txToRxElevAngleRad must be positive when towards sky, negative when towards ground
double CRCPathObscuraPropagModel::pFirstFresnelZoneMaxHeight(double freq_MHz, double distToTxKm, double distToRxKm,
                                                             double txHeight_mamsl, double txToRxElevAngleRad)
{
double fzRadius_m = pFirstFresnelZoneRadius(freq_MHz, distToTxKm, distToRxKm);
double fzMaxHeight_mamsl;

	// NOTE on tan(): "The function has mathematical poles at π(1/2 + n); however no common floating-point representation
	// is able to represent π/2 exactly, thus there is no value of the argument for which a pole error occurs"
	// (from https://en.cppreference.com).
	fzMaxHeight_mamsl = txHeight_mamsl + (distToTxKm*1000*tan(txToRxElevAngleRad)) + fzRadius_m;
	return fzMaxHeight_mamsl;// in meters above mean sea level
}

std::array<double, 7> CRCPathObscuraPropagModel::pComputeFeatures(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon,
                                                                  double txRcagl_m, double rxRcagl_m, unsigned int sizeProfiles,
                                                                  double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile)
{
double txHeight_mamsl = terrainElevProfile[0] + txRcagl_m; // tx height in meters above mean sea level
double rxHeight_mamsl = terrainElevProfile[sizeProfiles-1] + rxRcagl_m; // rx height in meters above mean sea level
double txRxPathLengthKm = distKmProfile[sizeProfiles-1];
double aeKm = pMedianEffectiveEarthRadius(txLat, txLon, rxLat, rxLon, txRxPathLengthKm);
double txToRxElevAngleRad = pElevationAngleRad(txHeight_mamsl, rxHeight_mamsl, txRxPathLengthKm, aeKm);
double totalObsDepthKm = 0;
double totalTerrainDepthKm = 0;
double firstBlockageLocationKm = 0; // in distance (km) from the tx
double lastBlockageLocationKm = 0;  // in distance (km) from the tx
double antennaHeightDiff_m = txHeight_mamsl - rxHeight_mamsl;
double txRxPathLength_m = txRxPathLengthKm*1000;
double dist3D_m = sqrt((txRxPathLength_m*txRxPathLength_m)+(antennaHeightDiff_m*antennaHeightDiff_m));
bool isObstructed_im1 = false; // is profile point at index i-1 obstructed
int obstructionBlockCount = 0;
int dsmMaximaAboveFzCount = 0;

	for(unsigned int i=0 ; i<sizeProfiles ; i++)
	{
		unsigned int prev_i = (i == 0) ? 0 : i-1;
		unsigned int next_i = (i == sizeProfiles-1) ? i : i+1;
		bool isObstructed_i = false;
		double surfElevAngleRad = pElevationAngleRad(txHeight_mamsl, surfaceElevProfile[i], distKmProfile[i], aeKm);
		if( surfElevAngleRad >= txToRxElevAngleRad )
		{
			isObstructed_i = true;
			double startDistKm_i = distKmProfile[i] - ((distKmProfile[i]-distKmProfile[prev_i])/2.0);
			double endDistKm_i   = distKmProfile[i] + ((distKmProfile[next_i]-distKmProfile[i])/2.0);
			double depthKm_i = endDistKm_i - startDistKm_i;

			totalObsDepthKm += depthKm_i;

			if( firstBlockageLocationKm == 0 )
				firstBlockageLocationKm = startDistKm_i;
			lastBlockageLocationKm = endDistKm_i;

			double terrElevAngleRad = pElevationAngleRad(txHeight_mamsl, terrainElevProfile[i], distKmProfile[i], aeKm);
			if( terrElevAngleRad >= txToRxElevAngleRad )
				totalTerrainDepthKm += depthKm_i;
		}
		else
		{
			if( isObstructed_im1 == true )
				obstructionBlockCount += 1;
		}
		isObstructed_im1 = isObstructed_i;

		// check for local DSM maxima
		if( surfaceElevProfile[i] > surfaceElevProfile[prev_i] && surfaceElevProfile[i] > surfaceElevProfile[next_i] )
		{
			double fzMaxHeight_mamsl = pFirstFresnelZoneMaxHeight(freq_MHz, distKmProfile[i],
				                                                  txRxPathLengthKm-distKmProfile[i],
                                                                  txHeight_mamsl, txToRxElevAngleRad);
			if( surfaceElevProfile[i] > fzMaxHeight_mamsl )
				dsmMaximaAboveFzCount += 1;
		}
	}

	// in case the rx antenna is below clutter (surface) height
	if( isObstructed_im1 == true )
		obstructionBlockCount += 1;

	/*
	Feature inputs for the CRC Path Obscura model:
	f1: Frequency in MHz
	f2: 3D distance between antennas, in meters
	f3: Total obstruction depth, in meters
	f4: Intersection depth (last_blockage_location – first_blockage_location), in meters
	f5: Number of obstruction (contiguous) blocks
	f6: Percentage of blockage that is terrain [% value between 0 and 1]
	f7: Number of local DSM (Digital Surface Model) maxima above the 1st Fresnel Zone
	*/
	double f1 = freq_MHz;
	double f2 = dist3D_m;
	double f3 = totalObsDepthKm*1000;
	double f4 = (lastBlockageLocationKm-firstBlockageLocationKm)*1000;
	double f5 = obstructionBlockCount;
	double f6 = (totalObsDepthKm > 0) ? totalTerrainDepthKm/totalObsDepthKm : 0;
	double f7 = dsmMaximaAboveFzCount;
	return {f1, f2, f3, f4, f5, f6, f7};
}
