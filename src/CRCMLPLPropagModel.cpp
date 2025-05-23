#include "CRCMLPLPropagModel.h"
#include "ITURP_2001.h"
#include "ITURP_DigitalMaps.h"
#include <cmath>

using namespace Crc::Covlib;


CRCMLPLPropagModel::CRCMLPLPropagModel() :
	pMLPLModel(NewMLPLModel())
{

}

CRCMLPLPropagModel::CRCMLPLPropagModel(const CRCMLPLPropagModel& original) :
	pMLPLModel(NewMLPLModel())
{
	*this = original;
}

CRCMLPLPropagModel::~CRCMLPLPropagModel()
{
	pMLPLModel->Release();
}

const CRCMLPLPropagModel& CRCMLPLPropagModel::operator=(const CRCMLPLPropagModel& original)
{
	if( &original != this )
	{
		pFreeSpaceModel = original.pFreeSpaceModel;
		*pMLPLModel = *(original.pMLPLModel);
	}

	return *this;
}

PropagationModel CRCMLPLPropagModel::Id()
{
	return 	PropagationModel::CRC_MLPL;
}

bool CRCMLPLPropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool CRCMLPLPropagModel::IsUsingMappedLandCoverData()
{
	return false;
}

bool CRCMLPLPropagModel::IsUsingItuRadioClimZoneData()
{
	return false;
}

bool CRCMLPLPropagModel::IsUsingSurfaceElevData()
{
	return true;
}

int CRCMLPLPropagModel::DefaultMappedLandCoverValue()
{
	return -1;
}

double CRCMLPLPropagModel::CalcPathLoss(double freq_MHz, double txLat, double txLon, double rxLat, double rxLon,
                                        double txRcagl_m, double rxRcagl_m, unsigned int sizeProfiles,
                                        double* distKmProfile, double* terrainElevProfile, double* surfaceElevProfile)
{
	if( sizeProfiles < 2 || distKmProfile[sizeProfiles-1] < 1E-5 )
		return 0;

double fspl_dB;
double mlplExcessLoss_dB;
double pathLength_m = distKmProfile[sizeProfiles-1]*1000.0;
double obsDepth_m;

	fspl_dB = pFreeSpaceModel.CalcPathLoss(freq_MHz, txRcagl_m, rxRcagl_m, sizeProfiles, distKmProfile, terrainElevProfile);

	obsDepth_m = pObstructionDepth(txLat, txLon, rxLat, rxLon, txRcagl_m, rxRcagl_m, sizeProfiles, distKmProfile,
	                               terrainElevProfile, surfaceElevProfile);

	mlplExcessLoss_dB = pMLPLModel->ExcessLoss(freq_MHz, pathLength_m, obsDepth_m);

	return fspl_dB + mlplExcessLoss_dB;
}

// Algorithm derived from:
//   ITU-R P.1812-7, Attachment 1 to Annex 1, Section 4 & 5.3
//   ITU-R P.452-18, Attachment 2 to Annex 1, Section 4 & 5.1.3
double CRCMLPLPropagModel::pObstructionDepth(double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m,
                                             double rxRcagl_m, unsigned int sizeProfiles, double* distKmProfile,
                                             double* terrainElevProfile, double* surfaceElevProfile)
{
double pathCentreLat, pathCentreLon;
double totalDistKm = distKmProfile[sizeProfiles-1];
double txHeight_mamsl = terrainElevProfile[0] + txRcagl_m; // tx height in meters above mean sea level
double rxHeight_mamsl = terrainElevProfile[sizeProfiles-1] + rxRcagl_m; // rx height in meters above mean sea level
double deltaN; // average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere (N-units/km)
double ae; // median effective Earth radius (km)
double txToRxElevAngleRad, elevAngleRad;
double totalObsDepth_km = 0;
unsigned int next_i, prev_i;

	ITURP_2001::GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, totalDistKm/2.0, pathCentreLat, pathCentreLon);
	deltaN = ITURP_DigitalMaps::DN50(pathCentreLat, pathCentreLon);
	ae = 157.0*ITURP_2001::Re / (157.0-deltaN);

	txToRxElevAngleRad = pElevationAngleRad(txHeight_mamsl, rxHeight_mamsl, totalDistKm, ae);

	for(unsigned int i=0 ; i<sizeProfiles ; i++)
	{
		elevAngleRad = pElevationAngleRad(txHeight_mamsl, surfaceElevProfile[i], distKmProfile[i], ae);
		if( elevAngleRad >= txToRxElevAngleRad )
		{
			prev_i = (i == 0) ? 0 : i-1;
			next_i = (i == sizeProfiles-1) ? i : i+1;
			totalObsDepth_km += (distKmProfile[next_i]-distKmProfile[i])/2.0 + (distKmProfile[i]-distKmProfile[prev_i])/2.0;
		}
	}

	return totalObsDepth_km * 1000.0;
}

// return value: pi/2 = towards sky, -pi/2 = towards ground, 0 = parallel to ground.
double CRCMLPLPropagModel::pElevationAngleRad(double hamsl_from, double hamsl_to, double distKm, double aeKm)
{
	distKm = std::max(1E-5, distKm); // avoid div by zero
	return atan(((hamsl_to-hamsl_from)/(1000.0*distKm))-(distKm/(2.0*aeKm)));
}
