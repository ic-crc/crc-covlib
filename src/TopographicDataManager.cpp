#include "TopographicDataManager.h"
#include "ITURP_2001.h"
#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/GeodesicLine.hpp>
#include <GeographicLib/Geocentric.hpp>
#include <cstring>
#include <algorithm>

using namespace Crc::Covlib;

TopographicDataManager::TopographicDataManager()
{
	pRadioClimaticZonesGeotiffs[ITU_R_P_1812] = GeoTIFFReader();
	pRadioClimaticZonesGeotiffs[ITU_R_P_452_V18] = GeoTIFFReader();
	pUsePairedSources = false;
}

TopographicDataManager::~TopographicDataManager()
{

}

void TopographicDataManager::AddTerrainElevSource(TerrainElevSource* source)
{
	pTerrainElevSources.push_back(source);
}

void TopographicDataManager::AddLandCoverSource(LandCoverSource* source)
{
	pLandCoverSources.push_back(source);
}

void TopographicDataManager::ClearTerrainElevSources()
{
	pTerrainElevSources.clear();
}

void TopographicDataManager::ClearLandCoverSources()
{
	pLandCoverSources.clear();
}

void TopographicDataManager::AddSurfaceElevSource(SurfaceElevSource* source)
{
	pSurfaceElevSources.push_back(source);
}

void TopographicDataManager::ClearSurfaceElevSources()
{
	pSurfaceElevSources.clear();
}

void TopographicDataManager::AddPairedTerrainAndSurfaceElevSources(TerrainElevSource* terrainElevSource, SurfaceElevSource* surfaceElevSource)
{
	pPairedTerrSurfElevSources.push_back(std::pair<TerrainElevSource*, SurfaceElevSource*>(terrainElevSource, surfaceElevSource));
}

void TopographicDataManager::ClearPairedTerrainAndSurfaceElevSources()
{
	pPairedTerrSurfElevSources.clear();
}

void TopographicDataManager::SetRadioClimaticZonesFile(Crc::Covlib::PropagationModel propagModel, const char* pathname)
{
	if( propagModel == Crc::Covlib::PropagationModel::ITU_R_P_452_V17 ) // both versions share shame GeoTIFFReader object
		propagModel = ITU_R_P_452_V18;
	pRadioClimaticZonesGeotiffs[propagModel].SetFile(pathname);
}

const char* TopographicDataManager::GetRadioClimaticZonesFile(Crc::Covlib::PropagationModel propagModel) const
{
	if( propagModel == Crc::Covlib::PropagationModel::ITU_R_P_452_V17 )
		propagModel = ITU_R_P_452_V18;
	return pRadioClimaticZonesGeotiffs.at(propagModel).GetFile();
}

void TopographicDataManager::UsePairedTerrainAndSurfaceElevSources(bool usePairedSources)
{
	pUsePairedSources = usePairedSources;
}

bool TopographicDataManager::UsePairedTerrainAndSurfaceElevSources() const
{
	return pUsePairedSources;
}

bool TopographicDataManager::GetTerrainElevation(double lat, double lon, float* terrainElevation)
{
	if( pUsePairedSources == false )
		return pGetUnpairedTerrainElevation(lat, lon, terrainElevation);
	else
	{
		float surfaceElevation;
		return pGetPairedTerrainAndSurfaceElev(lat, lon, terrainElevation, &surfaceElevation);
	}
}

bool TopographicDataManager::pGetUnpairedTerrainElevation(double lat, double lon, float* terrainElevation)
{
bool success;

	for(std::size_t i=0 ; i<pTerrainElevSources.size() ; i++)
	{
		try
		{
			success = pTerrainElevSources[i]->GetTerrainElevation(lat, lon, terrainElevation);
		}
		catch(const std::exception& e)
		{
			ReleaseResources(true);
			success = pTerrainElevSources[i]->GetTerrainElevation(lat, lon, terrainElevation);
		}
		
		if( success )
			return true;
	}
	return false;
}

bool TopographicDataManager::GetLandCover(double lat, double lon, int* landCover)
{
bool success;

	for(std::size_t i=0 ; i<pLandCoverSources.size() ; i++)
	{
		try
		{
			success = pLandCoverSources[i]->GetLandCoverClass(lat, lon, landCover);
		}
		catch(const std::exception& e)
		{
			ReleaseResources(true);
			success = pLandCoverSources[i]->GetLandCoverClass(lat, lon, landCover);
		}
		
		if( success )
			return true;
	}
	return false;
}

bool TopographicDataManager::GetLandCoverMappedValue(double lat, double lon, PropagationModel propagModel, int* modelValue)
{
bool success;

	for(std::size_t i=0 ; i<pLandCoverSources.size() ; i++)
	{
		try
		{
			success = pLandCoverSources[i]->GetPropagModelLandCoverClass(lat, lon, propagModel, modelValue);
		}
		catch(const std::exception& e)
		{
			ReleaseResources(true);
			success = pLandCoverSources[i]->GetPropagModelLandCoverClass(lat, lon, propagModel, modelValue);
		}
		
		if( success )
			return true;
	}
	return false;
}

bool TopographicDataManager::GetSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
	if( pUsePairedSources == false )
		return pGetUnpairedSurfaceElevation(lat, lon, surfaceElevation);
	else
	{
		float terrainElevation;
		return pGetPairedTerrainAndSurfaceElev(lat, lon, &terrainElevation, surfaceElevation);
	}	
}

bool TopographicDataManager::pGetUnpairedSurfaceElevation(double lat, double lon, float* surfaceElevation)
{
bool success;

	for(std::size_t i=0 ; i<pSurfaceElevSources.size() ; i++)
	{
		try
		{
			success = pSurfaceElevSources[i]->GetSurfaceElevation(lat, lon, surfaceElevation);
		}
		catch(const std::exception& e)
		{
			ReleaseResources(true);
			success = pSurfaceElevSources[i]->GetSurfaceElevation(lat, lon, surfaceElevation);
		}
		
		if( success )
			return true;
	}
	return false;
}

bool TopographicDataManager::pGetPairedTerrainAndSurfaceElev(double lat, double lon, float* terrainElevation, float* surfaceElevation)
{
bool successTerr, successSurf;

	for(std::size_t i=0 ; i<pPairedTerrSurfElevSources.size() ; i++)
	{
		try
		{
			successTerr = pPairedTerrSurfElevSources[i].first->GetTerrainElevation(lat, lon, terrainElevation);
			successSurf = pPairedTerrSurfElevSources[i].second->GetSurfaceElevation(lat, lon, surfaceElevation);
		}
		catch(const std::exception& e)
		{
			ReleaseResources(true);
			successTerr = pPairedTerrSurfElevSources[i].first->GetTerrainElevation(lat, lon, terrainElevation);
			successSurf = pPairedTerrSurfElevSources[i].second->GetSurfaceElevation(lat, lon, surfaceElevation);
		}
		
		if( successTerr && successSurf )
			return true;
	}
	return false;
}

bool TopographicDataManager::pGetRadioClimaticZone(GeoTIFFReader& source, double lat, double lon, ITURadioClimaticZone* zone)
{
bool success;
int value;
	
	try
	{
		success = source.GetClosestIntValue(lat, lon, &value);
	}
	catch(const std::exception& e)
	{
		ReleaseResources(true);
		success = source.GetClosestIntValue(lat, lon, &value);
	}
	
	if( success )
		*zone = (ITURadioClimaticZone)value;
	return success;
}

double TopographicDataManager::GetDistKm(double startLat, double startLon, double endLat, double endLon, GeodesicAlgo geodesicAlgo)
{
double distKm = 0;

	if( geodesicAlgo == ITU_GREAT_CIRCLE)
		ITURP_2001::GreatCircleDistance(startLat, startLon, endLat, endLon, distKm);
	else
	{
		double distMeters;
		const GeographicLib::Geodesic& geod = GeographicLib::Geodesic::WGS84();
		geod.Inverse(startLat, startLon, endLat, endLon, distMeters);
		distKm = distMeters/1000.0;
	}
	return distKm;
}

void TopographicDataManager::GetIntermediatePoint(double startLat, double startLon, double endLat, double endLon, double distKm, GeodesicAlgo geodesicAlgo, double* lat, double* lon)
{
	if( geodesicAlgo == ITU_GREAT_CIRCLE)
		ITURP_2001::GreatCircleIntermediatePoint(startLat, startLon, endLat, endLon, distKm, *lat, *lon);
	else
	{
		double azi1, azi2;
		const GeographicLib::Geodesic& geod = GeographicLib::Geodesic::WGS84();
		geod.Inverse(startLat, startLon, endLat, endLon, azi1, azi2);
		GeographicLib::GeodesicLine line = geod.Line(startLat, startLon, azi1);
		line.Position(distKm*1000.0, *lat, *lon);
	}
}

double TopographicDataManager::GetStraightLineDistKm(double startLat, double startLon, double startHeight_m, double endLat, double endLon, double endHeight_m, GeodesicAlgo geodesicAlgo)
{
	if( geodesicAlgo == ITU_GREAT_CIRCLE)
	{
	// note: algorithm taken from ITU-R P.619-5
	const double PI_ON_180 = 0.017453292519943295769;
	const double RE = 6371.0;

		double deltaLon_deg = endLon-startLon;
		if( deltaLon_deg > 180 )
			deltaLon_deg -= 360;
		else if( deltaLon_deg < -180 )
			deltaLon_deg += 360;

		double R0 = RE + (startHeight_m/1000.0);
		double R1 = RE + (endHeight_m/1000.0);

		double cos_d   = cos(deltaLon_deg*PI_ON_180);
		double sin_d   = sin(deltaLon_deg*PI_ON_180);
		double cosLat1 = cos(endLat*PI_ON_180);
		double cosLat0 = cos(startLat*PI_ON_180);
		double sinLat1 = sin(endLat*PI_ON_180);
		double sinLat0 = sin(startLat*PI_ON_180);

		double X1 = R1*cosLat1*cos_d;
		double Y1 = R1*cosLat1*sin_d;
		double Z1 = R1*sinLat1;

		double X2 = X1*sinLat0 - Z1*cosLat0;
		double Y2 = Y1;
		double Z2 = Z1*sinLat0 + X1*cosLat0 - R0;

		return sqrt(X2*X2 + Y2*Y2 + Z2*Z2);
	}
	else
	{
	const GeographicLib::Geocentric& earth = GeographicLib::Geocentric::WGS84();
	double x1, y1, z1;
	double x2, y2, z2;
	double dist_meters;

		earth.Forward(startLat, startLon, startHeight_m, x1, y1, z1);
		earth.Forward(endLat, endLon, endHeight_m, x2, y2, z2);
		dist_meters = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
		return dist_meters / 1000.0;
	}
}

void TopographicDataManager::GetLatLonProfileByRes(double startLat, double startLon, double endLat, double endLon, double resolutionKm, GeodesicAlgo geodesicAlgo,
                                                   std::vector<std::pair<double,double>>* latLonProfile, std::vector<double>* distKmProfile)
{
double distKm;
unsigned int numPoints;

	distKm = GetDistKm(startLat, startLon, endLat, endLon, geodesicAlgo);
	numPoints = 1 + (unsigned int)((distKm / resolutionKm) + 1);
	numPoints = std::max(3u, numPoints);
	return GetLatLonProfileByNumPoints(startLat, startLon, endLat, endLon, numPoints, geodesicAlgo, latLonProfile, distKmProfile);
}

void TopographicDataManager::GetLatLonProfileByNumPoints(double startLat, double startLon, double endLat, double endLon, unsigned int numPoints, GeodesicAlgo geodesicAlgo,
                                                         std::vector<std::pair<double,double>>* latLonProfile, std::vector<double>* distKmProfile)
{
double distKm, deltaDistKm;
double lat, lon;

	distKmProfile->clear();
	distKmProfile->reserve(numPoints);
	distKm = GetDistKm(startLat, startLon, endLat, endLon, geodesicAlgo);
	deltaDistKm = distKm/(numPoints-1);
	for (unsigned int i=0 ; i<numPoints ; i++)
		distKmProfile->push_back(i*deltaDistKm);

	if( geodesicAlgo == ITU_GREAT_CIRCLE )
		ITURP_2001::GreatCircleIntermediatePoints(startLat, startLon, endLat, endLon, *distKmProfile, *latLonProfile);
	else
	{
		latLonProfile->clear();
		latLonProfile->reserve(numPoints);
		for (unsigned int i=0 ; i<numPoints ; i++)
		{
			GetIntermediatePoint(startLat, startLon, endLat, endLon, deltaDistKm*i, geodesicAlgo, &lat, &lon);
			latLonProfile->push_back(std::pair<double,double>(lat, lon));
		}
	}

	// make sure start and end points stay exactly the same
	latLonProfile->operator[](0).first = startLat;
	latLonProfile->operator[](0).second = startLon;
	latLonProfile->back().first = endLat;
	latLonProfile->back().second = endLon;
}

int TopographicDataManager::GetTerrainElevProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<double>* terrainElevProfile)
{
double lat, lon;
float terrainElev;
int numMisses = 0;

	terrainElevProfile->clear();
	terrainElevProfile->reserve(latLonProfile.size());

	for(unsigned int i=0 ; i<latLonProfile.size() ; i++)
	{
		lat = latLonProfile[i].first;
		lon = latLonProfile[i].second;
		if ( GetTerrainElevation(lat, lon, &terrainElev) == false)
		{
			terrainElevProfile->push_back(0);
			if( pTerrainElevSources.size() > 0 )
				numMisses++;
		}
		else
			terrainElevProfile->push_back(terrainElev);
	}
	return numMisses;
}

int TopographicDataManager::GetLandCoverProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<int>* landCoverProfile)
{
double lat, lon;
int landCover;
int numMisses = 0;

	landCoverProfile->clear();
	landCoverProfile->reserve(latLonProfile.size());

	for(unsigned int i=0 ; i<latLonProfile.size() ; i++)
	{
		lat = latLonProfile[i].first;
		lon = latLonProfile[i].second;
		if ( GetLandCover(lat, lon, &landCover) == false)
		{
			landCoverProfile->push_back(-1);
			if( pLandCoverSources.size() > 0 )
				numMisses++;
		}
		else
			landCoverProfile->push_back(landCover);
	}
	return numMisses;
}

int TopographicDataManager::GetMappedLandCoverProfile(std::vector<std::pair<double,double>>& latLonProfile, PropagationModel propagModel,
                                                      int defaultValue, std::vector<int>* mappedLandCoverProfile)
{
double lat, lon;
int mappedValue;
int numMisses = 0;

	mappedLandCoverProfile->clear();
	mappedLandCoverProfile->reserve(latLonProfile.size());

	for(unsigned int i=0 ; i<latLonProfile.size() ; i++)
	{
		lat = latLonProfile[i].first;
		lon = latLonProfile[i].second;
		if( GetLandCoverMappedValue(lat, lon, propagModel, &mappedValue) == false )
		{
			mappedLandCoverProfile->push_back(defaultValue);
			if( pLandCoverSources.size() > 0 )
				numMisses++;
		}
		else
			mappedLandCoverProfile->push_back(mappedValue);
	}

	return numMisses;
}

int TopographicDataManager::GetRadioClimaticZoneProfile(std::vector<std::pair<double,double>>& latLonProfile, PropagationModel propagModel,
                                                        std::vector<ITURadioClimaticZone>* radioClimaticZoneProfile)
{
	if( propagModel == Crc::Covlib::PropagationModel::ITU_R_P_452_V17 ) // both versions share shame GeoTIFFReader object
		propagModel = ITU_R_P_452_V18;
	GeoTIFFReader& source = pRadioClimaticZonesGeotiffs[propagModel];
	return pGetRadioClimaticZoneProfile(source, latLonProfile, radioClimaticZoneProfile);
}

int TopographicDataManager::GetSurfaceElevProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<double>* surfaceElevProfile)
{
double lat, lon;
float surfaceElev;
int numMisses = 0;

	surfaceElevProfile->clear();
	surfaceElevProfile->reserve(latLonProfile.size());

	for(unsigned int i=0 ; i<latLonProfile.size() ; i++)
	{
		lat = latLonProfile[i].first;
		lon = latLonProfile[i].second;
		if ( GetSurfaceElevation(lat, lon, &surfaceElev) == false)
		{
			surfaceElevProfile->push_back(0);
			if( pSurfaceElevSources.size() > 0 )
				numMisses++;
		}
		else
			surfaceElevProfile->push_back(surfaceElev);
	}
	return numMisses;
}

int TopographicDataManager::pGetRadioClimaticZoneProfile(GeoTIFFReader& source, std::vector<std::pair<double,double>>& latLonProfile,
                                                         std::vector<ITURadioClimaticZone>* radioClimaticZoneProfile)
{
double lat, lon;
ITURadioClimaticZone zone;
bool isRCZFileSpecified = (strlen(source.GetFile()) != 0);
int numMisses = 0;

	radioClimaticZoneProfile->clear();
	radioClimaticZoneProfile->reserve(latLonProfile.size());

	for(unsigned int i=0 ; i<latLonProfile.size() ; i++)
	{
		lat = latLonProfile[i].first;
		lon = latLonProfile[i].second;
		if( isRCZFileSpecified == true ) 
		{
			if( pGetRadioClimaticZone(source, lat, lon, &zone) == false )
			{
				radioClimaticZoneProfile->push_back(ITURadioClimaticZone::ITU_INLAND);
				numMisses++;
			}
			else
				radioClimaticZoneProfile->push_back(zone);
		}
		else
			radioClimaticZoneProfile->push_back(ITURadioClimaticZone::ITU_INLAND);
	}
	return numMisses;
}

void TopographicDataManager::ReleaseResources(bool clearCaches)
{
	for(std::size_t i=0 ; i<pTerrainElevSources.size() ; i++)
		pTerrainElevSources[i]->ReleaseResources(clearCaches);
	for(std::size_t i=0 ; i<pLandCoverSources.size() ; i++)
		pLandCoverSources[i]->ReleaseResources(clearCaches);
	for(std::size_t i=0 ; i<pSurfaceElevSources.size() ; i++)
		pSurfaceElevSources[i]->ReleaseResources(clearCaches);
	// Note: don't use pointers from pPairedTerrSurfElevSources here, they should already be
	//       covered from pTerrainElevSources and pSurfaceElevSources.
	pRadioClimaticZonesGeotiffs[ITU_R_P_1812].CloseAllFiles(clearCaches);
	pRadioClimaticZonesGeotiffs[ITU_R_P_452_V18].CloseAllFiles(clearCaches);
}