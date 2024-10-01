#pragma once
#include <vector>
#include <map>
#include <utility>
#include "TerrainElevSource.h"
#include "LandCoverSource.h"
#include "SurfaceElevSource.h"
#include "GeoTIFFReader.h"


class TopographicDataManager
{
public:
	TopographicDataManager();
	virtual ~TopographicDataManager();

	void AddTerrainElevSource(TerrainElevSource* source);
	void ClearTerrainElevSources();
	void AddLandCoverSource(LandCoverSource* source);
	void ClearLandCoverSources();
	void AddSurfaceElevSource(SurfaceElevSource* source);
	void ClearSurfaceElevSources();
	void AddPairedTerrainAndSurfaceElevSources(TerrainElevSource* terrainElevSource, SurfaceElevSource* surfaceElevSource);
	void ClearPairedTerrainAndSurfaceElevSources();
	void SetRadioClimaticZonesFile(Crc::Covlib::PropagationModel propagModel, const char* pathname);
	const char* GetRadioClimaticZonesFile(Crc::Covlib::PropagationModel propagModel) const;

	bool GetTerrainElevation(double lat, double lon, float* terrainElevation);
	bool GetLandCover(double lat, double lon, int* landCover);
	bool GetLandCoverMappedValue(double lat, double lon, Crc::Covlib::PropagationModel propagModel, int* modelValue);
	bool GetSurfaceElevation(double lat, double lon, float* surfaceElevation);
	bool GetPairedTerrainAndSurfaceElev(double lat, double lon, float* terrainElevation, float* surfaceElevation);

	enum GeodesicAlgo
	{
		WGS84_PRECISE,
		ITU_GREAT_CIRCLE
	};

	void GetLatLonProfileByRes(double startLat, double startLon, double endLat, double endLon, double resolutionKm, GeodesicAlgo geodesicAlgo, std::vector<std::pair<double,double>>* latLonProfile, std::vector<double>* distKmProfile);
	void GetLatLonProfileByNumPoints(double startLat, double startLon, double endLat, double endLon, unsigned int numPoints, GeodesicAlgo geodesicAlgo, std::vector<std::pair<double,double>>* latLonProfile, std::vector<double>* distKmProfile);
	int GetTerrainElevProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<double>* terrainElevProfile);
	int GetLandCoverProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<int>* landCoverProfile);
	int GetMappedLandCoverProfile(std::vector<std::pair<double,double>>& latLonProfile, Crc::Covlib::PropagationModel propagModel, int defaultValue, std::vector<int>* mappedLandCoverProfile);
	int GetRadioClimaticZoneProfile(std::vector<std::pair<double,double>>& latLonProfile, Crc::Covlib::PropagationModel propagModel, std::vector<Crc::Covlib::ITURadioClimaticZone>* radioClimaticZoneProfile);
	int GetSurfaceElevProfile(std::vector<std::pair<double,double>>& latLonProfile, std::vector<double>* surfaceElevProfile);
	int GetPairedTerrainAndSurfaceElevProfiles(std::vector<std::pair<double,double>>& latLonProfile, std::vector<double>* terrainElevProfile, std::vector<double>* surfaceElevProfile);

	void ReleaseResources(bool clearCaches); // clear caches for topographic data and close file handles

	static double GetDistKm(double startLat, double startLon, double endLat, double endLon, GeodesicAlgo geodesicAlgo);
	static void GetIntermediatePoint(double startLat, double startLon, double endLat, double endLon, double distKm, GeodesicAlgo geodesicAlgo, double* lat, double* lon);
	static double GetStraightLineDistKm(double startLat, double startLon, double startHeight_m, double endLat, double endLon, double endHeight_m, GeodesicAlgo geodesicAlgo);

private:
	bool pGetRadioClimaticZone(GeoTIFFReader& source, double lat, double lon, Crc::Covlib::ITURadioClimaticZone* zone);
	int pGetRadioClimaticZoneProfile(GeoTIFFReader& source, std::vector<std::pair<double,double>>& latLonProfile, std::vector<Crc::Covlib::ITURadioClimaticZone>* radioClimaticZoneProfile);

	std::vector<TerrainElevSource*> pTerrainElevSources;
	std::vector<LandCoverSource*> pLandCoverSources;
	std::vector<SurfaceElevSource*> pSurfaceElevSources;
	std::vector<std::pair<TerrainElevSource*, SurfaceElevSource*>> pPairedTerrSurfElevSources;
	std::map<Crc::Covlib::PropagationModel, GeoTIFFReader> pRadioClimaticZonesGeotiffs;
};