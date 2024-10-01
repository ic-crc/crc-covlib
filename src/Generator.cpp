#include "Generator.h"
#include "Simulation.h"
#include "ITURP_2001.h"
#include "ITURP_DigitalMaps.h"
#include <GeographicLib/Geodesic.hpp>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace Crc::Covlib;


Generator::Generator(void)
{
	pRunPointCount = 0;
}

Generator::~Generator(void)
{
}

void Generator::RunAreaCalculation(Simulation& sim)
{
PathLossFuncOutput rxPosResult;
MissesStats areaStats = {0,0,0,0,0};
CustomData nullCustomData = {0, nullptr, nullptr, nullptr, nullptr};
Position rxPos;
double rxAreaMinLat, rxAreaMinLon, rxAreaMaxLat, rxAreaMaxLon;

	pConditionalResourceRelease(sim, 0U);

	sim.pRxAreaResults.SetDataUnit(pGetResultUnitString(sim.pResultType));
	sim.pRxAreaResults.SetDataDescription(pGetResultNameString(sim.pResultType));

	sim.pRxAreaResults.GetBordersCoordinates(&rxAreaMinLat, &rxAreaMinLon, &rxAreaMaxLat, &rxAreaMaxLon);

	for(unsigned int x=0 ; x<sim.pRxAreaResults.SizeX() ; x++)
	{
		for(unsigned int y=0 ; y<sim.pRxAreaResults.SizeY() ; y++)
		{
			rxPos = sim.pRxAreaResults.GetPos(x, y);

			rxPosResult = pPathLoss(sim, rxPos.m_lat, rxPos.m_lon, nullCustomData, nullptr);

			areaStats.numPoints += rxPosResult.stats.numPoints;
			areaStats.terrainElevMisses += rxPosResult.stats.terrainElevMisses;
			areaStats.landCoverMisses += rxPosResult.stats.landCoverMisses;
			areaStats.radioClimaticZoneMisses += rxPosResult.stats.radioClimaticZoneMisses;
			areaStats.surfaceElevMisses += rxPosResult.stats.surfaceElevMisses;

			sim.pRxAreaResults.SetData(x, y, (float) pToSelectedResultType(sim, rxPos.m_lat, rxPos.m_lon, pLatLonProfile.size(),
			                                                               pDistKmProfile.data(), pTerrainElevProfile.data(), rxPosResult.pathLoss));
		}
	}

	sim.pTopoManager.ReleaseResources(true);

	sim.pGenerateStatus = pGetStatus(areaStats);
}

double Generator::RunPointCalculation(Simulation& sim, double lat, double lon, unsigned int numCustomSamples/*=0*/,
                                      const double* customTerrainElevProfile/*=nullptr*/,
                                      const int* customLandCoverMappedValueProfile/*=nullptr*/,
                                      const double* customSurfaceElevProfile/*=nullptr*/,
                                      const Crc::Covlib::ITURadioClimaticZone* customItuRadioClimaticZoneProfile/*=nullptr*/)
{
PathLossFuncOutput pathlossOutput = {0, {0,0,0,0,0}};
Position rxPos = {lat, lon};
CustomData customData;
double result;

	// Note: do not use pTerrainManager.SetWorkingArea()/ReleaseWorkingArea() here as it will degrade performance

	customData.numSamples = numCustomSamples;
	customData.terrainElevProfile = customTerrainElevProfile;
	customData.mappedLandCoverProfile = customLandCoverMappedValueProfile;
	customData.ituRCZProfile = customItuRadioClimaticZoneProfile;
	customData.surfaceElevProfile = customSurfaceElevProfile;

	pConditionalResourceRelease(sim, 1E5);

	pathlossOutput = pPathLoss(sim, rxPos.m_lat, rxPos.m_lon, customData, nullptr);
	sim.pGenerateStatus = pGetStatus(pathlossOutput.stats);
	result = pToSelectedResultType(sim, rxPos.m_lat, rxPos.m_lon, pLatLonProfile.size(), pDistKmProfile.data(), pTerrainElevProfile.data(), pathlossOutput.pathLoss);

	pRunPointCount++;

	return result;
}

bool Generator::ExportProfilesToCsvFile(Simulation& sim, const char* pathname, double lat, double lon)
{
PathLossFuncOutput pathlossOutput = {0, {0,0,0,0,0}};
std::string propagModelName = pGetPropagModelShortName(sim.pPropagModelId);
PropagModel* propagModelPtr = pGetPropagModelPtr(sim);
Position rxPos = {lat, lon};
CustomData nullCustomData = {0, nullptr, nullptr, nullptr, nullptr};
std::vector<double> resultProfile;
std::vector<int> unmappedlandCoverProfile;
std::vector<double> reprClutterHeightProfile;
bool success = false;
std::ofstream csvFile;

	pConditionalResourceRelease(sim, 0U);

	pathlossOutput = pPathLoss(sim, rxPos.m_lat, rxPos.m_lon, nullCustomData, &resultProfile);
	sim.pGenerateStatus = pGetStatus(pathlossOutput.stats);

	for(size_t i=0 ; i<resultProfile.size() ; i++)
		resultProfile[i] = pToSelectedResultType(sim, pLatLonProfile[i].first, pLatLonProfile[i].second, i+1,
		                                         pDistKmProfile.data(), pTerrainElevProfile.data(), resultProfile[i]);

	// get extra profiles not readily available but useful to have exported in .csv file
	if( propagModelPtr->IsUsingMappedLandCoverData() )
	{
		sim.pTopoManager.GetLandCoverProfile(pLatLonProfile, &unmappedlandCoverProfile);
		if( sim.pPropagModelId == ITU_R_P_1812 )
			sim.pIturp1812Model.GetReprClutterHeightProfile(pMappedLandCoverProfile.size(), pMappedLandCoverProfile.data(), &reprClutterHeightProfile);
		if( sim.pPropagModelId == ITU_R_P_452_V18 )
			sim.pIturp452v18Model.GetReprClutterHeightProfile(pMappedLandCoverProfile.size(), pMappedLandCoverProfile.data(), &reprClutterHeightProfile);
	}

	csvFile.open(pathname, std::ios::out | std::ios::trunc);
	if(csvFile)
	{
		// write header row
		csvFile << "latitude,longitude,path length (km),";
		csvFile << pGetResultNameString(sim.pResultType) << " (" << pGetResultUnitString(sim.pResultType) << "),";
		csvFile << "terrain elevation (m)";
		if( unmappedlandCoverProfile.size() > 0 )
			csvFile << ",land cover class";
		if( pMappedLandCoverProfile.size() > 0 )
			csvFile << "," << propagModelName << " clutter category";
		if( reprClutterHeightProfile.size() > 0 )
			csvFile << "," << propagModelName << " representative clutter height (m)";
		if( pSurfaceElevProfile.size() > 0 )
			csvFile << ",surface elevation (m)";
		if( pRadioClimaticZoneProfile.size() > 0 )
			csvFile << "," << propagModelName << " radio climatic zone";
		csvFile << std::endl;

		// write data rows
		csvFile << std::fixed << std::showpoint;
		for(unsigned int i=0 ; i<resultProfile.size() ; i++)
		{
			csvFile << std::setprecision(7) << pLatLonProfile[i].first << "," << pLatLonProfile[i].second << ","
			        << std::setprecision(5) << pDistKmProfile[i] << ",";
			csvFile << std::setprecision(5) << resultProfile[i];
			csvFile << "," << std::setprecision(5) << pTerrainElevProfile[i];
			if( unmappedlandCoverProfile.size() > 0 )
				csvFile << "," << unmappedlandCoverProfile[i];
			if( pMappedLandCoverProfile.size() > 0 )
				csvFile << "," << pMappedLandCoverProfile[i];
			if( reprClutterHeightProfile.size() > 0 )
				csvFile << "," << std::setprecision(1) << reprClutterHeightProfile[i];
			if( pSurfaceElevProfile.size() > 0 )
				csvFile << "," << std::setprecision(5) << pSurfaceElevProfile[i];
			if( pRadioClimaticZoneProfile.size() > 0 )
				csvFile << "," << (int) pRadioClimaticZoneProfile[i];
			csvFile << std::endl;
		}

		success = true;
	}
	csvFile.close();

	sim.pTopoManager.ReleaseResources(true);

	return success;
}

void Generator::pConditionalResourceRelease(Simulation& sim, unsigned int pointCountLimit)
{
	if( pRunPointCount > pointCountLimit )
	{
		sim.pTopoManager.ReleaseResources(true);
		pRunPointCount = 0;
	}
}

Generator::PathLossFuncOutput Generator::pPathLoss(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                   std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result;

	switch (sim.pPropagModelId)
	{
	case LONGLEY_RICE:
		result = pPathLossLongleyRice(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	case ITU_R_P_1812:
		result = pPathLossP1812(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	case ITU_R_P_452_V17:
		result = pPathLossP452v17(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	case ITU_R_P_452_V18:
		result = pPathLossP452v18(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	case FREE_SPACE:
		result = pPathLossFreeSpace(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	case EXTENDED_HATA:
		result = pPathLossEHata(sim, rxLat, rxLon, customData, optionalOutputPathLossProfile);
		break;
	default:
		result = {0, {0,0,0,0,0}};
		break;
	}

	result.pathLoss += pAdditionalPathLosses(sim, optionalOutputPathLossProfile);

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossLongleyRice(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                              std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pLongleyRiceModel), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		result.pathLoss = sim.pLongleyRiceModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL, sim.pTx.pol, pTerrainElevProfile.size(), pDistKmProfile[1],
		                                                     pTerrainElevProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			result.pathLoss = sim.pLongleyRiceModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL, sim.pTx.pol, i+1, pDistKmProfile[1], pTerrainElevProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossP1812(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                        std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pIturp1812Model), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		result.pathLoss = sim.pIturp1812Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, sim.pTx.rcagl, sim.pRx.heightAGL, sim.pTx.pol,
	                                                       pLatLonProfile.size(), pDistKmProfile.data(), pTerrainElevProfile.data(), pMappedLandCoverProfile.data(),
		                                                   pSurfaceElevProfile.data(), pRadioClimaticZoneProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			result.pathLoss = sim.pIturp1812Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, pLatLonProfile[i].first, pLatLonProfile[i].second,
			                                                   sim.pTx.rcagl, sim.pRx.heightAGL, sim.pTx.pol, i+1, pDistKmProfile.data(), pTerrainElevProfile.data(),
			                                                   pMappedLandCoverProfile.data(), pSurfaceElevProfile.data(), pRadioClimaticZoneProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossP452v17(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                          std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};
double txAntennaGain_dBi, rxAntennaGain_dBi;

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pIturp452v17Model), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		pGetAntennaGains(sim, rxLat, rxLon, pLatLonProfile.size(), pDistKmProfile.data(), pTerrainElevProfile.data(), &txAntennaGain_dBi, &rxAntennaGain_dBi);
		result.pathLoss = sim.pIturp452v17Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, sim.pTx.rcagl, sim.pRx.heightAGL,
		                                                     txAntennaGain_dBi, rxAntennaGain_dBi, sim.pTx.pol, pLatLonProfile.size(), pDistKmProfile.data(),
		                                                     pTerrainElevProfile.data(), pRadioClimaticZoneProfile.data(),
		                                                     (P452HeightGainModelClutterCategory*) pMappedLandCoverProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			pGetAntennaGains(sim, pLatLonProfile[i].first, pLatLonProfile[i].second, i+1, pDistKmProfile.data(), pTerrainElevProfile.data(),
			                 &txAntennaGain_dBi, &rxAntennaGain_dBi);
			result.pathLoss = sim.pIturp452v17Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, pLatLonProfile[i].first, pLatLonProfile[i].second,
			                                                     sim.pTx.rcagl, sim.pRx.heightAGL, txAntennaGain_dBi, rxAntennaGain_dBi, sim.pTx.pol, i+1, 
			                                                     pDistKmProfile.data(), pTerrainElevProfile.data(), pRadioClimaticZoneProfile.data(),
			                                                     (P452HeightGainModelClutterCategory*) pMappedLandCoverProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossP452v18(Simulation& sim, double rxLat, double rxLon, CustomData customData, 
                                                          std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};
double txAntennaGain_dBi, rxAntennaGain_dBi;

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pIturp452v18Model), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		pGetAntennaGains(sim, rxLat, rxLon, pLatLonProfile.size(), pDistKmProfile.data(), pTerrainElevProfile.data(), &txAntennaGain_dBi, &rxAntennaGain_dBi);
		result.pathLoss = sim.pIturp452v18Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, sim.pTx.rcagl, sim.pRx.heightAGL,
		                                                     txAntennaGain_dBi, rxAntennaGain_dBi, sim.pTx.pol, pLatLonProfile.size(), pDistKmProfile.data(),
		                                                     pTerrainElevProfile.data(), pMappedLandCoverProfile.data(), pSurfaceElevProfile.data(),
		                                                     pRadioClimaticZoneProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			pGetAntennaGains(sim, pLatLonProfile[i].first, pLatLonProfile[i].second, i+1, pDistKmProfile.data(), pTerrainElevProfile.data(),
			                 &txAntennaGain_dBi, &rxAntennaGain_dBi);
			result.pathLoss = sim.pIturp452v18Model.CalcPathLoss(sim.pTx.freqMHz/1000.0, sim.pTx.lat, sim.pTx.lon, pLatLonProfile[i].first, pLatLonProfile[i].second,
			                                                     sim.pTx.rcagl, sim.pRx.heightAGL, txAntennaGain_dBi, rxAntennaGain_dBi, sim.pTx.pol, i+1, pDistKmProfile.data(),
			                                                     pTerrainElevProfile.data(), pMappedLandCoverProfile.data(), pSurfaceElevProfile.data(), pRadioClimaticZoneProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossFreeSpace(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                            std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pFreeSpaceModel), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		result.pathLoss = sim.pFreeSpaceModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL,
		                                                   pLatLonProfile.size(), pDistKmProfile.data(), pTerrainElevProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			result.pathLoss = sim.pFreeSpaceModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL,
			                                                   i+1, pDistKmProfile.data(), pTerrainElevProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::PathLossFuncOutput Generator::pPathLossEHata(Simulation& sim, double rxLat, double rxLon, CustomData customData,
                                                        std::vector<double>* optionalOutputPathLossProfile)
{
PathLossFuncOutput result = {0, {0,0,0,0,0}};

	result.stats = pFillProfiles(sim, rxLat, rxLon, (PropagModel*) &(sim.pEHataModel), customData);

	if( optionalOutputPathLossProfile == nullptr )
	{
		result.pathLoss = sim.pEHataModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL, pTerrainElevProfile.size(),
		                                               pDistKmProfile[1], pTerrainElevProfile.data());
	}
	else
	{
		optionalOutputPathLossProfile->clear();
		optionalOutputPathLossProfile->reserve(pLatLonProfile.size());
		for(unsigned int i=0 ; i<pLatLonProfile.size() ; i++)
		{
			result.pathLoss = sim.pEHataModel.CalcPathLoss(sim.pTx.freqMHz, sim.pTx.rcagl, sim.pRx.heightAGL,
			                                               i+1, pDistKmProfile[1], pTerrainElevProfile.data());
			optionalOutputPathLossProfile->push_back(result.pathLoss);
		}
	}

	return result;
}

Generator::MissesStats Generator::pFillProfiles(Simulation& sim, double rxLat, double rxLon, PropagModel* propagModel, CustomData customData)
{
MissesStats stats = {0,0,0,0,0};
const int MIN_SAMPLES = 3;
bool customElev = (customData.numSamples >= MIN_SAMPLES && customData.terrainElevProfile != nullptr) ? true : false;
bool customLandCover = (customData.numSamples >= MIN_SAMPLES && customData.mappedLandCoverProfile != nullptr) ? true : false;
bool customItuRCZ = (customData.numSamples >= MIN_SAMPLES && customData.ituRCZProfile != nullptr) ? true : false;
bool customSurf = (customData.numSamples >= MIN_SAMPLES && customData.surfaceElevProfile != nullptr) ? true : false;

	pLatLonProfile.clear();
	pDistKmProfile.clear();
	pTerrainElevProfile.clear();
	pMappedLandCoverProfile.clear();
	pRadioClimaticZoneProfile.clear();
	pSurfaceElevProfile.clear();

	if( customElev || customLandCover || customItuRCZ || customSurf )
	{
		sim.pTopoManager.GetLatLonProfileByNumPoints(sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, customData.numSamples,
		                                             TopographicDataManager::ITU_GREAT_CIRCLE, &pLatLonProfile, &pDistKmProfile);
	}
	else
	{
		sim.pTopoManager.GetLatLonProfileByRes(sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, sim.pTerrainElevDataSamplingResKm,
		                                       TopographicDataManager::ITU_GREAT_CIRCLE, &pLatLonProfile, &pDistKmProfile);
	}
	stats.numPoints = (int)pLatLonProfile.size();

	if( propagModel->IsUsingSurfaceElevData() )
	{
		// Priority on using custom data if present.
		// Do not apply pairing if either or both custom terrain profile and custom surface profile is present.
		if( customSurf )
		{
			pSurfaceElevProfile.resize(customData.numSamples);
			std::memcpy(pSurfaceElevProfile.data(), customData.surfaceElevProfile, customData.numSamples*sizeof(double));
		}
		if( customElev )
		{
			pTerrainElevProfile.resize(customData.numSamples);
			std::memcpy(pTerrainElevProfile.data(), customData.terrainElevProfile, customData.numSamples*sizeof(double));
		}

		if( pSurfaceElevProfile.size() == 0 && pTerrainElevProfile.size() > 0 )
			stats.surfaceElevMisses = sim.pTopoManager.GetSurfaceElevProfile(pLatLonProfile, &pSurfaceElevProfile);
		else if( pSurfaceElevProfile.size() > 0 && pTerrainElevProfile.size() == 0 )
			stats.terrainElevMisses = sim.pTopoManager.GetTerrainElevProfile(pLatLonProfile, &pTerrainElevProfile);
		else if( pSurfaceElevProfile.size() == 0 && pTerrainElevProfile.size() == 0 )
		{
			if( sim.pPairSurfAndTerrSources )
			{
				stats.surfaceElevMisses = sim.pTopoManager.GetPairedTerrainAndSurfaceElevProfiles(pLatLonProfile, &pTerrainElevProfile, &pSurfaceElevProfile);
				stats.terrainElevMisses = stats.surfaceElevMisses;
			}
			else
			{
				stats.terrainElevMisses = sim.pTopoManager.GetTerrainElevProfile(pLatLonProfile, &pTerrainElevProfile);
				stats.surfaceElevMisses = sim.pTopoManager.GetSurfaceElevProfile(pLatLonProfile, &pSurfaceElevProfile);
			}
		}
	}

	// always get a terrain elevation profile, needed for elev angle calculation, antenna gains
	if( pTerrainElevProfile.size() == 0 )
	{
		if( customElev )
		{
			pTerrainElevProfile.resize(customData.numSamples);
			std::memcpy(pTerrainElevProfile.data(), customData.terrainElevProfile, customData.numSamples*sizeof(double));
		}
		else
			stats.terrainElevMisses = sim.pTopoManager.GetTerrainElevProfile(pLatLonProfile, &pTerrainElevProfile);
	}

	if( propagModel->IsUsingMappedLandCoverData() )
	{
		if( customLandCover )
		{
			pMappedLandCoverProfile.resize(customData.numSamples);
			std::memcpy(pMappedLandCoverProfile.data(), customData.mappedLandCoverProfile, customData.numSamples*sizeof(int));
		}
		else
			stats.landCoverMisses = sim.pTopoManager.GetMappedLandCoverProfile(pLatLonProfile, propagModel->Id(),
																			   propagModel->DefaultMappedLandCoverValue(),
																			   &pMappedLandCoverProfile);
	}

	if( propagModel->IsUsingItuRadioClimZoneData() )
	{
		if( customItuRCZ )
		{
			pRadioClimaticZoneProfile.resize(customData.numSamples);
			std::memcpy(pRadioClimaticZoneProfile.data(), customData.ituRCZProfile, customData.numSamples*sizeof(ITURadioClimaticZone));
		}
		else
			stats.radioClimaticZoneMisses = sim.pTopoManager.GetRadioClimaticZoneProfile(pLatLonProfile, propagModel->Id(), &pRadioClimaticZoneProfile);
	}

	return stats;
}

double Generator::pAdditionalPathLosses(Simulation& sim, std::vector<double>* optionalOutputPathLossProfile)
{
bool applyP2108 = sim.pIturp2108Model.IsActive();
bool applyP2109 = sim.pIturp2109Model.IsActive();
bool applyP676 = sim.pIturp676Model.IsActive();
double f_GHz = sim.pTx.freqMHz/1000.0;
unsigned int firstIndex;
unsigned int lastIndex = pLatLonProfile.size()-1;
double rxLat_i, rxLon_i;
double txElevAngleDeg, rxElevAngleDeg;
double additionalLosses_dB;

	if( optionalOutputPathLossProfile != nullptr )
		firstIndex = 0;
	else
		firstIndex = lastIndex;

	for(unsigned int i=firstIndex ; i<=lastIndex ; i++)
	{
		additionalLosses_dB = 0;
		rxLat_i = pLatLonProfile[i].first;
		rxLon_i = pLatLonProfile[i].second;

		if( applyP2108 )
			additionalLosses_dB += sim.pIturp2108Model.CalcTerrestrialStatisticalLoss(f_GHz, pDistKmProfile[i]);

		if( applyP2109 )
		{
			pGetElevationAngles(sim.pTx.lat, sim.pTx.lon, rxLat_i, rxLon_i, sim.pTx.rcagl, sim.pRx.heightAGL,
								i+1, pDistKmProfile.data(), pTerrainElevProfile.data(), &txElevAngleDeg, &rxElevAngleDeg);
			additionalLosses_dB += sim.pIturp2109Model.CalcBuildingEntryLoss(f_GHz, rxElevAngleDeg);
		}

		if( applyP676 )
			additionalLosses_dB += sim.pIturp676Model.CalcGaseousAttenuation(f_GHz, sim.pTx.lat, sim.pTx.lon, rxLat_i, rxLon_i, sim.pTx.rcagl,
			                                                                 sim.pRx.heightAGL, i+1, pDistKmProfile.data(), pTerrainElevProfile.data());
		
		if( optionalOutputPathLossProfile != nullptr )
			(*optionalOutputPathLossProfile)[i] += additionalLosses_dB;
	}

	return additionalLosses_dB;
}

double Generator::pToSelectedResultType(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles,
                                        double* distKmProfile, double* terrainElevProfile, double pathLoss_dB)
{
	switch (sim.pResultType)
	{
	case FIELD_STRENGTH_DBUVM:
		return pToFieldStrength(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, pathLoss_dB);
	case PATH_LOSS_DB:
		return pathLoss_dB;
	case TRANSMISSION_LOSS_DB:
		return pToTransmissionLoss(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, pathLoss_dB);
	case RECEIVED_POWER_DBM:
		return pToReceivedPower(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, pathLoss_dB);
	default:
		return 0;
	}
}

double Generator::pToFieldStrength(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles,
                                   double* distKmProfile, double* terrainElevProfile, double pathLoss_dB)
{
double fs_dBuVm, fs_1kW_erp_dBuVm, txAntGain_dBi;
double freq_GHz = sim.pTx.freqMHz/1000.0;
double erp_dBkW = sim.pTx.erp(Transmitter::PowerUnit::DBW) - 30;

	fs_1kW_erp_dBuVm = ITURP_1812::FieldStrength(freq_GHz, pathLoss_dB);

	// get tx anntenna gain (note: rx antenna gain is never included in field strength)
	pGetAntennaGains(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, &txAntGain_dBi, nullptr);

	// substract max gain since it is already included in the ERP/EIRP
	// do not add tx losses since they are already included in the ERP/EIRP
	fs_dBuVm = fs_1kW_erp_dBuVm + erp_dBkW + txAntGain_dBi - sim.pTx.maxGain_dBi;

	// alternately, could be calculated this way...
	// fs_dBuVm = fs_1kW_erp_dBuVm + sim.pTx.tpo(Transmitter::PowerUnit::DBM) - 60 + txAntGain_dBi - sim.pTx.losses_dB - 2.15;

	return fs_dBuVm;
}

double Generator::pToTransmissionLoss(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles,
                                      double* distKmProfile, double* terrainElevProfile, double pathLoss_dB)
{
double tl_dB, txAntGain_dBi, rxAntGain_dBi;

	pGetAntennaGains(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, &txAntGain_dBi, &rxAntGain_dBi);
	tl_dB = pathLoss_dB - txAntGain_dBi + sim.pTx.losses_dB - rxAntGain_dBi + sim.pRx.losses_dB;
	return tl_dB;
}

// see https://en.wikipedia.org/wiki/Link_budget
double Generator::pToReceivedPower(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles,
                                   double* distKmProfile, double* terrainElevProfile, double pathLoss_dB)
{
double tpo_dBm = sim.pTx.tpo(Transmitter::PowerUnit::DBM);
double rp_dbuVm, txAntGain_dBi, rxAntGain_dBi;

	pGetAntennaGains(sim, rxLat, rxLon, sizeProfiles, distKmProfile, terrainElevProfile, &txAntGain_dBi, &rxAntGain_dBi);
	rp_dbuVm = tpo_dBm - pathLoss_dB + txAntGain_dBi - sim.pTx.losses_dB + rxAntGain_dBi - sim.pRx.losses_dB;
	return rp_dbuVm;
}

void Generator::pGetAntennaGains(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile,
                                 double* terrainElevProfile, double* txGain_dBi, double* rxGain_dBi)
{
const GeographicLib::Geodesic& geod = GeographicLib::Geodesic::WGS84();
double txElevAngleDeg, rxElevAngleDeg;
double txToRxAzmDeg, rxToTxAzmDeg, tempAzm;
double azmDeg;

	pGetElevationAngles(sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, sim.pTx.rcagl, sim.pRx.heightAGL, sizeProfiles, distKmProfile,
	                    terrainElevProfile, &txElevAngleDeg, &rxElevAngleDeg);

	if( txGain_dBi != nullptr )
	{
		if( sim.pTx.bearingRef == OTHER_TERMINAL )
			azmDeg = sim.pTx.bearingDeg;
		else // sim.pTx.bearingRef == TRUE_NORTH
		{
			geod.Inverse(sim.pTx.lat, sim.pTx.lon, rxLat, rxLon, txToRxAzmDeg, tempAzm);
			azmDeg = txToRxAzmDeg - sim.pTx.bearingDeg;
		}
		*txGain_dBi = sim.pTx.antPattern.Gain(azmDeg, txElevAngleDeg, (AntennaPattern::INTERPOLATION_ALGORITHM)sim.pTx.patternApproxMethod, true);
		*txGain_dBi += sim.pTx.maxGain_dBi;
	}

	if( rxGain_dBi != nullptr )
	{
		if( sim.pRx.bearingRef == OTHER_TERMINAL )
			azmDeg = sim.pRx.bearingDeg;
		else // sim.pRx.bearingRef == TRUE_NORTH
		{
			geod.Inverse(rxLat, rxLon, sim.pTx.lat, sim.pTx.lon, rxToTxAzmDeg, tempAzm);
			azmDeg = rxToTxAzmDeg - sim.pRx.bearingDeg;
		}
		
		*rxGain_dBi = sim.pRx.antPattern.Gain(azmDeg, rxElevAngleDeg, (AntennaPattern::INTERPOLATION_ALGORITHM)sim.pRx.patternApproxMethod, true);
		*rxGain_dBi += sim.pRx.maxGain_dBi;
	}
}

// Calculates elevation angles for getting tx and rx antenna pattern gains.
// This corresponds to the terminal-to-terminal angle in line-of-sight situations, or the terrain clearance angle in non-LOS situations.
// For output values (txElevAngleDeg, rxElevAngleDeg): -90 degrees = towards sky, +90 degrees = towards ground, 0 = parallel to ground.
// Algorithm derived from:
//   ITU-R P.1812-6, Attachment 1 to Annex 1, Section 4 & 5.3
//   ITU-R P.452-17, Attachment 2 to Annex 1, Section 4 & 5.1.3
void Generator::pGetElevationAngles(double txLat, double txLon, double rxLat, double rxLon, double txHeightAGL, double rxHeightAGL,
                                    unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile,
                                    double* txElevAngleDeg, double* rxElevAngleDeg)
{
	if( sizeProfiles < 2 )
	{
		*txElevAngleDeg = *rxElevAngleDeg = 0;
		return;
	}
	if( distKmProfile[sizeProfiles-1] < 1E-5 )
	{
		*txElevAngleDeg = *rxElevAngleDeg = 0;
		return;
	}

double pathCentreLat, pathCentreLon;
unsigned int n = sizeProfiles;
double* d = distKmProfile;
double* h = terrainElevProfile;
double dn = d[n-1]; // total path distance in km
double txHeight_mamsl = h[0] + txHeightAGL; // tx height in meters above mean sea level
double rxHeight_mamsl = h[n-1] + rxHeightAGL; // rx height in meters above mean sea level
double txMaxAngleRad, rxMaxAngleRad;
double terminalToTerrainAngleRad;
double deltaN; // average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere (N-units/km)
double ae; // median effective Earth radius (km)

	ITURP_2001::GreatCircleIntermediatePoint(txLat, txLon, rxLat, rxLon, dn/2.0, pathCentreLat, pathCentreLon);
	deltaN = ITURP_DigitalMaps::DN50(pathCentreLat, pathCentreLon);
	ae = 157.0*ITURP_2001::Re / (157.0-deltaN);

	auto ElevationAngleRad = [ae] (double hamsl_from, double hamsl_to, double distKm) -> double
	{
		return atan(((hamsl_to-hamsl_from)/(1000.0*distKm))-(distKm/(2.0*ae))); // elev angle in radians
	};

	txMaxAngleRad = ElevationAngleRad(txHeight_mamsl, rxHeight_mamsl, dn); // elevation angle from the transmit to the receive antenna (radians)
	rxMaxAngleRad =  ElevationAngleRad(rxHeight_mamsl, txHeight_mamsl, dn); // elevation angle from the receive to the transmit antenna (radians)
	for(unsigned int i=1 ; i<n-1 ; i++) // excludes transmit and receive locations
	{
		terminalToTerrainAngleRad = ElevationAngleRad(txHeight_mamsl, h[i], d[i]);
		txMaxAngleRad = std::max(terminalToTerrainAngleRad, txMaxAngleRad);
		terminalToTerrainAngleRad = ElevationAngleRad(rxHeight_mamsl, h[i], dn-d[i]);
		rxMaxAngleRad = std::max(terminalToTerrainAngleRad, rxMaxAngleRad);
	}

	*txElevAngleDeg = -txMaxAngleRad/ITURP_2001::PI_ON_180;
	*rxElevAngleDeg = -rxMaxAngleRad/ITURP_2001::PI_ON_180;
}

int Generator::pGetStatus(MissesStats stats)
{
int status = STATUS_OK;

	if( stats.terrainElevMisses > 0 )
	{
		status += STATUS_SOME_TERRAIN_ELEV_DATA_MISSING;
		if( stats.terrainElevMisses == stats.numPoints )
			status += STATUS_NO_TERRAIN_ELEV_DATA;
	}

	if( stats.landCoverMisses > 0 )
	{
		status += STATUS_SOME_LAND_COVER_DATA_MISSING;
		if( stats.landCoverMisses == stats.numPoints )
			status += STATUS_NO_LAND_COVER_DATA;
	}

	if( stats.radioClimaticZoneMisses > 0 )
	{
		status += STATUS_SOME_ITU_RCZ_DATA_MISSING;
		if( stats.radioClimaticZoneMisses == stats.numPoints )
			status += STATUS_NO_ITU_RCZ_DATA;
	}

	if( stats.surfaceElevMisses > 0 )
	{
		status += STATUS_SOME_SURFACE_ELEV_DATA_MISSING;
		if( stats.surfaceElevMisses == stats.numPoints )
			status += STATUS_NO_SURFACE_ELEV_DATA;
	}

	return status;
}

const char* Generator::pGetResultUnitString(ResultType resultType)
{
	switch (resultType)
	{
	case FIELD_STRENGTH_DBUVM:
		return "dBuV/m";
	case PATH_LOSS_DB:
	case TRANSMISSION_LOSS_DB:
		return "dB";
	case RECEIVED_POWER_DBM:
		return "dBm";
	default:
		return "";
	}
}

const char* Generator::pGetResultNameString(ResultType resultType)
{
	switch (resultType)
	{
	case FIELD_STRENGTH_DBUVM:
		return "field strength";
	case PATH_LOSS_DB:
		return "path loss";
	case TRANSMISSION_LOSS_DB:
		return "transmission loss";
	case RECEIVED_POWER_DBM:
		return "received power";
	default:
		return "";
	}
}

const char* Generator::pGetPropagModelShortName(Crc::Covlib::PropagationModel propagModel)
{
	switch (propagModel)
	{
	case LONGLEY_RICE:
		return "L-R";
	case ITU_R_P_1812:
		return "P1812";
	case ITU_R_P_452_V17:
		return "P452-17";
	case ITU_R_P_452_V18:
		return "P452-18";
	case FREE_SPACE:
		return "FreeSpace";
	case EXTENDED_HATA:
		return "eHata";
	default:
		return "";
	}
}

PropagModel* Generator::pGetPropagModelPtr(Simulation& sim)
{
	switch (sim.pPropagModelId)
	{
	case LONGLEY_RICE:
		return (PropagModel*) &(sim.pLongleyRiceModel);
	case ITU_R_P_1812:
		return (PropagModel*) &(sim.pIturp1812Model);
	case ITU_R_P_452_V17:
		return (PropagModel*) &(sim.pIturp452v17Model);
	case ITU_R_P_452_V18:
		return (PropagModel*) &(sim.pIturp452v18Model);
	case FREE_SPACE:
		return (PropagModel*) &(sim.pFreeSpaceModel);
	case EXTENDED_HATA:
		return (PropagModel*) &(sim.pEHataModel);
	default:
		return nullptr;
	}
}