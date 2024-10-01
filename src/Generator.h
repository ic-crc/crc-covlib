#pragma once
#include "CRC-COVLIB.h"
#include "PropagModel.h"
#include <vector>

class Simulation;

class Generator
{
public:
	Generator(void);
	virtual ~Generator(void);

	void RunAreaCalculation(Simulation& sim);
	double RunPointCalculation(Simulation& sim, double lat, double lon, unsigned int numCustomSamples=0,
	                           const double* customTerrainElevProfile=nullptr,
	                           const int* customLandCoverMappedValueProfile=nullptr,
	                           const double* customSurfaceElevProfile=nullptr,
	                           const Crc::Covlib::ITURadioClimaticZone* customItuRadioClimaticZoneProfile=nullptr);
	bool ExportProfilesToCsvFile(Simulation& sim, const char* pathname, double lat, double lon);

private:
	struct MissesStats
	{
		int numPoints;
		int terrainElevMisses;
		int landCoverMisses;
		int radioClimaticZoneMisses;
		int surfaceElevMisses;
	};

	struct PathLossFuncOutput
	{
		double pathLoss;
		MissesStats stats;
	};

	struct CustomData
	{
		unsigned int numSamples;
		const double* terrainElevProfile;
		const int* mappedLandCoverProfile;
		const Crc::Covlib::ITURadioClimaticZone* ituRCZProfile;
		const double* surfaceElevProfile;
	};

	void pConditionalResourceRelease(Simulation& sim, unsigned int pointCountLimit);

	PathLossFuncOutput pPathLoss(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossLongleyRice(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossP1812(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossP452v17(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossP452v18(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossFreeSpace(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);
	PathLossFuncOutput pPathLossEHata(Simulation& sim, double rxLat, double rxLon, CustomData customData, std::vector<double>* optionalOutputPathLossProfile);

	MissesStats pFillProfiles(Simulation& sim, double rxLat, double rxLon, PropagModel* propagModel, CustomData customData);

	double pAdditionalPathLosses(Simulation& sim, std::vector<double>* optionalOutputPathLossProfile);
	double pToSelectedResultType(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double pathLoss_dB);
	double pToFieldStrength(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double pathLoss_dB);
	double pToTransmissionLoss(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double pathLoss_dB);
	double pToReceivedPower(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double pathLoss_dB);
	void pGetAntennaGains(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile,
	                      double* txGain_dBi, double* rxGain_dBi);
	void pGetElevationAngles(double txLat, double txLon, double rxLat, double rxLon, double txHeightAGL, double rxHeightAGL, unsigned int sizeProfiles,
	                         double* distKmProfile, double* terrainElevProfile, double* txElevAngleDeg, double* rxElevAngleDeg);
	int pGetStatus(MissesStats stats);
	const char* pGetResultUnitString(Crc::Covlib::ResultType resultType);
	const char* pGetResultNameString(Crc::Covlib::ResultType resultType);
	const char* pGetPropagModelShortName(Crc::Covlib::PropagationModel propagModel);
	PropagModel* pGetPropagModelPtr(Simulation& sim);

	std::vector<std::pair<double,double>> pLatLonProfile;
	std::vector<double> pDistKmProfile;
	std::vector<double> pTerrainElevProfile;
	std::vector<int> pMappedLandCoverProfile;
	std::vector<Crc::Covlib::ITURadioClimaticZone> pRadioClimaticZoneProfile;
	std::vector<double> pSurfaceElevProfile;

	unsigned int pRunPointCount;
};
