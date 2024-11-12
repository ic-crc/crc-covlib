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
	                           const Crc::Covlib::ITURadioClimaticZone* customItuRadioClimaticZoneProfile=nullptr,
							   Crc::Covlib::ReceptionPointDetailedResult* detailedResult=nullptr);
	bool ExportProfilesToCsvFile(Simulation& sim, const char* pathname, double lat, double lon);

	double GetTransmitterAntennaGain(const Simulation& sim, double azmDeg, double elevAngleDeg, double rxLat, double rxLon) const;
	double GetReceiverAntennaGain(const Simulation& sim, double azmDeg, double elevAngleDeg, double rxLat, double rxLon) const;

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
	double pToSelectedResultType(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile, double pathLoss_dB, Crc::Covlib::ReceptionPointDetailedResult* detailedResult=nullptr);
	double pToFieldStrength(Simulation& sim, double pathLoss_dB, double txAntennaGain_dBi);
	double pToTransmissionLoss(Simulation& sim, double pathLoss_dB, double txAntGain_dBi, double rxAntGain_dBi);
	double pToReceivedPower(Simulation& sim, double pathLoss_dB, double txAntGain_dBi, double rxAntGain_dBi);
	double pGetTransmitterToReceiverAzimuth(const Simulation& sim, double rxLat, double rxLon) const;
	double pGetReceiverToTransmitterAzimuth(const Simulation& sim, double rxLat, double rxLon) const;
	double pGetAzimuth(double fromLat, double fromLon, double toLat, double toLon) const;
	double pGetTransmitterToReceiverElevAngle(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile);
	double pGetReceiverToTransmitterElevAngle(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile);
	double pElevationAngleRad(double hamsl_from, double hamsl_to, double distKm, double aeKm);
	double pGetTransmitterAntennaGain(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile);
	double pGetReceiverAntennaGain(Simulation& sim, double rxLat, double rxLon, unsigned int sizeProfiles, double* distKmProfile, double* terrainElevProfile);
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
