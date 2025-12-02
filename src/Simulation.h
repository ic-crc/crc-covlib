/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "CRC-COVLIB.h"
#include "Generator.h"
#include "GeoDataGrid.h"
#include "ContourFillsEngine.h"
#include "TopographicDataManager.h"
#include "GeoTIFFTerrainElevSource.h"
#include "GeoTIFFLandCoverSource.h"
#include "GeoTIFFSurfaceElevSource.h"
#include "CustomTerrainElevSource.h"
#include "CustomLandCoverSource.h"
#include "CustomSurfaceElevSource.h"
#include "SRTMTerrainElevSource.h"
#include "SRTMSurfaceElevSource.h"
#include "LongleyRicePropagModel.h"
#include "ITURP1812PropagModel.h"
#include "ITURP452v17PropagModel.h"
#include "ITURP452v18PropagModel.h"
#include "ITURP2108ClutterLossModel.h"
#include "CommTerminal.h"
#include "ITURP2109BldgEntryLossModel.h"
#include "FreeSpacePropagModel.h"
#include "ITURP676GaseousAttenuationModel.h"
#include "EHataPropagModel.h"
#include "CRCMLPLPropagModel.h"
#include "CRCPathObscuraPropagModel.h"


class Simulation : public Crc::Covlib::ISimulation
{
friend class Generator;
public:
	Simulation(void);
	Simulation(const Simulation& original);
	virtual ~Simulation(void);

	const Simulation& operator=(const Simulation& original);

	virtual void Release();

	// Transmitter parameters
	virtual void SetTransmitterLocation(double latitude_degrees, double longitude_degrees);
	virtual double GetTransmitterLatitude() const;
	virtual double GetTransmitterLongitude() const;
	virtual void SetTransmitterHeight(double height_meters);
	virtual double GetTransmitterHeight() const;
	virtual void SetTransmitterFrequency(double frequency_MHz);
	virtual double GetTransmitterFrequency() const;
	virtual void SetTransmitterPower(double power_watts, Crc::Covlib::PowerType powerType=Crc::Covlib::EIRP);
	virtual double GetTransmitterPower(Crc::Covlib::PowerType powerType=Crc::Covlib::EIRP) const;
	virtual void SetTransmitterLosses(double losses_dB);
	virtual double GetTransmitterLosses() const;
	virtual void SetTransmitterPolarization(Crc::Covlib::Polarization polarization);
	virtual Crc::Covlib::Polarization GetTransmitterPolarization() const;

	// Receiver parameters
	virtual void SetReceiverHeightAboveGround(double height_meters);
	virtual double GetReceiverHeightAboveGround() const;
	virtual void SetReceiverLosses(double losses_dB);
	virtual double GetReceiverLosses() const;

	// Antenna parameters
	virtual void ClearAntennaPatterns(Crc::Covlib::Terminal terminal, bool clearHorizontalPattern=true, bool clearVerticalPattern=true);
	virtual void AddAntennaHorizontalPatternEntry(Crc::Covlib::Terminal terminal, double azimuth_degrees, double gain_dB);
	virtual void AddAntennaVerticalPatternEntry(Crc::Covlib::Terminal terminal, int azimuth_degrees, double elevAngle_degrees, double gain_dB);
	virtual void SetAntennaElectricalTilt(Crc::Covlib::Terminal terminal, double elecricalTilt_degrees);
	virtual double GetAntennaElectricalTilt(Crc::Covlib::Terminal terminal) const;
	virtual void SetAntennaMechanicalTilt(Crc::Covlib::Terminal terminal, double mechanicalTilt_degrees, double azimuth_degrees=0);
	virtual double GetAntennaMechanicalTilt(Crc::Covlib::Terminal terminal) const;
	virtual double GetAntennaMechanicalTiltAzimuth(Crc::Covlib::Terminal terminal) const;
	virtual void SetAntennaMaximumGain(Crc::Covlib::Terminal terminal, double maxGain_dBi);
	virtual double GetAntennaMaximumGain(Crc::Covlib::Terminal terminal) const;
	virtual void SetAntennaBearing(Crc::Covlib::Terminal terminal, Crc::Covlib::BearingReference bearingRef, double bearing_degrees);
	virtual Crc::Covlib::BearingReference GetAntennaBearingReference(Crc::Covlib::Terminal terminal) const;
	virtual double GetAntennaBearing(Crc::Covlib::Terminal terminal) const;
	virtual double NormalizeAntennaHorizontalPattern(Crc::Covlib::Terminal terminal);
	virtual double NormalizeAntennaVerticalPattern(Crc::Covlib::Terminal terminal);
	virtual void SetAntennaPatternApproximationMethod(Crc::Covlib::Terminal terminal, Crc::Covlib::PatternApproximationMethod method);
	virtual Crc::Covlib::PatternApproximationMethod GetAntennaPatternApproximationMethod(Crc::Covlib::Terminal terminal) const;
	virtual double GetAntennaGain(Crc::Covlib::Terminal terminal, double azimuth_degrees, double elevAngle_degrees, double receiverLatitude_degrees=0, double receiverLongitude_degrees=0) const;

	// Propagation model selection
	virtual void SetPropagationModel(Crc::Covlib::PropagationModel propagationModel);
	virtual Crc::Covlib::PropagationModel GetPropagationModel() const;

	// Longley-Rice propagation model parameters
	virtual void SetLongleyRiceSurfaceRefractivity(double refractivity_NUnits);
	virtual double GetLongleyRiceSurfaceRefractivity() const;
	virtual void SetLongleyRiceGroundDielectricConst(double dielectricConst);
	virtual double GetLongleyRiceGroundDielectricConst() const;
	virtual void SetLongleyRiceGroundConductivity(double groundConduct_Sm);
	virtual double GetLongleyRiceGroundConductivity() const;
	virtual void SetLongleyRiceClimaticZone(Crc::Covlib::LRClimaticZone climaticZone);
	virtual Crc::Covlib::LRClimaticZone GetLongleyRiceClimaticZone() const;
	virtual void SetLongleyRiceActivePercentageSet(Crc::Covlib::LRPercentageSet percentageSet);
	virtual Crc::Covlib::LRPercentageSet GetLongleyRiceActivePercentageSet() const;
	virtual void SetLongleyRiceTimePercentage(double time_percent);
	virtual double GetLongleyRiceTimePercentage() const;
	virtual void SetLongleyRiceLocationPercentage(double location_percent);
	virtual double GetLongleyRiceLocationPercentage() const;
	virtual void SetLongleyRiceSituationPercentage(double situation_percent);
	virtual double GetLongleyRiceSituationPercentage() const;
	virtual void SetLongleyRiceConfidencePercentage(double confidence_percent);
	virtual double GetLongleyRiceConfidencePercentage() const;
	virtual void SetLongleyRiceReliabilityPercentage(double reliability_percent);
	virtual double GetLongleyRiceReliabilityPercentage() const;
	virtual void SetLongleyRiceModeOfVariability(int mode);
	virtual int GetLongleyRiceModeOfVariability() const;

	// ITU-R P.1812 propagation model parameters
	virtual void SetITURP1812TimePercentage(double time_percent);
	virtual double GetITURP1812TimePercentage() const;
	virtual void SetITURP1812LocationPercentage(double location_percent);
	virtual double GetITURP1812LocationPercentage() const;
	virtual void SetITURP1812AverageRadioRefractivityLapseRate(double deltaN_Nunitskm);
	virtual double GetITURP1812AverageRadioRefractivityLapseRate() const;
	virtual void SetITURP1812SeaLevelSurfaceRefractivity(double N0_Nunits);
	virtual double GetITURP1812SeaLevelSurfaceRefractivity() const;
	virtual void SetITURP1812PredictionResolution(double resolution_meters);
	virtual double GetITURP1812PredictionResolution() const;
	virtual void SetITURP1812RepresentativeClutterHeight(Crc::Covlib::P1812ClutterCategory clutterCategory, double reprHeight_meters);
	virtual double GetITURP1812RepresentativeClutterHeight(Crc::Covlib::P1812ClutterCategory clutterCategory) const;
	virtual void SetITURP1812RadioClimaticZonesFile(const char* pathname);
	virtual const char* GetITURP1812RadioClimaticZonesFile() const;
	virtual void SetITURP1812LandCoverMappingType(Crc::Covlib::P1812LandCoverMappingType mappingType);
	virtual Crc::Covlib::P1812LandCoverMappingType GetITURP1812LandCoverMappingType() const;
	virtual void SetITURP1812SurfaceProfileMethod(Crc::Covlib::P1812SurfaceProfileMethod method);
	virtual Crc::Covlib::P1812SurfaceProfileMethod GetITURP1812SurfaceProfileMethod() const;

	// ITU-R P.452 propagation model parameters
	virtual void SetITURP452TimePercentage(double time_percent);
	virtual double GetITURP452TimePercentage() const;
    virtual void SetITURP452PredictionType(Crc::Covlib::P452PredictionType predictionType);
    virtual Crc::Covlib::P452PredictionType GetITURP452PredictionType() const;
	virtual void SetITURP452AverageRadioRefractivityLapseRate(double deltaN_Nunitskm);
	virtual double GetITURP452AverageRadioRefractivityLapseRate() const;
	virtual void SetITURP452SeaLevelSurfaceRefractivity(double N0_Nunits);
	virtual double GetITURP452SeaLevelSurfaceRefractivity() const;
	virtual void SetITURP452AirTemperature(double temperature_C);
	virtual double GetITURP452AirTemperature() const;
	virtual void SetITURP452AirPressure(double pressure_hPa);
	virtual double GetITURP452AirPressure() const;
	virtual void SetITURP452RadioClimaticZonesFile(const char* pathname);
	virtual const char* GetITURP452RadioClimaticZonesFile() const;
	// specific to P.452-17 version
	virtual void SetITURP452HeightGainModelClutterValue(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory, Crc::Covlib::P452HeightGainModelClutterParam nominalParam, double nominalValue);
	virtual double GetITURP452HeightGainModelClutterValue(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory, Crc::Covlib::P452HeightGainModelClutterParam nominalParam) const;
	virtual void SetITURP452HeightGainModelMode(Crc::Covlib::Terminal terminal, Crc::Covlib::P452HeightGainModelMode mode);
	virtual Crc::Covlib::P452HeightGainModelMode GetITURP452HeightGainModelMode(Crc::Covlib::Terminal terminal) const;
	// specific to P.452-18 version
	virtual void SetITURP452RepresentativeClutterHeight(Crc::Covlib::P452ClutterCategory clutterCategory, double reprHeight_meters);
	virtual double GetITURP452RepresentativeClutterHeight(Crc::Covlib::P452ClutterCategory clutterCategory) const;
	virtual void SetITURP452LandCoverMappingType(Crc::Covlib::P452LandCoverMappingType mappingType);
	virtual Crc::Covlib::P452LandCoverMappingType GetITURP452LandCoverMappingType() const;
	virtual void SetITURP452SurfaceProfileMethod(Crc::Covlib::P452SurfaceProfileMethod method);
	virtual Crc::Covlib::P452SurfaceProfileMethod GetITURP452SurfaceProfileMethod() const;

	// Extended Hata propagation model parameters
	virtual void SetEHataClutterEnvironment(Crc::Covlib::EHataClutterEnvironment clutterEnvironment);
	virtual Crc::Covlib::EHataClutterEnvironment GetEHataClutterEnvironment() const;
	virtual void SetEHataReliabilityPercentage(double percent);
	virtual double GetEHataReliabilityPercentage() const;

	// ITU-R P.2108 statistical clutter loss model for terrestrial paths
	virtual void SetITURP2108TerrestrialStatModelActiveState(bool active);
	virtual bool GetITURP2108TerrestrialStatModelActiveState() const;
	virtual void SetITURP2108TerrestrialStatModelLocationPercentage(double location_percent);
	virtual double GetITURP2108TerrestrialStatModelLocationPercentage() const;
	virtual double GetITURP2108TerrestrialStatModelLoss(double frequency_GHz, double distance_km) const;

	// ITU-R P.2109 building entry loss model
	virtual void SetITURP2109ActiveState(bool active);
	virtual bool GetITURP2109ActiveState() const;
	virtual void SetITURP2109Probability(double probability_percent);
	virtual double GetITURP2109Probability() const;
	virtual void SetITURP2109DefaultBuildingType(Crc::Covlib::P2109BuildingType buildingType);
	virtual Crc::Covlib::P2109BuildingType GetITURP2109DefaultBuildingType() const;
	virtual double GetITURP2109BuildingEntryLoss(double frequency_GHz, double elevAngle_degrees) const;

	// ITU-R P.676 gaseous attenuation model for terrestrial paths
	virtual void SetITURP676TerrPathGaseousAttenuationActiveState(bool active, double atmPressure_hPa=Crc::Covlib::AUTOMATIC, double temperature_C=Crc::Covlib::AUTOMATIC, double waterVapourDensity_gm3=Crc::Covlib::AUTOMATIC);
	virtual bool GetITURP676TerrPathGaseousAttenuationActiveState() const;
	virtual double GetITURP676GaseousAttenuation(double frequency_GHz, double atmPressure_hPa=1013.25, double temperature_C=15, double waterVapourDensity_gm3=7.5) const;

	// ITU digial maps
	virtual double GetITUDigitalMapValue(Crc::Covlib::ITUDigitalMap map, double latitude_degrees, double longitude_degrees) const;

	// Terrain elevation data parameters
	virtual void SetPrimaryTerrainElevDataSource(Crc::Covlib::TerrainElevDataSource terrainElevSource);
	virtual Crc::Covlib::TerrainElevDataSource GetPrimaryTerrainElevDataSource() const;
	virtual void SetSecondaryTerrainElevDataSource(Crc::Covlib::TerrainElevDataSource terrainElevSource);
	virtual Crc::Covlib::TerrainElevDataSource GetSecondaryTerrainElevDataSource() const;
	virtual void SetTertiaryTerrainElevDataSource(Crc::Covlib::TerrainElevDataSource terrainElevSource);
	virtual Crc::Covlib::TerrainElevDataSource GetTertiaryTerrainElevDataSource() const;
	virtual void SetTerrainElevDataSourceDirectory(Crc::Covlib::TerrainElevDataSource terrainElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
	virtual const char* GetTerrainElevDataSourceDirectory(Crc::Covlib::TerrainElevDataSource terrainElevSource) const;
	virtual bool AddCustomTerrainElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* terrainElevData_meters, bool defineNoDataValue=false, float noDataValue=0);
	virtual void ClearCustomTerrainElevData();
	virtual void SetTerrainElevDataSourceSamplingMethod(Crc::Covlib::TerrainElevDataSource terrainElevSource, Crc::Covlib::SamplingMethod samplingMethod);
	virtual Crc::Covlib::SamplingMethod GetTerrainElevDataSourceSamplingMethod(Crc::Covlib::TerrainElevDataSource terrainElevSource) const;
	virtual void SetTerrainElevDataSamplingResolution(double samplingResolution_meters);
	virtual double GetTerrainElevDataSamplingResolution() const;
	virtual double GetTerrainElevation(double latitude_degrees, double longitude_degrees, double noDataValue=0);
	virtual int GetTerrainElevationProfile(double latitude_degrees, double longitude_degrees, double* outputProfile, int sizeOutputProfile);

	// Land cover data parameters
	virtual void SetPrimaryLandCoverDataSource(Crc::Covlib::LandCoverDataSource landCoverSource);
	virtual Crc::Covlib::LandCoverDataSource GetPrimaryLandCoverDataSource() const;
	virtual void SetSecondaryLandCoverDataSource(Crc::Covlib::LandCoverDataSource landCoverSource);
	virtual Crc::Covlib::LandCoverDataSource GetSecondaryLandCoverDataSource() const;
	virtual void SetLandCoverDataSourceDirectory(Crc::Covlib::LandCoverDataSource landCoverSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
	virtual const char* GetLandCoverDataSourceDirectory(Crc::Covlib::LandCoverDataSource landCoverSource) const;
	virtual bool AddCustomLandCoverData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const short* landCoverData, bool defineNoDataValue=false, short noDataValue=0);
	virtual void ClearCustomLandCoverData();
	virtual int GetLandCoverClass(double latitude_degrees, double longitude_degrees);
	virtual int GetLandCoverClassProfile(double latitude_degrees, double longitude_degrees, int* outputProfile, int sizeOutputProfile);
	virtual int GetLandCoverClassMappedValue(double latitude_degrees, double longitude_degrees, Crc::Covlib::PropagationModel propagationModel);
	virtual int GetLandCoverClassMappedValueProfile(double latitude_degrees, double longitude_degrees, Crc::Covlib::PropagationModel propagationModel, int* outputProfile, int sizeOutputProfile);
	virtual void SetLandCoverClassMapping(Crc::Covlib::LandCoverDataSource landCoverSource, int sourceClass, Crc::Covlib::PropagationModel propagationModel, int modelValue);
	virtual int GetLandCoverClassMapping(Crc::Covlib::LandCoverDataSource landCoverSource, int sourceClass, Crc::Covlib::PropagationModel propagationModel) const;
	virtual void SetDefaultLandCoverClassMapping(Crc::Covlib::LandCoverDataSource landCoverSource, Crc::Covlib::PropagationModel propagationModel, int modelValue);
	virtual int GetDefaultLandCoverClassMapping(Crc::Covlib::LandCoverDataSource landCoverSource, Crc::Covlib::PropagationModel propagationModel) const;
	virtual void ClearLandCoverClassMappings(Crc::Covlib::LandCoverDataSource landCoverSource, Crc::Covlib::PropagationModel propagationModel);

	// Surface elevation data parameters
	virtual void SetPrimarySurfaceElevDataSource(Crc::Covlib::SurfaceElevDataSource surfaceElevSource);
	virtual Crc::Covlib::SurfaceElevDataSource GetPrimarySurfaceElevDataSource() const;
	virtual void SetSecondarySurfaceElevDataSource(Crc::Covlib::SurfaceElevDataSource surfaceElevSource);
	virtual Crc::Covlib::SurfaceElevDataSource GetSecondarySurfaceElevDataSource() const;
	virtual void SetTertiarySurfaceElevDataSource(Crc::Covlib::SurfaceElevDataSource surfaceElevSource);
	virtual Crc::Covlib::SurfaceElevDataSource GetTertiarySurfaceElevDataSource() const;
	virtual void SetSurfaceElevDataSourceDirectory(Crc::Covlib::SurfaceElevDataSource surfaceElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
	virtual const char* GetSurfaceElevDataSourceDirectory(Crc::Covlib::SurfaceElevDataSource surfaceElevSource) const;
	virtual void SetSurfaceAndTerrainDataSourcePairing(bool usePairing);
	virtual bool GetSurfaceAndTerrainDataSourcePairing() const;
	virtual void SetSurfaceElevDataSourceSamplingMethod(Crc::Covlib::SurfaceElevDataSource surfaceElevSource, Crc::Covlib::SamplingMethod samplingMethod);
	virtual Crc::Covlib::SamplingMethod GetSurfaceElevDataSourceSamplingMethod(Crc::Covlib::SurfaceElevDataSource surfaceElevSource) const;
	virtual bool AddCustomSurfaceElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* surfaceElevData_meters, bool defineNoDataValue=false, float noDataValue=0);
	virtual void ClearCustomSurfaceElevData();
	virtual double GetSurfaceElevation(double latitude_degrees, double longitude_degrees, double noDataValue=0);
	virtual int GetSurfaceElevationProfile(double latitude_degrees, double longitude_degrees, double* outputProfile, int sizeOutputProfile);

	// Reception area parameters
	virtual void SetReceptionAreaCorners(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees);
	virtual double GetReceptionAreaLowerLeftCornerLatitude() const;
	virtual double GetReceptionAreaLowerLeftCornerLongitude() const;
	virtual double GetReceptionAreaUpperRightCornerLatitude() const;
	virtual double GetReceptionAreaUpperRightCornerLongitude() const;
	virtual void SetReceptionAreaNumHorizontalPoints(int numPoints);
	virtual int GetReceptionAreaNumHorizontalPoints() const;
	virtual void SetReceptionAreaNumVerticalPoints(int numPoints);
	virtual int GetReceptionAreaNumVerticalPoints() const;

	// Result type parameters
	virtual void SetResultType(Crc::Covlib::ResultType resultType);
	virtual Crc::Covlib::ResultType GetResultType() const;

	// Coverage display parameters for vector files (.mif and .kml)
	virtual void ClearCoverageDisplayFills();
	virtual void AddCoverageDisplayFill(double fromValue, double toValue, int rgbColor);
	virtual int GetCoverageDisplayNumFills() const;
	virtual double GetCoverageDisplayFillFromValue(int index) const;
	virtual double GetCoverageDisplayFillToValue(int index) const;
	virtual int GetCoverageDisplayFillColor(int index) const;

	// Generating and accessing results
	virtual double GenerateReceptionPointResult(double latitude_degrees, double longitude_degrees);
	virtual Crc::Covlib::ReceptionPointDetailedResult GenerateReceptionPointDetailedResult(double latitude_degrees, double longitude_degrees);
	virtual double GenerateProfileReceptionPointResult(double latitude_degrees, double longitude_degrees, int numSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile=nullptr, const double* surfaceElevProfile=nullptr, const Crc::Covlib::ITURadioClimaticZone* ituRadioClimaticZoneProfile=nullptr);
	virtual Crc::Covlib::ReceptionPointDetailedResult GenerateProfileReceptionPointDetailedResult(double latitude_degrees, double longitude_degrees, int numSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile=nullptr, const double* surfaceElevProfile=nullptr, const Crc::Covlib::ITURadioClimaticZone* ituRadioClimaticZoneProfile=nullptr);
	virtual void GenerateReceptionAreaResults();
	virtual int GetGenerateStatus() const;
	virtual double GetReceptionAreaResultValue(int xIndex, int yIndex) const;
	virtual void SetReceptionAreaResultValue(int xIndex, int yIndex, double value);
	virtual double GetReceptionAreaResultValueAtLatLon(double latitude_degrees, double longitude_degrees) const;
	virtual double GetReceptionAreaResultLatitude(int xIndex, int yIndex) const;
	virtual double GetReceptionAreaResultLongitude(int xIndex, int yIndex) const;
	virtual bool ExportReceptionAreaResultsToTextFile(const char* pathname, const char* resultsColumnName=nullptr) const;
	virtual bool ExportReceptionAreaResultsToMifFile(const char* pathname, const char* resultsUnits=nullptr) const;
	virtual bool ExportReceptionAreaResultsToKmlFile(const char* pathname, double fillOpacity_percent=50, double lineOpacity_percent=50, const char* resultsUnits=nullptr) const;
	virtual bool ExportReceptionAreaResultsToBilFile(const char* pathname) const;
	virtual bool ExportReceptionAreaTerrainElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false);
	virtual bool ExportReceptionAreaSurfaceElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false);
	virtual bool ExportReceptionAreaLandCoverClassesToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool mapValues);
	
	virtual bool ExportProfilesToCsvFile(const char* pathname, double latitude_degrees, double longitude_degrees);

private:
	bool pIsTerrainElevDataSourceValid(Crc::Covlib::TerrainElevDataSource terrainElevSource);
	bool pIsLandCoverDataSourceValid(Crc::Covlib::LandCoverDataSource landCoverSource);
	bool pIsSurfaceElevDataSourceValid(Crc::Covlib::SurfaceElevDataSource surfaceElevSource);
	void pUpdateTerrainManagerSourcePtrs();
	TerrainElevSource* pGetTerrainElevSourceObjPtr(Crc::Covlib::TerrainElevDataSource terrainElevSource);
	const TerrainElevSource* pGetTerrainElevSourceConstObjPtr(Crc::Covlib::TerrainElevDataSource terrainElevSource) const;
	LandCoverSource* pGetLandCoverSourceObjPtr(Crc::Covlib::LandCoverDataSource landCoverSource);
	const LandCoverSource* pGetLandCoverSourceConstObjPtr(Crc::Covlib::LandCoverDataSource landCoverSource) const;
	SurfaceElevSource* pGetSurfaceElevSourceObjPtr(Crc::Covlib::SurfaceElevDataSource surfaceElevSource);
	const SurfaceElevSource* pGetSurfaceElevSourceConstObjPtr(Crc::Covlib::SurfaceElevDataSource surfaceElevSource) const;
	void pSetDefaultEsaWorldcoverToP1812Mappings();
	void pSetDefaultEsaWorldcoverToP452v17Mappings();
	void pSetDefaultEsaWorldcoverToP452v18Mappings();
	void pSetDefaultNrcanLandCoverToP1812Mappings();
	void pSetDefaultNrcanLandCoverToP452v17Mappings();
	void pSetDefaultNrcanLandCoverToP452v18Mappings();
	const CommTerminal* pGetTerminalConstObjPtr(Crc::Covlib::Terminal terminal) const;
	CommTerminal* pGetTerminalObjPtr(Crc::Covlib::Terminal terminal);
	PropagModel* pGetPropagModelPtr(Crc::Covlib::PropagationModel propagModelId);

	// Note: when adding new members here, add them to the operator=() method as well

	Transmitter pTx;
	Receiver pRx;

	Crc::Covlib::PropagationModel pPropagModelId;
	LongleyRicePropagModel pLongleyRiceModel;
	ITURP1812PropagModel pIturp1812Model;
	ITURP452v17PropagModel pIturp452v17Model;
	ITURP452v18PropagModel pIturp452v18Model;
	FreeSpacePropagModel pFreeSpaceModel;
	EHataPropagModel pEHataModel;
	CRCMLPLPropagModel pCrcMlplModel;
	CRCPathObscuraPropagModel pCrcPathObscuraModel;

	ITURP2108ClutterLossModel pIturp2108Model;
	ITURP2109BldgEntryLossModel pIturp2109Model;
	ITURP676GaseousAttenuationModel pIturp676Model;

	Crc::Covlib::TerrainElevDataSource pPrimaryTerrainElevSourceId;
	Crc::Covlib::TerrainElevDataSource pSecondaryTerrainElevSourceId;
	Crc::Covlib::TerrainElevDataSource pTertiaryTerrainElevSourceId;
	double pTerrainElevDataSamplingResKm;
	SRTMTerrainElevSource pTerrainElevSrtm;
	CustomTerrainElevSource pTerrainElevCustom;
	GeoTIFFTerrainElevSource pTerrainElevCdem;
	GeoTIFFTerrainElevSource pTerrainElevHrdemDtm;
	GeoTIFFTerrainElevSource pTerrainElevGeotiff;
	GeoTIFFTerrainElevSource pTerrainElevMrdemDtm;

	Crc::Covlib::LandCoverDataSource pPrimaryLandCoverSourceId;
	Crc::Covlib::LandCoverDataSource pSecondaryLandCoverSourceId;
	GeoTIFFLandCoverSource pLandCoverEsaWorldcover;
	GeoTIFFLandCoverSource pLandCoverGeotiff;
	CustomLandCoverSource pLandCoverCustom;
	GeoTIFFLandCoverSource pLandCoverNrcan;

	Crc::Covlib::SurfaceElevDataSource pPrimarySurfElevSourceId;
	Crc::Covlib::SurfaceElevDataSource pSecondarySurfElevSourceId;
	Crc::Covlib::SurfaceElevDataSource pTertiarySurfElevSourceId;
	SRTMSurfaceElevSource pSurfElevSrtm;
	CustomSurfaceElevSource pSurfElevCustom;
	GeoTIFFSurfaceElevSource pSurfElevCdsm;
	GeoTIFFSurfaceElevSource pSurfElevHrdemDsm;
	GeoTIFFSurfaceElevSource pSurfElevGeotiff;
	GeoTIFFSurfaceElevSource pSurfElevMrdemDsm;

	Crc::Covlib::ResultType pResultType;

	std::vector<ContourFillsEngine::FillZone> pCoverageFills;

	GeoDataGrid<float> pRxAreaResults;
	int pGenerateStatus;
	Generator pGenerator;

	TopographicDataManager pTopoManager;
};
