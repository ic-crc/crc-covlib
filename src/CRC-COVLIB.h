#pragma once
#include <cstddef>
#include <limits>

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the CRCCOVLIB_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// CRCCOVLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
	#ifdef CRCCOVLIB_EXPORTS
		#define CRCCOVLIB_API __declspec(dllexport)
	#else
		#define CRCCOVLIB_API __declspec(dllimport)
	#endif
#else
	#define CRCCOVLIB_API
	#define __stdcall
#endif


namespace Crc
{
	namespace Covlib
	{

		enum Polarization
		{
			HORIZONTAL_POL = 0,
			VERTICAL_POL   = 1
		};

		enum LRClimaticZone
		{
			LR_EQUATORIAL                   = 1,
			LR_CONTINENTAL_SUBTROPICAL      = 2,
			LR_MARITIME_SUBTROPICAL         = 3,
			LR_DESERT                       = 4,
			LR_CONTINENTAL_TEMPERATE        = 5,
			LR_MARITIME_TEMPERATE_OVER_LAND = 6,
			LR_MARITIME_TEMPERATE_OVER_SEA  = 7,
		};

		enum LRPercentageSet
		{
			LR_TIME_LOCATION_SITUATION = 1,
			LR_CONFIDENCE_RELIABILITY  = 2
		};

		enum LRModeOfVariability
		{
			LR_SINGLE_MESSAGE_MODE             =  0,
			LR_ACCIDENTAL_MESSAGE_MODE         =  1,
			LR_MOBILE_MODE                     =  2,
			LR_BROADCAST_MODE                  =  3,
			LR_ELIMINATE_LOCATION_VARIABILITY  = 10,
			LR_ELIMINATE_SITUATION_VARIABILITY = 20
		};

		enum P1812ClutterCategory
		{
			P1812_WATER_SEA           = 1,
			P1812_OPEN_RURAL          = 2,
			P1812_SUBURBAN            = 3,
			P1812_URBAN_TREES_FOREST  = 4,
			P1812_DENSE_URBAN         = 5
		};

		enum P1812LandCoverMappingType
		{
			P1812_MAP_TO_CLUTTER_CATEGORY    = 1,
			P1812_MAP_TO_REPR_CLUTTER_HEIGHT = 2
		};

		enum P1812SurfaceProfileMethod
		{
			P1812_ADD_REPR_CLUTTER_HEIGHT  = 1,
			P1812_USE_SURFACE_ELEV_DATA    = 2
		};

		enum P452PredictionType
		{
			P452_AVERAGE_YEAR = 1,
			P452_WORST_MONTH  = 2
		};

		enum P452HeightGainModelClutterCategory
		{
			// most numerical values from ITU-R P.1058-2
			P452_HGM_HIGH_CROP_FIELDS                    = 13,
			P452_HGM_PARK_LAND                           = 19,
			P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES     = 21,
			P452_HGM_ORCHARD_REGULARLY_SPACED            = 22,
			P452_HGM_SPARSE_HOUSES                       = 31,
			P452_HGM_VILLAGE_CENTRE                      = 32,
			P452_HGM_DECIDUOUS_TREES_IRREGULARLY_SPACED  = 23,
			P452_HGM_DECIDUOUS_TREES_REGULARLY_SPACED    = 24,
			P452_HGM_MIXED_TREE_FOREST                   = 27,
			P452_HGM_CONIFEROUS_TREES_IRREGULARLY_SPACED = 25,
			P452_HGM_CONIFEROUS_TREES_REGULARLY_SPACED   = 26,
			P452_HGM_TROPICAL_RAIN_FOREST                = 28,
			P452_HGM_SUBURBAN                            = 33,
			P452_HGM_DENSE_SUBURBAN                      = 34,
			P452_HGM_URBAN                               = 35,
			P452_HGM_DENSE_URBAN                         = 36,
			P452_HGM_HIGH_RISE_URBAN                     = 38, // this one was not found in P.1058
			P452_HGM_INDUSTRIAL_ZONE                     = 37,
			P452_HGM_OTHER                               = 90,
			P452_HGM_CUSTOM_AT_TRANSMITTER               = 200,
			P452_HGM_CUSTOM_AT_RECEIVER                  = 201
		};

		enum P452HeightGainModelClutterParam
		{
			P452_NOMINAL_HEIGHT_M    = 1,
			P452_NOMINAL_DISTANCE_KM = 2
		};

		enum P452HeightGainModelMode
		{
			P452_NO_SHIELDING            = 1,
			P452_USE_CUSTOM_AT_CATEGORY  = 2,
			P452_USE_CLUTTER_PROFILE     = 3,
			P452_USE_CLUTTER_AT_ENDPOINT = 4
		};

		enum P452ClutterCategory
		{
			P452_WATER_SEA           = 1,
			P452_OPEN_RURAL          = 2,
			P452_SUBURBAN            = 3,
			P452_URBAN_TREES_FOREST  = 4,
			P452_DENSE_URBAN         = 5
		};

		enum P452LandCoverMappingType
		{
			P452_MAP_TO_CLUTTER_CATEGORY    = 1,
			P452_MAP_TO_REPR_CLUTTER_HEIGHT = 2
		};

		enum P452SurfaceProfileMethod
		{
			P452_ADD_REPR_CLUTTER_HEIGHT               = 1,
			P452_EXPERIMENTAL_USE_OF_SURFACE_ELEV_DATA = 2
		};

		enum EHataClutterEnvironment
		{
			EHATA_URBAN    = 24,
			EHATA_SUBURBAN = 22,
			EHATA_RURAL    = 20
		};

		enum P2109BuildingType
		{
			P2109_TRADITIONAL         = 1,
			P2109_THERMALLY_EFFICIENT = 2
		};

		enum ResultType
		{
			FIELD_STRENGTH_DBUVM = 1,
			PATH_LOSS_DB         = 2,
			TRANSMISSION_LOSS_DB = 3,
			RECEIVED_POWER_DBM   = 4
		};

		enum TerrainElevDataSource
		{
			TERR_ELEV_NONE            = 0,
			TERR_ELEV_SRTM            = 1,
			TERR_ELEV_CUSTOM          = 2,
			TERR_ELEV_NRCAN_CDEM      = 3,
			TERR_ELEV_NRCAN_HRDEM_DTM = 4,
			TERR_ELEV_GEOTIFF         = 5,
			TERR_ELEV_NRCAN_MRDEM_DTM = 6
		};

		enum SurfaceElevDataSource
		{
			SURF_ELEV_NONE            = 100,
			SURF_ELEV_SRTM            = 101,
			SURF_ELEV_CUSTOM          = 102,
			SURF_ELEV_NRCAN_CDSM      = 103,
			SURF_ELEV_NRCAN_HRDEM_DSM = 104,
			SURF_ELEV_GEOTIFF         = 105,
			SURF_ELEV_NRCAN_MRDEM_DSM = 106
		};

		enum LandCoverDataSource
		{
			LAND_COVER_NONE           = 200,
			LAND_COVER_GEOTIFF        = 201,
			LAND_COVER_ESA_WORLDCOVER = 202,
			LAND_COVER_CUSTOM         = 203,
			LAND_COVER_NRCAN          = 204
		};
		
		enum GenerateStatus
		{
			STATUS_OK                               =   0,
			STATUS_NO_TERRAIN_ELEV_DATA             =   1,
			STATUS_SOME_TERRAIN_ELEV_DATA_MISSING   =   2,
			STATUS_NO_LAND_COVER_DATA               =   4,
			STATUS_SOME_LAND_COVER_DATA_MISSING     =   8,
			STATUS_NO_ITU_RCZ_DATA                  =  16,
			STATUS_SOME_ITU_RCZ_DATA_MISSING        =  32,
			STATUS_NO_SURFACE_ELEV_DATA             =  64,
			STATUS_SOME_SURFACE_ELEV_DATA_MISSING   = 128
		};

		enum SamplingMethod
		{
			NEAREST_NEIGHBOR       = 0,
			BILINEAR_INTERPOLATION = 1
		};

		enum PropagationModel
		{
			LONGLEY_RICE    = 0,
			ITU_R_P_1812    = 1,
			ITU_R_P_452_V17 = 2,
			ITU_R_P_452_V18 = 3,
			FREE_SPACE      = 4,
			EXTENDED_HATA   = 5,
			CRC_MLPL        = 6
		};

		inline constexpr double AUTOMATIC = std::numeric_limits<double>::quiet_NaN();

		enum Terminal
		{
			TRANSMITTER  = 1,
			RECEIVER     = 2
		};

		enum BearingReference
		{
			TRUE_NORTH     = 1,
			OTHER_TERMINAL = 2
		};

		enum PatternApproximationMethod
		{
			H_PATTERN_ONLY   = 1,
			V_PATTERN_ONLY   = 2,
			SUMMING          = 3,
			WEIGHTED_SUMMING = 4,
			HYBRID           = 5,
		};

		enum PowerType
		{
			TPO =  1, // Transmitter power output
			ERP =  2, // Effective radiated power
			EIRP = 3  // Effective isotropic radiated power
		};

		enum ITURadioClimaticZone
		{
			ITU_COASTAL_LAND = 3,
			ITU_INLAND       = 4,
			ITU_SEA          = 1
		};

		enum ITUDigitalMap
		{
			ITU_MAP_DN50      = 1,
			ITU_MAP_N050      = 2,
			ITU_MAP_T_ANNUAL  = 3,
			ITU_MAP_SURFWV_50 = 4
		};

		struct ReceptionPointDetailedResult
		{
			double result;
			double pathLoss_dB;
			double pathLength_km;
			double transmitterHeightAMSL_m;
			double receiverHeightAMSL_m;
			double transmitterAntennaGain_dBi;
			double receiverAntennaGain_dBi;
			double azimuthFromTransmitter_degrees;
			double azimuthFromReceiver_degrees;
			double elevAngleFromTransmitter_degrees;
			double elevAngleFromReceiver_degrees;
		};

		// COM-Like abstract interface. 
		// This interface doesn't require __declspec(dllexport/dllimport) specifier. 
		// Method calls are dispatched via virtual table. 
		// Any C++ compiler can use it. 
		// Instances are obtained via factory function. 
		struct ISimulation
		{
			virtual void Release() = 0;

			// Transmitter parameters
			virtual void SetTransmitterLocation(double latitude_degrees, double longitude_degrees) = 0;
			virtual double GetTransmitterLatitude() const = 0;
			virtual double GetTransmitterLongitude() const = 0;
			virtual void SetTransmitterHeight(double height_meters) = 0;
			virtual double GetTransmitterHeight() const = 0;
			virtual void SetTransmitterFrequency(double frequency_MHz) = 0;
			virtual double GetTransmitterFrequency() const = 0;
			virtual void SetTransmitterPower(double power_watts, PowerType powerType=EIRP) = 0;
			virtual double GetTransmitterPower(PowerType powerType=EIRP) const = 0;
			virtual void SetTransmitterLosses(double losses_dB) = 0;
			virtual double GetTransmitterLosses() const = 0;
			virtual void SetTransmitterPolarization(Polarization polarization) = 0;
			virtual Polarization GetTransmitterPolarization() const = 0;

			// Receiver parameters
			virtual void SetReceiverHeightAboveGround(double height_meters) = 0;
			virtual double GetReceiverHeightAboveGround() const = 0;
			virtual void SetReceiverLosses(double losses_dB) = 0;
			virtual double GetReceiverLosses() const = 0;

			// Antenna parameters
			virtual void ClearAntennaPatterns(Terminal terminal, bool clearHorizontalPattern=true, bool clearVerticalPattern=true) = 0;
			virtual void AddAntennaHorizontalPatternEntry(Terminal terminal, double azimuth_degrees, double gain_dB) = 0;
			virtual void AddAntennaVerticalPatternEntry(Terminal terminal, int azimuth_degrees, double elevAngle_degrees, double gain_dB) = 0;
			virtual void SetAntennaElectricalTilt(Terminal terminal, double elecricalTilt_degrees) = 0;
			virtual double GetAntennaElectricalTilt(Terminal terminal) const = 0;
			virtual void SetAntennaMechanicalTilt(Terminal terminal, double mechanicalTilt_degrees, double azimuth_degrees=0) = 0;
			virtual double GetAntennaMechanicalTilt(Terminal terminal) const = 0;
			virtual double GetAntennaMechanicalTiltAzimuth(Terminal terminal) const = 0;
			virtual void SetAntennaMaximumGain(Terminal terminal, double maxGain_dBi) = 0;
			virtual double GetAntennaMaximumGain(Terminal terminal) const = 0;
			virtual void SetAntennaBearing(Terminal terminal, BearingReference bearingRef, double bearing_degrees) = 0;
			virtual BearingReference GetAntennaBearingReference(Terminal terminal) const = 0;
			virtual double GetAntennaBearing(Terminal terminal) const = 0;
			virtual double NormalizeAntennaHorizontalPattern(Terminal terminal) = 0;
			virtual double NormalizeAntennaVerticalPattern(Terminal terminal) = 0;
			virtual void SetAntennaPatternApproximationMethod(Terminal terminal, PatternApproximationMethod method) = 0;
			virtual PatternApproximationMethod GetAntennaPatternApproximationMethod(Terminal terminal) const = 0;
			virtual double GetAntennaGain(Terminal terminal, double azimuth_degrees, double elevAngle_degrees, double receiverLatitude_degrees=0, double receiverLongitude_degrees=0) const = 0;

			// Propagation model selection
			virtual void SetPropagationModel(PropagationModel propagationModel) = 0;
			virtual PropagationModel GetPropagationModel() const = 0;

			// Longley-Rice propagation model parameters
			virtual void SetLongleyRiceSurfaceRefractivity(double refractivity_NUnits) = 0;
			virtual double GetLongleyRiceSurfaceRefractivity() const = 0;
			virtual void SetLongleyRiceGroundDielectricConst(double dielectricConst) = 0;
			virtual double GetLongleyRiceGroundDielectricConst() const = 0;
			virtual void SetLongleyRiceGroundConductivity(double groundConduct_Sm) = 0;
			virtual double GetLongleyRiceGroundConductivity() const = 0;
			virtual void SetLongleyRiceClimaticZone(LRClimaticZone climaticZone) = 0;
			virtual LRClimaticZone GetLongleyRiceClimaticZone() const = 0;
			virtual void SetLongleyRiceActivePercentageSet(LRPercentageSet percentageSet) = 0;
			virtual LRPercentageSet GetLongleyRiceActivePercentageSet() const = 0;
			virtual void SetLongleyRiceTimePercentage(double time_percent) = 0;
			virtual double GetLongleyRiceTimePercentage() const = 0;
			virtual void SetLongleyRiceLocationPercentage(double location_percent) = 0;
			virtual double GetLongleyRiceLocationPercentage() const = 0;
			virtual void SetLongleyRiceSituationPercentage(double situation_percent) = 0;
			virtual double GetLongleyRiceSituationPercentage() const = 0;
			virtual void SetLongleyRiceConfidencePercentage(double confidence_percent) = 0;
			virtual double GetLongleyRiceConfidencePercentage() const = 0;
			virtual void SetLongleyRiceReliabilityPercentage(double reliability_percent) = 0;
			virtual double GetLongleyRiceReliabilityPercentage() const = 0;
			virtual void SetLongleyRiceModeOfVariability(int mode) = 0;
			virtual int GetLongleyRiceModeOfVariability() const = 0;

			// ITU-R P.1812 propagation model parameters
			virtual void SetITURP1812TimePercentage(double time_percent) = 0;
			virtual double GetITURP1812TimePercentage() const = 0;
			virtual void SetITURP1812LocationPercentage(double location_percent) = 0;
			virtual double GetITURP1812LocationPercentage() const = 0;
			virtual void SetITURP1812AverageRadioRefractivityLapseRate(double deltaN_Nunitskm) = 0;
			virtual double GetITURP1812AverageRadioRefractivityLapseRate() const = 0;
			virtual void SetITURP1812SeaLevelSurfaceRefractivity(double N0_Nunits) = 0;
			virtual double GetITURP1812SeaLevelSurfaceRefractivity() const = 0;
			virtual void SetITURP1812PredictionResolution(double resolution_meters) = 0;
			virtual double GetITURP1812PredictionResolution() const = 0;
			virtual void SetITURP1812RepresentativeClutterHeight(P1812ClutterCategory clutterCategory, double reprHeight_meters) = 0;
			virtual double GetITURP1812RepresentativeClutterHeight(P1812ClutterCategory clutterCategory) const = 0;
			virtual void SetITURP1812RadioClimaticZonesFile(const char* pathname) = 0;
			virtual const char* GetITURP1812RadioClimaticZonesFile() const = 0;
			virtual void SetITURP1812LandCoverMappingType(P1812LandCoverMappingType mappingType) = 0;
			virtual P1812LandCoverMappingType GetITURP1812LandCoverMappingType() const = 0;
			virtual void SetITURP1812SurfaceProfileMethod(P1812SurfaceProfileMethod method) = 0;
			virtual P1812SurfaceProfileMethod GetITURP1812SurfaceProfileMethod() const = 0;

			// ITU-R P.452 propagation model parameters
			virtual void SetITURP452TimePercentage(double time_percent) = 0;
			virtual double GetITURP452TimePercentage() const = 0;
			virtual void SetITURP452PredictionType(P452PredictionType predictionType) = 0;
			virtual P452PredictionType GetITURP452PredictionType() const = 0;
			virtual void SetITURP452AverageRadioRefractivityLapseRate(double deltaN_Nunitskm) = 0;
			virtual double GetITURP452AverageRadioRefractivityLapseRate() const = 0;
			virtual void SetITURP452SeaLevelSurfaceRefractivity(double N0_Nunits) = 0;
			virtual double GetITURP452SeaLevelSurfaceRefractivity() const = 0;
			virtual void SetITURP452AirTemperature(double temperature_C) = 0;
			virtual double GetITURP452AirTemperature() const = 0;
			virtual void SetITURP452AirPressure(double pressure_hPa) = 0;
			virtual double GetITURP452AirPressure() const = 0;
			virtual void SetITURP452RadioClimaticZonesFile(const char* pathname) = 0;
			virtual const char* GetITURP452RadioClimaticZonesFile() const = 0;
			// specific to P.452-17 version
			virtual void SetITURP452HeightGainModelClutterValue(P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam, double nominalValue) = 0;
			virtual double GetITURP452HeightGainModelClutterValue(P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam) const = 0;
			virtual void SetITURP452HeightGainModelMode(Terminal terminal, P452HeightGainModelMode mode) = 0;
			virtual P452HeightGainModelMode GetITURP452HeightGainModelMode(Terminal terminal) const = 0;
			// specific to P.452-18 version
			virtual void SetITURP452RepresentativeClutterHeight(P452ClutterCategory clutterCategory, double reprHeight_meters) = 0;
			virtual double GetITURP452RepresentativeClutterHeight(P452ClutterCategory clutterCategory) const = 0;
			virtual void SetITURP452LandCoverMappingType(P452LandCoverMappingType mappingType) = 0;
			virtual P452LandCoverMappingType GetITURP452LandCoverMappingType() const = 0;
			virtual void SetITURP452SurfaceProfileMethod(P452SurfaceProfileMethod method) = 0;
			virtual P452SurfaceProfileMethod GetITURP452SurfaceProfileMethod() const = 0;

			// Extended Hata propagation model parameters
			virtual void SetEHataClutterEnvironment(EHataClutterEnvironment clutterEnvironment) = 0;
			virtual EHataClutterEnvironment GetEHataClutterEnvironment() const = 0;
			virtual void SetEHataReliabilityPercentage(double percent) = 0;
			virtual double GetEHataReliabilityPercentage() const = 0;

			// ITU-R P.2108 statistical clutter loss model for terrestrial paths
			virtual void SetITURP2108TerrestrialStatModelActiveState(bool active) = 0;
			virtual bool GetITURP2108TerrestrialStatModelActiveState() const = 0;
			virtual void SetITURP2108TerrestrialStatModelLocationPercentage(double location_percent) = 0;
			virtual double GetITURP2108TerrestrialStatModelLocationPercentage() const = 0;
			virtual double GetITURP2108TerrestrialStatModelLoss(double frequency_GHz, double distance_km) const = 0;

			// ITU-R P.2109 building entry loss model
			virtual void SetITURP2109ActiveState(bool active) = 0;
			virtual bool GetITURP2109ActiveState() const = 0;
			virtual void SetITURP2109Probability(double probability_percent) = 0;
			virtual double GetITURP2109Probability() const = 0;
			virtual void SetITURP2109DefaultBuildingType(P2109BuildingType buildingType) = 0;
			virtual P2109BuildingType GetITURP2109DefaultBuildingType() const = 0;
			virtual double GetITURP2109BuildingEntryLoss(double frequency_GHz, double elevAngle_degrees) const = 0;

			// ITU-R P.676 gaseous attenuation model for terrestrial paths
			virtual void SetITURP676TerrPathGaseousAttenuationActiveState(bool active, double atmPressure_hPa=AUTOMATIC, double temperature_C=AUTOMATIC, double waterVapourDensity_gm3=AUTOMATIC) = 0;
			virtual bool GetITURP676TerrPathGaseousAttenuationActiveState() const = 0;
			virtual double GetITURP676GaseousAttenuation(double frequency_GHz, double atmPressure_hPa=1013.25, double temperature_C=15, double waterVapourDensity_gm3=7.5) const = 0;

			// ITU digial maps
			virtual double GetITUDigitalMapValue(ITUDigitalMap map, double latitude_degrees, double longitude_degrees) const = 0;

			// Terrain elevation data parameters
			virtual void SetPrimaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource) = 0;
			virtual TerrainElevDataSource GetPrimaryTerrainElevDataSource() const = 0;
			virtual void SetSecondaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource) = 0;
			virtual TerrainElevDataSource GetSecondaryTerrainElevDataSource() const = 0;
			virtual void SetTertiaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource) = 0;
			virtual TerrainElevDataSource GetTertiaryTerrainElevDataSource() const = 0;
			virtual void SetTerrainElevDataSourceDirectory(TerrainElevDataSource terrainElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false) = 0;
			virtual const char* GetTerrainElevDataSourceDirectory(TerrainElevDataSource terrainElevSource) const = 0;
			virtual void SetTerrainElevDataSamplingResolution(double samplingResolution_meters) = 0;
			virtual double GetTerrainElevDataSamplingResolution() const = 0;
			virtual void SetTerrainElevDataSourceSamplingMethod(TerrainElevDataSource terrainElevSource, SamplingMethod samplingMethod) = 0;
			virtual SamplingMethod GetTerrainElevDataSourceSamplingMethod(TerrainElevDataSource terrainElevSource) const = 0;
			virtual bool AddCustomTerrainElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* terrainElevData_meters, bool defineNoDataValue=false, float noDataValue=0) = 0;
			virtual void ClearCustomTerrainElevData() = 0;
			virtual double GetTerrainElevation(double latitude_degrees, double longitude_degrees, double noDataValue=0) = 0;

			// Land cover data parameters
			virtual void SetPrimaryLandCoverDataSource(LandCoverDataSource landCoverSource) = 0;
			virtual LandCoverDataSource GetPrimaryLandCoverDataSource() const = 0;
			virtual void SetSecondaryLandCoverDataSource(LandCoverDataSource landCoverSource) = 0;
			virtual LandCoverDataSource GetSecondaryLandCoverDataSource() const = 0;
			virtual void SetLandCoverDataSourceDirectory(LandCoverDataSource landCoverSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false) = 0;
			virtual const char* GetLandCoverDataSourceDirectory(LandCoverDataSource landCoverSource) const = 0;
			virtual bool AddCustomLandCoverData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const short* landCoverData, bool defineNoDataValue=false, short noDataValue=0) = 0;
			virtual void ClearCustomLandCoverData() = 0;
			virtual int GetLandCoverClass(double latitude_degrees, double longitude_degrees) = 0;
			virtual int GetLandCoverClassMappedValue(double latitude_degrees, double longitude_degrees, PropagationModel propagationModel) = 0;
			virtual void SetLandCoverClassMapping(LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel, int modelValue) = 0;
			virtual int GetLandCoverClassMapping(LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel) const = 0;
			virtual void SetDefaultLandCoverClassMapping(LandCoverDataSource landCoverSource, PropagationModel propagationModel, int modelValue) = 0;
			virtual int GetDefaultLandCoverClassMapping(LandCoverDataSource landCoverSource, PropagationModel propagationModel) const = 0;
			virtual void ClearLandCoverClassMappings(LandCoverDataSource landCoverSource, PropagationModel propagationModel) = 0;

			// Surface elevation data parameters
			virtual void SetPrimarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource) = 0;
			virtual SurfaceElevDataSource GetPrimarySurfaceElevDataSource() const = 0;
			virtual void SetSecondarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource) = 0;
			virtual SurfaceElevDataSource GetSecondarySurfaceElevDataSource() const = 0;
			virtual void SetTertiarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource) = 0;
			virtual SurfaceElevDataSource GetTertiarySurfaceElevDataSource() const = 0;
			virtual void SetSurfaceElevDataSourceDirectory(SurfaceElevDataSource surfaceElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false) = 0;
			virtual const char* GetSurfaceElevDataSourceDirectory(SurfaceElevDataSource surfaceElevSource) const = 0;
			virtual void SetSurfaceAndTerrainDataSourcePairing(bool usePairing) = 0;
			virtual bool GetSurfaceAndTerrainDataSourcePairing() const = 0;
			virtual void SetSurfaceElevDataSourceSamplingMethod(SurfaceElevDataSource surfaceElevSource, SamplingMethod samplingMethod) = 0;
			virtual SamplingMethod GetSurfaceElevDataSourceSamplingMethod(SurfaceElevDataSource surfaceElevSource) const = 0;
			virtual bool AddCustomSurfaceElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* surfaceElevData_meters, bool defineNoDataValue=false, float noDataValue=0) = 0;
			virtual void ClearCustomSurfaceElevData() = 0;
			virtual double GetSurfaceElevation(double latitude_degrees, double longitude_degrees, double noDataValue=0) = 0;

			// Reception area parameters
			virtual void SetReceptionAreaCorners(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees) = 0;
			virtual double GetReceptionAreaLowerLeftCornerLatitude() const = 0;
			virtual double GetReceptionAreaLowerLeftCornerLongitude() const = 0;
			virtual double GetReceptionAreaUpperRightCornerLatitude() const = 0;
			virtual double GetReceptionAreaUpperRightCornerLongitude() const = 0;
			virtual void SetReceptionAreaNumHorizontalPoints(int numPoints) = 0;
			virtual int GetReceptionAreaNumHorizontalPoints() const = 0;
			virtual void SetReceptionAreaNumVerticalPoints(int numPoints) = 0;
			virtual int GetReceptionAreaNumVerticalPoints() const = 0;

			// Result type parameters
			virtual void SetResultType(ResultType resultType) = 0;
			virtual ResultType GetResultType() const = 0;

			// Coverage display parameters for vector files (.mif and .kml)
			virtual void ClearCoverageDisplayFills() = 0;
			virtual void AddCoverageDisplayFill(double fromValue, double toValue, int rgbColor) = 0;
			virtual int GetCoverageDisplayNumFills() const = 0;
			virtual double GetCoverageDisplayFillFromValue(int index) const = 0;
			virtual double GetCoverageDisplayFillToValue(int index) const = 0;
			virtual int GetCoverageDisplayFillColor(int index) const = 0;

			// Generating and accessing results
			virtual double GenerateReceptionPointResult(double latitude_degrees, double longitude_degrees) = 0;
			virtual ReceptionPointDetailedResult GenerateReceptionPointDetailedResult(double latitude_degrees, double longitude_degrees) = 0;
			virtual double GenerateProfileReceptionPointResult(double latitude_degrees, double longitude_degrees, int numSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile=NULL, const double* surfaceElevProfile=NULL, const ITURadioClimaticZone* ituRadioClimaticZoneProfile=NULL) = 0;
			virtual void GenerateReceptionAreaResults() = 0;
			virtual int GetGenerateStatus() const = 0;
			virtual double GetReceptionAreaResultValue(int xIndex, int yIndex) const = 0;
			virtual void SetReceptionAreaResultValue(int xIndex, int yIndex, double value) = 0;
			virtual double GetReceptionAreaResultValueAtLatLon(double latitude_degrees, double longitude_degrees) const = 0;
			virtual double GetReceptionAreaResultLatitude(int xIndex, int yIndex) const = 0;
			virtual double GetReceptionAreaResultLongitude(int xIndex, int yIndex) const = 0;
			virtual bool ExportReceptionAreaResultsToTextFile(const char* pathname, const char* resultsColumnName=NULL) const = 0;
			virtual bool ExportReceptionAreaResultsToMifFile(const char* pathname, const char* resultsUnits=NULL) const = 0;
			virtual bool ExportReceptionAreaResultsToKmlFile(const char* pathname, double fillOpacity_percent=50, double lineOpacity_percent=50, const char* resultsUnits=NULL) const = 0;
			virtual bool ExportReceptionAreaResultsToBilFile(const char* pathname) const = 0;
			virtual bool ExportReceptionAreaTerrainElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false) = 0;
			virtual bool ExportReceptionAreaLandCoverClassesToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool mapValues) = 0;
			virtual bool ExportReceptionAreaSurfaceElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false) = 0;
			virtual bool ExportProfilesToCsvFile(const char* pathname, double latitude_degrees, double longitude_degrees) = 0;
		};

		// Factory functions that creates instances of the Simulation object.
		extern "C" CRCCOVLIB_API ISimulation* __stdcall NewSimulation();
		extern "C" CRCCOVLIB_API ISimulation* __stdcall DeepCopySimulation(ISimulation* sim);

		// Location of ITU digital maps
		extern "C" CRCCOVLIB_API bool __stdcall SetITUProprietaryDataDirectory(const char* directory);

#ifdef CRCCOVLIB_WRAP
		extern "C" CRCCOVLIB_API void __stdcall Release(ISimulation* sim);

		// Transmitter parameters
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterLocation(ISimulation* sim, double latitude_degrees, double longitude_degrees);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterLatitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterLongitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterHeight(ISimulation* sim, double height_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterHeight(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterFrequency(ISimulation* sim, double frequency_MHz);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterFrequency(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterPower(ISimulation* sim, double power_watts, PowerType powerType=Crc::Covlib::EIRP);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterPower(ISimulation* sim, PowerType powerType=Crc::Covlib::EIRP);
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterLosses(ISimulation* sim, double losses_dB);
		extern "C" CRCCOVLIB_API double __stdcall GetTransmitterLosses(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTransmitterPolarization(ISimulation* sim, Polarization polarization);
		extern "C" CRCCOVLIB_API Polarization __stdcall GetTransmitterPolarization(ISimulation* sim);
		
		// Receiver parameters
		extern "C" CRCCOVLIB_API void __stdcall SetReceiverHeightAboveGround(ISimulation* sim, double height_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetReceiverHeightAboveGround(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetReceiverLosses(ISimulation* sim, double losses_dB);
		extern "C" CRCCOVLIB_API double __stdcall GetReceiverLosses(ISimulation* sim);

		// Antenna parameters
		extern "C" CRCCOVLIB_API void __stdcall ClearAntennaPatterns(ISimulation* sim, Terminal terminal, bool clearHorizontalPattern=true, bool clearVerticalPattern=true);
		extern "C" CRCCOVLIB_API void __stdcall AddAntennaHorizontalPatternEntry(ISimulation* sim, Terminal terminal, double azimuth_degrees, double gain_dB);
		extern "C" CRCCOVLIB_API void __stdcall AddAntennaVerticalPatternEntry(ISimulation* sim, Terminal terminal, int azimuth_degrees, double elevAngle_degrees, double gain_dB);
		extern "C" CRCCOVLIB_API void __stdcall SetAntennaElectricalTilt(ISimulation* sim, Terminal terminal, double elecricalTilt_degrees);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaElectricalTilt(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API void __stdcall SetAntennaMechanicalTilt(ISimulation* sim, Terminal terminal, double mechanicalTilt_degrees, double azimuth_degrees=0);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaMechanicalTilt(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaMechanicalTiltAzimuth(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API void __stdcall SetAntennaMaximumGain(ISimulation* sim, Terminal terminal, double maxGain_dBi);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaMaximumGain(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API void __stdcall SetAntennaBearing(ISimulation* sim, Terminal terminal, BearingReference bearingRef, double bearing_degrees);
		extern "C" CRCCOVLIB_API BearingReference __stdcall GetAntennaBearingReference(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaBearing(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API double __stdcall NormalizeAntennaHorizontalPattern(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API double __stdcall NormalizeAntennaVerticalPattern(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API void __stdcall SetAntennaPatternApproximationMethod(ISimulation* sim, Terminal terminal, PatternApproximationMethod method);
		extern "C" CRCCOVLIB_API PatternApproximationMethod __stdcall GetAntennaPatternApproximationMethod(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API double __stdcall GetAntennaGain(ISimulation* sim, Terminal terminal, double azimuth_degrees, double elevAngle_degrees, double receiverLatitude_degrees=0, double receiverLongitude_degrees=0);

		// Propagation model selection
		extern "C" CRCCOVLIB_API void __stdcall SetPropagationModel(ISimulation* sim, PropagationModel propagationModel);
		extern "C" CRCCOVLIB_API __stdcall PropagationModel GetPropagationModel(ISimulation* sim);

		// Longley-Rice propagation model parameters
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceSurfaceRefractivity(ISimulation* sim, double refractivity_NUnits);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceSurfaceRefractivity(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceGroundDielectricConst(ISimulation* sim, double dielectricConst);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceGroundDielectricConst(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceGroundConductivity(ISimulation* sim, double groundConduct_Sm);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceGroundConductivity(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceClimaticZone(ISimulation* sim, LRClimaticZone climaticZone);
		extern "C" CRCCOVLIB_API LRClimaticZone __stdcall GetLongleyRiceClimaticZone(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceActivePercentageSet(ISimulation* sim, LRPercentageSet percentageSet);
		extern "C" CRCCOVLIB_API LRPercentageSet __stdcall GetLongleyRiceActivePercentageSet(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceTimePercentage(ISimulation* sim, double time_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceTimePercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceLocationPercentage(ISimulation* sim, double location_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceLocationPercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceSituationPercentage(ISimulation* sim, double situation_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceSituationPercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceConfidencePercentage(ISimulation* sim, double confidence_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceConfidencePercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceReliabilityPercentage(ISimulation* sim, double reliability_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetLongleyRiceReliabilityPercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLongleyRiceModeOfVariability(ISimulation* sim, int mode);
		extern "C" CRCCOVLIB_API int __stdcall GetLongleyRiceModeOfVariability(ISimulation* sim);

		// ITU-R P.1812 propagation model parameters
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812TimePercentage(ISimulation* sim, double time_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812TimePercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812LocationPercentage(ISimulation* sim, double location_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812LocationPercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812AverageRadioRefractivityLapseRate(ISimulation* sim, double deltaN_Nunitskm);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812AverageRadioRefractivityLapseRate(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812SeaLevelSurfaceRefractivity(ISimulation* sim, double N0_Nunits);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812SeaLevelSurfaceRefractivity(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812PredictionResolution(ISimulation* sim, double resolution_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812PredictionResolution(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812RepresentativeClutterHeight(ISimulation* sim, P1812ClutterCategory clutterCategory, double reprHeight_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP1812RepresentativeClutterHeight(ISimulation* sim, P1812ClutterCategory clutterCategory);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812RadioClimaticZonesFile(ISimulation* sim, const char* pathname);
		extern "C" CRCCOVLIB_API const char* __stdcall GetITURP1812RadioClimaticZonesFile(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812LandCoverMappingType(ISimulation* sim, P1812LandCoverMappingType mappingType);
		extern "C" CRCCOVLIB_API P1812LandCoverMappingType __stdcall GetITURP1812LandCoverMappingType(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP1812SurfaceProfileMethod(ISimulation* sim, P1812SurfaceProfileMethod method);
		extern "C" CRCCOVLIB_API P1812SurfaceProfileMethod __stdcall GetITURP1812SurfaceProfileMethod(ISimulation* sim);

		// ITU-R P.452 propagation model parameters
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452TimePercentage(ISimulation* sim, double time_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452TimePercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452PredictionType(ISimulation* sim, P452PredictionType predictionType);
		extern "C" CRCCOVLIB_API P452PredictionType __stdcall GetITURP452PredictionType(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452AverageRadioRefractivityLapseRate(ISimulation* sim, double deltaN_Nunitskm);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452AverageRadioRefractivityLapseRate(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452SeaLevelSurfaceRefractivity(ISimulation* sim, double N0_Nunits);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452SeaLevelSurfaceRefractivity(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452AirTemperature(ISimulation* sim, double temperature_C);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452AirTemperature(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452AirPressure(ISimulation* sim, double pressure_hPa);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452AirPressure(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452RadioClimaticZonesFile(ISimulation* sim, const char* pathname);
		extern "C" CRCCOVLIB_API const char* __stdcall GetITURP452RadioClimaticZonesFile(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452HeightGainModelClutterValue(ISimulation* sim, P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam, double nominalValue);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452HeightGainModelClutterValue(ISimulation* sim, P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452HeightGainModelMode(ISimulation* sim, Terminal terminal, P452HeightGainModelMode mode);
		extern "C" CRCCOVLIB_API P452HeightGainModelMode __stdcall GetITURP452HeightGainModelMode(ISimulation* sim, Terminal terminal);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452RepresentativeClutterHeight(ISimulation* sim, P452ClutterCategory clutterCategory, double reprHeight_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP452RepresentativeClutterHeight(ISimulation* sim, P452ClutterCategory clutterCategory);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452LandCoverMappingType(ISimulation* sim, P452LandCoverMappingType mappingType);
		extern "C" CRCCOVLIB_API P452LandCoverMappingType __stdcall GetITURP452LandCoverMappingType(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP452SurfaceProfileMethod(ISimulation* sim, P452SurfaceProfileMethod method);
		extern "C" CRCCOVLIB_API P452SurfaceProfileMethod __stdcall GetITURP452SurfaceProfileMethod(ISimulation* sim);

		// Extended Hata propagation model parameters
		extern "C" CRCCOVLIB_API void __stdcall SetEHataClutterEnvironment(ISimulation* sim, EHataClutterEnvironment clutterEnvironment);
		extern "C" CRCCOVLIB_API EHataClutterEnvironment __stdcall GetEHataClutterEnvironment(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetEHataReliabilityPercentage(ISimulation* sim, double percent);
		extern "C" CRCCOVLIB_API double __stdcall GetEHataReliabilityPercentage(ISimulation* sim);

		// ITU-R P.2108 statistical clutter loss model for terrestrial paths
		extern "C" CRCCOVLIB_API void __stdcall SetITURP2108TerrestrialStatModelActiveState(ISimulation* sim, bool active);
		extern "C" CRCCOVLIB_API bool __stdcall GetITURP2108TerrestrialStatModelActiveState(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP2108TerrestrialStatModelLocationPercentage(ISimulation* sim, double location_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP2108TerrestrialStatModelLocationPercentage(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP2108TerrestrialStatModelLoss(ISimulation* sim, double frequency_GHz, double distance_km);

		// ITU-R P.2109 building entry loss model
		extern "C" CRCCOVLIB_API void __stdcall SetITURP2109ActiveState(ISimulation* sim, bool active);
		extern "C" CRCCOVLIB_API bool __stdcall GetITURP2109ActiveState(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP2109Probability(ISimulation* sim, double probability_percent);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP2109Probability(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetITURP2109DefaultBuildingType(ISimulation* sim, P2109BuildingType buildingType);
		extern "C" CRCCOVLIB_API P2109BuildingType __stdcall GetITURP2109DefaultBuildingType(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP2109BuildingEntryLoss(ISimulation* sim, double frequency_GHz, double elevAngle_degrees);

		// ITU-R P.676 gaseous attenuation model for terrestrial paths
		extern "C" CRCCOVLIB_API void __stdcall SetITURP676TerrPathGaseousAttenuationActiveState(ISimulation* sim, bool active, double atmPressure_hPa=AUTOMATIC, double temperature_C=AUTOMATIC, double waterVapourDensity_gm3=AUTOMATIC);
		extern "C" CRCCOVLIB_API bool __stdcall GetITURP676TerrPathGaseousAttenuationActiveState(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetITURP676GaseousAttenuation(ISimulation* sim, double frequency_GHz, double atmPressure_hPa=1013.25, double temperature_C=15, double waterVapourDensity_gm3=7.5);

		// ITU digial maps
		extern "C" CRCCOVLIB_API double __stdcall GetITUDigitalMapValue(ISimulation* sim, ITUDigitalMap map, double latitude_degrees, double longitude_degrees);

		// Terrain elevation data parameters
		extern "C" CRCCOVLIB_API void __stdcall SetPrimaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource);
		extern "C" CRCCOVLIB_API TerrainElevDataSource __stdcall GetPrimaryTerrainElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetSecondaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource);
		extern "C" CRCCOVLIB_API TerrainElevDataSource __stdcall GetSecondaryTerrainElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTertiaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource);
		extern "C" CRCCOVLIB_API TerrainElevDataSource __stdcall GetTertiaryTerrainElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTerrainElevDataSourceDirectory(ISimulation* sim, TerrainElevDataSource terrainElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
		extern "C" CRCCOVLIB_API const char* __stdcall GetTerrainElevDataSourceDirectory(ISimulation* sim, TerrainElevDataSource terrainElevSource);
		extern "C" CRCCOVLIB_API void __stdcall SetTerrainElevDataSamplingResolution(ISimulation* sim, double samplingResolution_meters);
		extern "C" CRCCOVLIB_API double __stdcall GetTerrainElevDataSamplingResolution(ISimulation* sim);
		extern "C" CRCCOVLIB_API  void __stdcall SetTerrainElevDataSourceSamplingMethod(ISimulation* sim, TerrainElevDataSource terrainElevSource, SamplingMethod samplingMethod);
		extern "C" CRCCOVLIB_API  SamplingMethod __stdcall GetTerrainElevDataSourceSamplingMethod(ISimulation* sim, TerrainElevDataSource terrainElevSource);
		extern "C" CRCCOVLIB_API bool __stdcall AddCustomTerrainElevData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* terrainElevData_meters, bool defineNoDataValue=false, float noDataValue=0);
		extern "C" CRCCOVLIB_API void __stdcall ClearCustomTerrainElevData(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetTerrainElevation(ISimulation* sim, double latitude_degrees, double longitude_degrees, double noDataValue=0);

		// Land cover data parameters
		extern "C" CRCCOVLIB_API void __stdcall SetPrimaryLandCoverDataSource(ISimulation* sim, LandCoverDataSource landCoverSource);
		extern "C" CRCCOVLIB_API LandCoverDataSource __stdcall GetPrimaryLandCoverDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetSecondaryLandCoverDataSource(ISimulation* sim, LandCoverDataSource landCoverSource);
		extern "C" CRCCOVLIB_API LandCoverDataSource __stdcall GetSecondaryLandCoverDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetLandCoverDataSourceDirectory(ISimulation* sim, LandCoverDataSource landCoverSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
		extern "C" CRCCOVLIB_API const char* __stdcall GetLandCoverDataSourceDirectory(ISimulation* sim, LandCoverDataSource landCoverSource);
		extern "C" CRCCOVLIB_API bool __stdcall AddCustomLandCoverData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const short* landCoverData, bool defineNoDataValue=false, short noDataValue=0);
		extern "C" CRCCOVLIB_API void __stdcall ClearCustomLandCoverData(ISimulation* sim);
		extern "C" CRCCOVLIB_API int __stdcall GetLandCoverClass(ISimulation* sim, double latitude_degrees, double longitude_degrees);
		extern "C" CRCCOVLIB_API int __stdcall GetLandCoverClassMappedValue(ISimulation* sim, double latitude_degrees, double longitude_degrees, PropagationModel propagationModel);
		extern "C" CRCCOVLIB_API void __stdcall SetLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel, int modelValue);
		extern "C" CRCCOVLIB_API int __stdcall GetLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel);
		extern "C" CRCCOVLIB_API void __stdcall SetDefaultLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel, int modelValue);
		extern "C" CRCCOVLIB_API int __stdcall GetDefaultLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel);
		extern "C" CRCCOVLIB_API void __stdcall ClearLandCoverClassMappings(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel);

		// Surface elevation data parameters
		extern "C" CRCCOVLIB_API void __stdcall SetPrimarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource);
		extern "C" CRCCOVLIB_API SurfaceElevDataSource __stdcall GetPrimarySurfaceElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetSecondarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource);
		extern "C" CRCCOVLIB_API SurfaceElevDataSource __stdcall GetSecondarySurfaceElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetTertiarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource);
		extern "C" CRCCOVLIB_API SurfaceElevDataSource __stdcall GetTertiarySurfaceElevDataSource(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetSurfaceElevDataSourceDirectory(ISimulation* sim, SurfaceElevDataSource surfaceElevSource, const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
		extern "C" CRCCOVLIB_API const char* __stdcall GetSurfaceElevDataSourceDirectory(ISimulation* sim, SurfaceElevDataSource surfaceElevSource);
		extern "C" CRCCOVLIB_API void __stdcall SetSurfaceAndTerrainDataSourcePairing(ISimulation* sim, bool usePairing);
		extern "C" CRCCOVLIB_API bool __stdcall GetSurfaceAndTerrainDataSourcePairing(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetSurfaceElevDataSourceSamplingMethod(ISimulation* sim, SurfaceElevDataSource surfaceElevSource, SamplingMethod samplingMethod);
		extern "C" CRCCOVLIB_API SamplingMethod __stdcall GetSurfaceElevDataSourceSamplingMethod(ISimulation* sim, SurfaceElevDataSource surfaceElevSource);
		extern "C" CRCCOVLIB_API bool __stdcall AddCustomSurfaceElevData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* surfaceElevData_meters, bool defineNoDataValue=false, float noDataValue=0);
		extern "C" CRCCOVLIB_API void __stdcall ClearCustomSurfaceElevData(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetSurfaceElevation(ISimulation* sim, double latitude_degrees, double longitude_degrees, double noDataValue=0);

		// Reception area parameters
		extern "C" CRCCOVLIB_API void __stdcall SetReceptionAreaCorners(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaLowerLeftCornerLatitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaLowerLeftCornerLongitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaUpperRightCornerLatitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaUpperRightCornerLongitude(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetReceptionAreaNumHorizontalPoints(ISimulation* sim, int numPoints);
		extern "C" CRCCOVLIB_API int __stdcall GetReceptionAreaNumHorizontalPoints(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall SetReceptionAreaNumVerticalPoints(ISimulation* sim, int numPoints);
		extern "C" CRCCOVLIB_API int __stdcall GetReceptionAreaNumVerticalPoints(ISimulation* sim);

		// Result type parameters
		extern "C" CRCCOVLIB_API void __stdcall SetResultType(ISimulation* sim, ResultType resultType);
		extern "C" CRCCOVLIB_API ResultType __stdcall GetResultType(ISimulation* sim);

		// Coverage display parameters for vector files (.mif and .kml)
		extern "C" CRCCOVLIB_API void __stdcall ClearCoverageDisplayFills(ISimulation* sim);
		extern "C" CRCCOVLIB_API void __stdcall AddCoverageDisplayFill(ISimulation* sim, double fromValue, double toValue, int rgbColor);
		extern "C" CRCCOVLIB_API int __stdcall GetCoverageDisplayNumFills(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetCoverageDisplayFillFromValue(ISimulation* sim, int index);
		extern "C" CRCCOVLIB_API double __stdcall GetCoverageDisplayFillToValue(ISimulation* sim, int index);
		extern "C" CRCCOVLIB_API int __stdcall GetCoverageDisplayFillColor(ISimulation* sim, int index);

		// Generating and accessing results
		extern "C" CRCCOVLIB_API double __stdcall GenerateReceptionPointResult(ISimulation* sim, double latitude_degrees, double longitude_degrees);
		extern "C" CRCCOVLIB_API ReceptionPointDetailedResult __stdcall GenerateReceptionPointDetailedResult(ISimulation* sim, double latitude_degrees, double longitude_degrees);
		extern "C" CRCCOVLIB_API double __stdcall GenerateProfileReceptionPointResult(ISimulation* sim, double latitude_degrees, double longitude_degrees, int numSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile=NULL, const double* surfaceElevProfile=NULL, const ITURadioClimaticZone* ituRadioClimaticZoneProfile=NULL);
		extern "C" CRCCOVLIB_API void __stdcall GenerateReceptionAreaResults(ISimulation* sim);
		extern "C" CRCCOVLIB_API int __stdcall GetGenerateStatus(ISimulation* sim);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaResultValue(ISimulation* sim, int xIndex, int yIndex);
		extern "C" CRCCOVLIB_API void SetReceptionAreaResultValue(ISimulation* sim, int xIndex, int yIndex, double value);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaResultValueAtLatLon(ISimulation* sim, double latitude_degrees, double longitude_degrees);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaResultLatitude(ISimulation* sim, int xIndex, int yIndex);
		extern "C" CRCCOVLIB_API double __stdcall GetReceptionAreaResultLongitude(ISimulation* sim, int xIndex, int yIndex);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaResultsToTextFile(ISimulation* sim, const char* pathname, const char* resultsColumnName=NULL);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaResultsToMifFile(ISimulation* sim, const char* pathname, const char* resultsUnits=NULL);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaResultsToKmlFile(ISimulation* sim, const char* pathname, double fillOpacity_percent=50, double lineOpacity_percent=50, const char* resultsUnits=NULL);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaResultsToBilFile(ISimulation* sim, const char* pathname);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaTerrainElevationToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaLandCoverClassesToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool mapValues);
		extern "C" CRCCOVLIB_API bool __stdcall ExportReceptionAreaSurfaceElevationToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero=false) ;
		extern "C" CRCCOVLIB_API bool __stdcall ExportProfilesToCsvFile(ISimulation* sim, const char* pathname, double latitude_degrees, double longitude_degrees);

#endif // ifdef CRCCOVLIB_WRAP
	}
}
