// CRC-COVLIB.cpp : Defines the exported functions for the DLL application.
//

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#else
	#define APIENTRY
#endif

#include "Simulation.h"
#include "ITURP_DigitalMaps.h"

namespace Crc
{
	namespace Covlib
	{
		CRCCOVLIB_API ISimulation* APIENTRY NewSimulation()
		{
			return new Simulation;
		}

		CRCCOVLIB_API ISimulation* APIENTRY DeepCopySimulation(ISimulation* sim)
		{
			return new Simulation(*((Simulation*)sim));
		}

		CRCCOVLIB_API bool APIENTRY SetITUProprietaryDataDirectory(const char* directory)
		{
		std::string ituDir = directory;
		bool success;

			success = ITURP_DigitalMaps::Init_DN50((ituDir + "/DN50.TXT").c_str());
			success &= ITURP_DigitalMaps::Init_N050((ituDir + "/N050.TXT").c_str());
			success &= ITURP_DigitalMaps::Init_T_Annual((ituDir + "/T_Annual.TXT").c_str());
			success &= ITURP_DigitalMaps::Init_Surfwv_50((ituDir + "/surfwv_50_fixed.txt").c_str());
			return success;
		}

#ifdef CRCCOVLIB_WRAP
		CRCCOVLIB_API void APIENTRY Release(ISimulation* sim)
		{
			return sim->Release();
		}


		// Transmitter parameters

		CRCCOVLIB_API void APIENTRY SetTransmitterLocation(ISimulation* sim, double latitude_degrees, double longitude_degrees)
		{
			return sim->SetTransmitterLocation(latitude_degrees, longitude_degrees);
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterLatitude(ISimulation* sim)
		{
			return sim->GetTransmitterLatitude();
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterLongitude(ISimulation* sim)
		{
			return sim->GetTransmitterLongitude();
		}

		CRCCOVLIB_API void APIENTRY SetTransmitterHeight(ISimulation* sim, double height_meters)
		{
			return sim->SetTransmitterHeight(height_meters);
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterHeight(ISimulation* sim)
		{
			return sim->GetTransmitterHeight();
		}

		CRCCOVLIB_API void APIENTRY SetTransmitterFrequency(ISimulation* sim, double frequency_MHz)
		{
			return sim->SetTransmitterFrequency(frequency_MHz);
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterFrequency(ISimulation* sim)
		{
			return sim->GetTransmitterFrequency();
		}

		CRCCOVLIB_API void APIENTRY SetTransmitterPower(ISimulation* sim, double power_watts, PowerType powerType)
		{
			return sim->SetTransmitterPower(power_watts, powerType);
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterPower(ISimulation* sim, PowerType powerType)
		{
			return sim->GetTransmitterPower(powerType);
		}

		CRCCOVLIB_API void APIENTRY SetTransmitterLosses(ISimulation* sim, double losses_dB)
		{
			return sim->SetTransmitterLosses(losses_dB);
		}

		CRCCOVLIB_API double APIENTRY GetTransmitterLosses(ISimulation* sim)
		{
			return sim->GetTransmitterLosses();
		}

		CRCCOVLIB_API void APIENTRY SetTransmitterPolarization(ISimulation* sim, Polarization polarization)
		{
			return sim->SetTransmitterPolarization(polarization);
		}

		CRCCOVLIB_API Polarization APIENTRY GetTransmitterPolarization(ISimulation* sim)
		{
			return sim->GetTransmitterPolarization();
		}


		// Receiver parameters

		CRCCOVLIB_API void APIENTRY SetReceiverHeightAboveGround(ISimulation* sim, double height_meters)
		{
			return sim->SetReceiverHeightAboveGround(height_meters);
		}

		CRCCOVLIB_API double APIENTRY GetReceiverHeightAboveGround(ISimulation* sim)
		{
			return sim->GetReceiverHeightAboveGround();
		}

		CRCCOVLIB_API void APIENTRY SetReceiverLosses(ISimulation* sim, double losses_dB)
		{
			return sim->SetReceiverLosses(losses_dB);
		}

		CRCCOVLIB_API double APIENTRY GetReceiverLosses(ISimulation* sim)
		{
			return sim->GetReceiverLosses();
		}


		// Antenna parameters

		CRCCOVLIB_API void APIENTRY ClearAntennaPatterns(ISimulation* sim, Terminal terminal, bool clearHorizontalPattern, bool clearVerticalPattern)
		{
			return sim->ClearAntennaPatterns(terminal, clearHorizontalPattern, clearVerticalPattern);
		}

		CRCCOVLIB_API void APIENTRY AddAntennaHorizontalPatternEntry(ISimulation* sim, Terminal terminal, double azimuth_degrees, double gain_dB)
		{
			return sim->AddAntennaHorizontalPatternEntry(terminal, azimuth_degrees, gain_dB);
		}

		CRCCOVLIB_API void APIENTRY AddAntennaVerticalPatternEntry(ISimulation* sim, Terminal terminal, int azimuth_degrees, double elevAngle_degrees, double gain_dB)
		{
			return sim->AddAntennaVerticalPatternEntry(terminal, azimuth_degrees, elevAngle_degrees, gain_dB);
		}

		CRCCOVLIB_API void APIENTRY SetAntennaElectricalTilt(ISimulation* sim, Terminal terminal, double elecricalTilt_degrees)
		{
			return sim->SetAntennaElectricalTilt(terminal, elecricalTilt_degrees);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaElectricalTilt(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaElectricalTilt(terminal);
		}

		CRCCOVLIB_API void APIENTRY SetAntennaMechanicalTilt(ISimulation* sim, Terminal terminal, double mechanicalTilt_degrees, double azimuth_degrees)
		{
			return sim->SetAntennaMechanicalTilt(terminal, mechanicalTilt_degrees, azimuth_degrees);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaMechanicalTilt(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaMechanicalTilt(terminal);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaMechanicalTiltAzimuth(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaMechanicalTiltAzimuth(terminal);
		}

		CRCCOVLIB_API void APIENTRY SetAntennaMaximumGain(ISimulation* sim, Terminal terminal, double maxGain_dBi)
		{
			return sim->SetAntennaMaximumGain(terminal, maxGain_dBi);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaMaximumGain(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaMaximumGain(terminal);
		}

		CRCCOVLIB_API void APIENTRY SetAntennaBearing(ISimulation* sim, Terminal terminal, BearingReference bearingRef, double bearing_degrees)
		{
			return sim->SetAntennaBearing(terminal, bearingRef, bearing_degrees);
		}

		CRCCOVLIB_API BearingReference APIENTRY GetAntennaBearingReference(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaBearingReference(terminal);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaBearing(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaBearing(terminal);
		}

		CRCCOVLIB_API double APIENTRY NormalizeAntennaHorizontalPattern(ISimulation* sim, Terminal terminal)
		{
			return sim->NormalizeAntennaHorizontalPattern(terminal);
		}

		CRCCOVLIB_API double APIENTRY NormalizeAntennaVerticalPattern(ISimulation* sim, Terminal terminal)
		{
			return sim->NormalizeAntennaVerticalPattern(terminal);
		}

		CRCCOVLIB_API void APIENTRY SetAntennaPatternApproximationMethod(ISimulation* sim, Terminal terminal, PatternApproximationMethod method)
		{
			return sim->SetAntennaPatternApproximationMethod(terminal, method);
		}

		CRCCOVLIB_API PatternApproximationMethod APIENTRY GetAntennaPatternApproximationMethod(ISimulation* sim, Terminal terminal)
		{
			return sim->GetAntennaPatternApproximationMethod(terminal);
		}

		CRCCOVLIB_API double APIENTRY GetAntennaGain(ISimulation* sim, Terminal terminal, double azimuth_degrees, double elevAngle_degrees, double receiverLatitude_degrees, double receiverLongitude_degrees)
		{
			return sim->GetAntennaGain(terminal, azimuth_degrees, elevAngle_degrees, receiverLatitude_degrees, receiverLongitude_degrees);
		}


		// Propagation model selection

		CRCCOVLIB_API void APIENTRY SetPropagationModel(ISimulation* sim, PropagationModel propagationModel)
		{
			return sim->SetPropagationModel(propagationModel);
		}
		
		CRCCOVLIB_API APIENTRY PropagationModel GetPropagationModel(ISimulation* sim)
		{
			return sim->GetPropagationModel();
		}


		// Longley-Rice propagation model parameters

		CRCCOVLIB_API void APIENTRY SetLongleyRiceSurfaceRefractivity(ISimulation* sim, double refractivity_NUnits)
		{
			return sim->SetLongleyRiceSurfaceRefractivity(refractivity_NUnits);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceSurfaceRefractivity(ISimulation* sim)
		{
			return sim->GetLongleyRiceSurfaceRefractivity();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceGroundDielectricConst(ISimulation* sim, double dielectricConst)
		{
			return sim->SetLongleyRiceGroundDielectricConst(dielectricConst);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceGroundDielectricConst(ISimulation* sim)
		{
			return sim->GetLongleyRiceGroundDielectricConst();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceGroundConductivity(ISimulation* sim, double groundConduct_Sm)
		{
			return sim->SetLongleyRiceGroundConductivity(groundConduct_Sm);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceGroundConductivity(ISimulation* sim)
		{
			return sim->GetLongleyRiceGroundConductivity();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceClimaticZone(ISimulation* sim, LRClimaticZone climaticZone)
		{
			return sim->SetLongleyRiceClimaticZone(climaticZone);
		}

		CRCCOVLIB_API LRClimaticZone APIENTRY GetLongleyRiceClimaticZone(ISimulation* sim)
		{
			return sim->GetLongleyRiceClimaticZone();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceActivePercentageSet(ISimulation* sim, LRPercentageSet percentageSet)
		{
			return sim->SetLongleyRiceActivePercentageSet(percentageSet);
		}

		CRCCOVLIB_API LRPercentageSet __stdcall GetLongleyRiceActivePercentageSet(ISimulation* sim)
		{
			return sim->GetLongleyRiceActivePercentageSet();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceTimePercentage(ISimulation* sim, double time_percent)
		{
			return sim->SetLongleyRiceTimePercentage(time_percent);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceTimePercentage(ISimulation* sim)
		{
			return sim->GetLongleyRiceTimePercentage();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceLocationPercentage(ISimulation* sim, double location_percent)
		{
			return sim->SetLongleyRiceLocationPercentage(location_percent);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceLocationPercentage(ISimulation* sim)
		{
			return sim->GetLongleyRiceLocationPercentage();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceSituationPercentage(ISimulation* sim, double situation_percent)
		{
			return sim->SetLongleyRiceSituationPercentage(situation_percent);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceSituationPercentage(ISimulation* sim)
		{
			return sim->GetLongleyRiceSituationPercentage();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceConfidencePercentage(ISimulation* sim, double confidence_percent)
		{
			return sim->SetLongleyRiceConfidencePercentage(confidence_percent);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceConfidencePercentage(ISimulation* sim)
		{
			return sim->GetLongleyRiceConfidencePercentage();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceReliabilityPercentage(ISimulation* sim, double reliability_percent)
		{
			return sim->SetLongleyRiceReliabilityPercentage(reliability_percent);
		}

		CRCCOVLIB_API double APIENTRY GetLongleyRiceReliabilityPercentage(ISimulation* sim)
		{
			return sim->GetLongleyRiceReliabilityPercentage();
		}

		CRCCOVLIB_API void APIENTRY SetLongleyRiceModeOfVariability(ISimulation* sim, int mode)
		{
			return sim->SetLongleyRiceModeOfVariability(mode);
		}

		CRCCOVLIB_API int APIENTRY GetLongleyRiceModeOfVariability(ISimulation* sim)
		{
			return sim->GetLongleyRiceModeOfVariability();
		}


		// ITU-R P.1812 propagation model parameters

		CRCCOVLIB_API void APIENTRY SetITURP1812TimePercentage(ISimulation* sim, double time_percent)
		{
			return sim->SetITURP1812TimePercentage(time_percent);
		}

		CRCCOVLIB_API double APIENTRY GetITURP1812TimePercentage(ISimulation* sim)
		{
			return sim->GetITURP1812TimePercentage();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812LocationPercentage(ISimulation* sim, double location_percent)
		{
			return sim->SetITURP1812LocationPercentage(location_percent);
		}

		CRCCOVLIB_API double APIENTRY GetITURP1812LocationPercentage(ISimulation* sim)
		{
			return sim->GetITURP1812LocationPercentage();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812AverageRadioRefractivityLapseRate(ISimulation* sim, double deltaN_Nunitskm)
		{
			return sim->SetITURP1812AverageRadioRefractivityLapseRate(deltaN_Nunitskm);
		}
		
		CRCCOVLIB_API double APIENTRY GetITURP1812AverageRadioRefractivityLapseRate(ISimulation* sim)
		{
			return sim->GetITURP1812AverageRadioRefractivityLapseRate();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812SeaLevelSurfaceRefractivity(ISimulation* sim, double N0_Nunits)
		{
			return sim->SetITURP1812SeaLevelSurfaceRefractivity(N0_Nunits);
		}

		CRCCOVLIB_API double APIENTRY GetITURP1812SeaLevelSurfaceRefractivity(ISimulation* sim)
		{
			return sim->GetITURP1812SeaLevelSurfaceRefractivity();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812PredictionResolution(ISimulation* sim, double resolution_meters)
		{
			return sim->SetITURP1812PredictionResolution(resolution_meters);
		}

		CRCCOVLIB_API double APIENTRY GetITURP1812PredictionResolution(ISimulation* sim)
		{
			return sim->GetITURP1812PredictionResolution();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812RepresentativeClutterHeight(ISimulation* sim, P1812ClutterCategory clutterCategory, double reprHeight_meters)
		{
			return sim->SetITURP1812RepresentativeClutterHeight(clutterCategory, reprHeight_meters);
		}

		CRCCOVLIB_API double APIENTRY GetITURP1812RepresentativeClutterHeight(ISimulation* sim, P1812ClutterCategory clutterCategory)
		{
			return sim->GetITURP1812RepresentativeClutterHeight(clutterCategory);
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812RadioClimaticZonesFile(ISimulation* sim, const char* pathname)
		{
			return sim->SetITURP1812RadioClimaticZonesFile(pathname);
		}

		CRCCOVLIB_API const char* APIENTRY GetITURP1812RadioClimaticZonesFile(ISimulation* sim)
		{
			return sim->GetITURP1812RadioClimaticZonesFile();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812LandCoverMappingType(ISimulation* sim, P1812LandCoverMappingType mappingType)
		{
			return sim->SetITURP1812LandCoverMappingType(mappingType);
		}

		CRCCOVLIB_API P1812LandCoverMappingType APIENTRY GetITURP1812LandCoverMappingType(ISimulation* sim)
		{
			return sim->GetITURP1812LandCoverMappingType();
		}

		CRCCOVLIB_API void APIENTRY SetITURP1812SurfaceProfileMethod(ISimulation* sim, P1812SurfaceProfileMethod method)
		{
			return sim->SetITURP1812SurfaceProfileMethod(method);
		}

		CRCCOVLIB_API P1812SurfaceProfileMethod APIENTRY GetITURP1812SurfaceProfileMethod(ISimulation* sim)
		{
			return sim->GetITURP1812SurfaceProfileMethod();
		}


		// ITU-R P.452 propagation model parameters

		CRCCOVLIB_API void APIENTRY SetITURP452TimePercentage(ISimulation* sim, double time_percent)
		{
			return sim->SetITURP452TimePercentage(time_percent);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452TimePercentage(ISimulation* sim)
		{
			return sim->GetITURP452TimePercentage();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452PredictionType(ISimulation* sim, P452PredictionType predictionType)
		{
			return sim->SetITURP452PredictionType(predictionType);
		}

		CRCCOVLIB_API P452PredictionType APIENTRY GetITURP452PredictionType(ISimulation* sim)
		{
			return sim->GetITURP452PredictionType();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452AverageRadioRefractivityLapseRate(ISimulation* sim, double deltaN_Nunitskm)
		{
			return sim->SetITURP452AverageRadioRefractivityLapseRate(deltaN_Nunitskm);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452AverageRadioRefractivityLapseRate(ISimulation* sim)
		{
			return sim->GetITURP452AverageRadioRefractivityLapseRate();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452SeaLevelSurfaceRefractivity(ISimulation* sim, double N0_Nunits)
		{
			return sim->SetITURP452SeaLevelSurfaceRefractivity(N0_Nunits);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452SeaLevelSurfaceRefractivity(ISimulation* sim)
		{
			return sim->GetITURP452SeaLevelSurfaceRefractivity();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452AirTemperature(ISimulation* sim, double temperature_C)
		{
			return sim->SetITURP452AirTemperature(temperature_C);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452AirTemperature(ISimulation* sim)
		{
			return sim->GetITURP452AirTemperature();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452AirPressure(ISimulation* sim, double pressure_hPa)
		{
			return sim->SetITURP452AirPressure(pressure_hPa);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452AirPressure(ISimulation* sim)
		{
			return sim->GetITURP452AirPressure();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452RadioClimaticZonesFile(ISimulation* sim, const char* pathname)
		{
			return sim->SetITURP452RadioClimaticZonesFile(pathname);
		}

		CRCCOVLIB_API const char* APIENTRY GetITURP452RadioClimaticZonesFile(ISimulation* sim)
		{
			return sim->GetITURP452RadioClimaticZonesFile();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452HeightGainModelClutterValue(ISimulation* sim, P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam, double nominalValue)
		{
			return sim->SetITURP452HeightGainModelClutterValue(clutterCategory, nominalParam, nominalValue);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452HeightGainModelClutterValue(ISimulation* sim, P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam)
		{
			return sim->GetITURP452HeightGainModelClutterValue(clutterCategory, nominalParam);
		}

		CRCCOVLIB_API void APIENTRY SetITURP452HeightGainModelMode(ISimulation* sim, Terminal terminal, P452HeightGainModelMode mode)
		{
			return sim->SetITURP452HeightGainModelMode(terminal, mode);
		}

		CRCCOVLIB_API P452HeightGainModelMode APIENTRY GetITURP452HeightGainModelMode(ISimulation* sim, Terminal terminal)
		{
			return sim->GetITURP452HeightGainModelMode(terminal);
		}

		CRCCOVLIB_API void APIENTRY SetITURP452RepresentativeClutterHeight(ISimulation* sim, P452ClutterCategory clutterCategory, double reprHeight_meters)
		{
			return sim->SetITURP452RepresentativeClutterHeight(clutterCategory, reprHeight_meters);
		}

		CRCCOVLIB_API double APIENTRY GetITURP452RepresentativeClutterHeight(ISimulation* sim, P452ClutterCategory clutterCategory)
		{
			return sim->GetITURP452RepresentativeClutterHeight(clutterCategory);
		}

		CRCCOVLIB_API void APIENTRY SetITURP452LandCoverMappingType(ISimulation* sim, P452LandCoverMappingType mappingType)
		{
			return sim->SetITURP452LandCoverMappingType(mappingType);
		}

		CRCCOVLIB_API P452LandCoverMappingType APIENTRY GetITURP452LandCoverMappingType(ISimulation* sim)
		{
			return sim->GetITURP452LandCoverMappingType();
		}

		CRCCOVLIB_API void APIENTRY SetITURP452SurfaceProfileMethod(ISimulation* sim, P452SurfaceProfileMethod method)
		{
			return sim->SetITURP452SurfaceProfileMethod(method);
		}

		CRCCOVLIB_API P452SurfaceProfileMethod APIENTRY GetITURP452SurfaceProfileMethod(ISimulation* sim)
		{
			return sim->GetITURP452SurfaceProfileMethod();
		}


		// Extended Hata propagation model parameters
		CRCCOVLIB_API void APIENTRY SetEHataClutterEnvironment(ISimulation* sim, EHataClutterEnvironment clutterEnvironment)
		{
			return sim->SetEHataClutterEnvironment(clutterEnvironment);
		}

		CRCCOVLIB_API EHataClutterEnvironment APIENTRY GetEHataClutterEnvironment(ISimulation* sim)
		{
			return sim->GetEHataClutterEnvironment();
		}

		CRCCOVLIB_API void APIENTRY SetEHataReliabilityPercentage(ISimulation* sim, double percent)
		{
			return sim->SetEHataReliabilityPercentage(percent);
		}

		CRCCOVLIB_API double APIENTRY GetEHataReliabilityPercentage(ISimulation* sim)
		{
			return sim->GetEHataReliabilityPercentage();
		}


		// ITU-R P.2108 statistical clutter loss model for terrestrial paths

		CRCCOVLIB_API void APIENTRY SetITURP2108TerrestrialStatModelActiveState(ISimulation* sim, bool active)
		{
			return sim->SetITURP2108TerrestrialStatModelActiveState(active);
		}

		CRCCOVLIB_API bool APIENTRY GetITURP2108TerrestrialStatModelActiveState(ISimulation* sim)
		{
			return sim->GetITURP2108TerrestrialStatModelActiveState();
		}

		CRCCOVLIB_API void APIENTRY SetITURP2108TerrestrialStatModelLocationPercentage(ISimulation* sim, double location_percent)
		{
			return sim->SetITURP2108TerrestrialStatModelLocationPercentage(location_percent);
		}

		CRCCOVLIB_API double APIENTRY GetITURP2108TerrestrialStatModelLocationPercentage(ISimulation* sim)
		{
			return sim->GetITURP2108TerrestrialStatModelLocationPercentage();
		}

		CRCCOVLIB_API double APIENTRY GetITURP2108TerrestrialStatModelLoss(ISimulation* sim, double frequency_GHz, double distance_km)
		{
			return sim->GetITURP2108TerrestrialStatModelLoss(frequency_GHz, distance_km);
		}


		// ITU-R P.2109 building entry loss model

		CRCCOVLIB_API void APIENTRY SetITURP2109ActiveState(ISimulation* sim, bool active)
		{
			return sim->SetITURP2109ActiveState(active);
		}

		CRCCOVLIB_API bool APIENTRY GetITURP2109ActiveState(ISimulation* sim)
		{
			return sim->GetITURP2109ActiveState();
		}

		CRCCOVLIB_API void APIENTRY SetITURP2109Probability(ISimulation* sim, double probability_percent)
		{
			return sim->SetITURP2109Probability(probability_percent);
		}

		CRCCOVLIB_API double APIENTRY GetITURP2109Probability(ISimulation* sim)
		{
			return sim->GetITURP2109Probability();
		}

		CRCCOVLIB_API void APIENTRY SetITURP2109DefaultBuildingType(ISimulation* sim, P2109BuildingType buildingType)
		{
			return sim->SetITURP2109DefaultBuildingType(buildingType);
		}

		CRCCOVLIB_API P2109BuildingType APIENTRY GetITURP2109DefaultBuildingType(ISimulation* sim)
		{
			return sim->GetITURP2109DefaultBuildingType();
		}

		CRCCOVLIB_API double APIENTRY GetITURP2109BuildingEntryLoss(ISimulation* sim, double frequency_GHz, double elevAngle_degrees)
		{
			return sim->GetITURP2109BuildingEntryLoss(frequency_GHz, elevAngle_degrees);
		}


		// ITU-R P.676 gaseous attenuation model for terrestrial paths

		CRCCOVLIB_API void APIENTRY SetITURP676TerrPathGaseousAttenuationActiveState(ISimulation* sim, bool active, double atmPressure_hPa, double temperature_C, double waterVapourDensity_gm3)
		{
			return sim->SetITURP676TerrPathGaseousAttenuationActiveState(active, atmPressure_hPa, temperature_C, waterVapourDensity_gm3);
		}

		CRCCOVLIB_API bool APIENTRY GetITURP676TerrPathGaseousAttenuationActiveState(ISimulation* sim)
		{
			return sim->GetITURP676TerrPathGaseousAttenuationActiveState();
		}

		CRCCOVLIB_API double APIENTRY GetITURP676GaseousAttenuation(ISimulation* sim, double frequency_GHz, double atmPressure_hPa, double temperature_C, double waterVapourDensity_gm3)
		{
			return sim->GetITURP676GaseousAttenuation(frequency_GHz, atmPressure_hPa, temperature_C, waterVapourDensity_gm3);
		}


		// ITU digial maps
		
		CRCCOVLIB_API double APIENTRY GetITUDigitalMapValue(ISimulation* sim, ITUDigitalMap map, double latitude_degrees, double longitude_degrees)
		{
			return sim->GetITUDigitalMapValue(map, latitude_degrees, longitude_degrees);
		}


		// Terrain elevation data parameters

		CRCCOVLIB_API void APIENTRY SetPrimaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource)
		{
			return sim->SetPrimaryTerrainElevDataSource(terrainElevSource);
		}

		CRCCOVLIB_API TerrainElevDataSource APIENTRY GetPrimaryTerrainElevDataSource(ISimulation* sim)
		{
			return sim->GetPrimaryTerrainElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetSecondaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource)
		{
			return sim->SetSecondaryTerrainElevDataSource(terrainElevSource);
		}

		CRCCOVLIB_API TerrainElevDataSource APIENTRY GetSecondaryTerrainElevDataSource(ISimulation* sim)
		{
			return sim->GetSecondaryTerrainElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetTertiaryTerrainElevDataSource(ISimulation* sim, TerrainElevDataSource terrainElevSource)
		{
			return sim->SetTertiaryTerrainElevDataSource(terrainElevSource);
		}

		CRCCOVLIB_API TerrainElevDataSource APIENTRY GetTertiaryTerrainElevDataSource(ISimulation* sim)
		{
			return sim->GetTertiaryTerrainElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetTerrainElevDataSourceDirectory(ISimulation* sim, TerrainElevDataSource terrainElevSource, const char* directory, bool useIndexFile, bool overwriteIndexFile)
		{
			return sim->SetTerrainElevDataSourceDirectory(terrainElevSource, directory, useIndexFile, overwriteIndexFile);
		}

		CRCCOVLIB_API const char* APIENTRY GetTerrainElevDataSourceDirectory(ISimulation* sim, TerrainElevDataSource terrainElevSource)
		{
			return sim->GetTerrainElevDataSourceDirectory(terrainElevSource);
		}

		CRCCOVLIB_API void APIENTRY SetTerrainElevDataSamplingResolution(ISimulation* sim, double samplingResolution_meters)
		{
			return sim->SetTerrainElevDataSamplingResolution(samplingResolution_meters);
		}

		CRCCOVLIB_API double APIENTRY GetTerrainElevDataSamplingResolution(ISimulation* sim)
		{
			return sim->GetTerrainElevDataSamplingResolution();
		}

		CRCCOVLIB_API void APIENTRY SetTerrainElevDataSourceSamplingMethod(ISimulation* sim, TerrainElevDataSource terrainElevSource, SamplingMethod samplingMethod)
		{
			return sim->SetTerrainElevDataSourceSamplingMethod(terrainElevSource, samplingMethod);
		}

		CRCCOVLIB_API SamplingMethod APIENTRY GetTerrainElevDataSourceSamplingMethod(ISimulation* sim, TerrainElevDataSource terrainElevSource)
		{
			return sim->GetTerrainElevDataSourceSamplingMethod(terrainElevSource);
		}

		CRCCOVLIB_API bool APIENTRY AddCustomTerrainElevData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* terrainElevData_meters, bool defineNoDataValue/*=false*/, float noDataValue/*=0*/)
		{
			return sim->AddCustomTerrainElevData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, terrainElevData_meters, defineNoDataValue, noDataValue);
		}

		CRCCOVLIB_API void __stdcall ClearCustomTerrainElevData(ISimulation* sim)
		{
			return sim->ClearCustomTerrainElevData();
		}

		CRCCOVLIB_API double APIENTRY GetTerrainElevation(ISimulation* sim, double latitude_degrees, double longitude_degrees, double noDataValue)
		{
			return sim->GetTerrainElevation(latitude_degrees, longitude_degrees, noDataValue);
		}


		// Land cover data parameters

		CRCCOVLIB_API void APIENTRY SetPrimaryLandCoverDataSource(ISimulation* sim, LandCoverDataSource landCoverSource)
		{
			return sim->SetPrimaryLandCoverDataSource(landCoverSource);
		}

		CRCCOVLIB_API LandCoverDataSource APIENTRY GetPrimaryLandCoverDataSource(ISimulation* sim)
		{
			return sim->GetPrimaryLandCoverDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetSecondaryLandCoverDataSource(ISimulation* sim, LandCoverDataSource landCoverSource)
		{
			return sim->SetSecondaryLandCoverDataSource(landCoverSource);
		}

		CRCCOVLIB_API LandCoverDataSource APIENTRY GetSecondaryLandCoverDataSource(ISimulation* sim)
		{
			return sim->GetSecondaryLandCoverDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetLandCoverDataSourceDirectory(ISimulation* sim, LandCoverDataSource landCoverSource, const char* directory, bool useIndexFile, bool overwriteIndexFile)
		{
			return sim->SetLandCoverDataSourceDirectory(landCoverSource, directory, useIndexFile, overwriteIndexFile);
		}

		CRCCOVLIB_API const char* APIENTRY GetLandCoverDataSourceDirectory(ISimulation* sim, LandCoverDataSource landCoverSource)
		{
			return sim->GetLandCoverDataSourceDirectory(landCoverSource);
		}

		CRCCOVLIB_API bool APIENTRY AddCustomLandCoverData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const short* landCoverData, bool defineNoDataValue/*=false*/, short noDataValue/*=0*/)
		{
			return sim->AddCustomLandCoverData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, landCoverData, defineNoDataValue, noDataValue);
		}

		CRCCOVLIB_API void APIENTRY ClearCustomLandCoverData(ISimulation* sim)
		{
			return sim->ClearCustomLandCoverData();
		}

		CRCCOVLIB_API int APIENTRY GetLandCoverClass(ISimulation* sim, double latitude_degrees, double longitude_degrees)
		{
			return sim->GetLandCoverClass(latitude_degrees, longitude_degrees);
		}

		CRCCOVLIB_API int APIENTRY GetLandCoverClassMappedValue(ISimulation* sim, double latitude_degrees, double longitude_degrees, PropagationModel propagationModel)
		{
			return sim->GetLandCoverClassMappedValue(latitude_degrees, longitude_degrees, propagationModel);
		}

		CRCCOVLIB_API void APIENTRY SetLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel, int modelValue)
		{
			return sim->SetLandCoverClassMapping(landCoverSource, sourceClass, propagationModel, modelValue);
		}

		CRCCOVLIB_API int APIENTRY GetLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel)
		{
			return sim->GetLandCoverClassMapping(landCoverSource, sourceClass, propagationModel);
		}

		CRCCOVLIB_API void APIENTRY SetDefaultLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel, int modelValue)
		{
			return sim->SetDefaultLandCoverClassMapping(landCoverSource, propagationModel, modelValue);
		}

		CRCCOVLIB_API int APIENTRY GetDefaultLandCoverClassMapping(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel)
		{
			return sim->GetDefaultLandCoverClassMapping(landCoverSource, propagationModel);
		}

		CRCCOVLIB_API void APIENTRY ClearLandCoverClassMappings(ISimulation* sim, LandCoverDataSource landCoverSource, PropagationModel propagationModel)
		{
			return sim->ClearLandCoverClassMappings(landCoverSource, propagationModel);
		}


		// Surface elevation data parameters

		CRCCOVLIB_API void APIENTRY SetPrimarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource)
		{
			return sim->SetPrimarySurfaceElevDataSource(surfaceElevSource);
		}

		CRCCOVLIB_API SurfaceElevDataSource APIENTRY GetPrimarySurfaceElevDataSource(ISimulation* sim)
		{
			return sim->GetPrimarySurfaceElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetSecondarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource)
		{
			return sim->SetSecondarySurfaceElevDataSource(surfaceElevSource);
		}

		CRCCOVLIB_API SurfaceElevDataSource APIENTRY GetSecondarySurfaceElevDataSource(ISimulation* sim)
		{
			return sim->GetSecondarySurfaceElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetTertiarySurfaceElevDataSource(ISimulation* sim, SurfaceElevDataSource surfaceElevSource)
		{
			return sim->SetTertiarySurfaceElevDataSource(surfaceElevSource);
		}

		CRCCOVLIB_API SurfaceElevDataSource APIENTRY GetTertiarySurfaceElevDataSource(ISimulation* sim)
		{
			return sim->GetTertiarySurfaceElevDataSource();
		}

		CRCCOVLIB_API void APIENTRY SetSurfaceElevDataSourceDirectory(ISimulation* sim, SurfaceElevDataSource surfaceElevSource, const char* directory, bool useIndexFile, bool overwriteIndexFile)
		{
			return sim->SetSurfaceElevDataSourceDirectory(surfaceElevSource, directory, useIndexFile, overwriteIndexFile);
		}

		CRCCOVLIB_API const char* APIENTRY GetSurfaceElevDataSourceDirectory(ISimulation* sim, SurfaceElevDataSource surfaceElevSource)
		{
			return sim->GetSurfaceElevDataSourceDirectory(surfaceElevSource);
		}

		CRCCOVLIB_API void APIENTRY SetSurfaceAndTerrainDataSourcePairing(ISimulation* sim, bool usePairing)
		{
			return sim->SetSurfaceAndTerrainDataSourcePairing(usePairing);
		}

		CRCCOVLIB_API bool APIENTRY GetSurfaceAndTerrainDataSourcePairing(ISimulation* sim)
		{
			return sim->GetSurfaceAndTerrainDataSourcePairing();
		}

		CRCCOVLIB_API void APIENTRY SetSurfaceElevDataSourceSamplingMethod(ISimulation* sim, SurfaceElevDataSource surfaceElevSource, SamplingMethod samplingMethod)
		{
			return sim->SetSurfaceElevDataSourceSamplingMethod(surfaceElevSource, samplingMethod);
		}

		CRCCOVLIB_API SamplingMethod APIENTRY GetSurfaceElevDataSourceSamplingMethod(ISimulation* sim, SurfaceElevDataSource surfaceElevSource)
		{
			return sim->GetSurfaceElevDataSourceSamplingMethod(surfaceElevSource);
		}

		CRCCOVLIB_API bool APIENTRY AddCustomSurfaceElevData(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, int numHorizSamples, int numVertSamples, const float* surfaceElevData_meters, bool defineNoDataValue, float noDataValue)
		{
			return sim->AddCustomSurfaceElevData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, surfaceElevData_meters, defineNoDataValue, noDataValue);
		}

		CRCCOVLIB_API void APIENTRY ClearCustomSurfaceElevData(ISimulation* sim)
		{
			return sim->ClearCustomSurfaceElevData();
		}

		CRCCOVLIB_API double APIENTRY GetSurfaceElevation(ISimulation* sim, double latitude_degrees, double longitude_degrees, double noDataValue)
		{
			return sim->GetSurfaceElevation(latitude_degrees, longitude_degrees, noDataValue);
		}


		// Reception area parameters

		CRCCOVLIB_API void APIENTRY SetReceptionAreaCorners(ISimulation* sim, double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees)
		{
			return sim->SetReceptionAreaCorners(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees);
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaLowerLeftCornerLatitude(ISimulation* sim)
		{
			return sim->GetReceptionAreaLowerLeftCornerLatitude();
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaLowerLeftCornerLongitude(ISimulation* sim)
		{
			return sim->GetReceptionAreaLowerLeftCornerLongitude();
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaUpperRightCornerLatitude(ISimulation* sim)
		{
			return sim->GetReceptionAreaUpperRightCornerLatitude();
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaUpperRightCornerLongitude(ISimulation* sim)
		{
			return sim->GetReceptionAreaUpperRightCornerLongitude();
		}

		CRCCOVLIB_API void APIENTRY SetReceptionAreaNumHorizontalPoints(ISimulation* sim, int numPoints)
		{
			return sim->SetReceptionAreaNumHorizontalPoints(numPoints);
		}

		CRCCOVLIB_API int APIENTRY GetReceptionAreaNumHorizontalPoints(ISimulation* sim)
		{
			return sim->GetReceptionAreaNumHorizontalPoints();
		}

		CRCCOVLIB_API void APIENTRY SetReceptionAreaNumVerticalPoints(ISimulation* sim, int numPoints)
		{
			return sim->SetReceptionAreaNumVerticalPoints(numPoints);
		}

		CRCCOVLIB_API int APIENTRY GetReceptionAreaNumVerticalPoints(ISimulation* sim)
		{
			return sim->GetReceptionAreaNumVerticalPoints();
		}


		// Result type parameters

		CRCCOVLIB_API void APIENTRY SetResultType(ISimulation* sim, ResultType resultType)
		{
			return sim->SetResultType(resultType);
		}

		CRCCOVLIB_API ResultType APIENTRY GetResultType(ISimulation* sim)
		{
			return sim->GetResultType();
		}


		// Coverage display parameters for vector files (.mif and .kml)

		CRCCOVLIB_API void APIENTRY ClearCoverageDisplayFills(ISimulation* sim)
		{
			return sim->ClearCoverageDisplayFills();
		}

		CRCCOVLIB_API void APIENTRY AddCoverageDisplayFill(ISimulation* sim, double fromValue, double toValue, int rgbColor)
		{
			return sim->AddCoverageDisplayFill(fromValue, toValue, rgbColor);
		}

		CRCCOVLIB_API int APIENTRY GetCoverageDisplayNumFills(ISimulation* sim)
		{
			return sim->GetCoverageDisplayNumFills();
		}

		CRCCOVLIB_API double APIENTRY GetCoverageDisplayFillFromValue(ISimulation* sim, int index)
		{
			return sim->GetCoverageDisplayFillFromValue(index);
		}

		CRCCOVLIB_API double APIENTRY GetCoverageDisplayFillToValue(ISimulation* sim, int index)
		{
			return sim->GetCoverageDisplayFillToValue(index);
		}

		CRCCOVLIB_API int APIENTRY GetCoverageDisplayFillColor(ISimulation* sim, int index)
		{
			return sim->GetCoverageDisplayFillColor(index);
		}


		// Generating and accessing results

		CRCCOVLIB_API double APIENTRY GenerateReceptionPointResult(ISimulation* sim, double latitude_degrees, double longitude_degrees)
		{
			return sim->GenerateReceptionPointResult(latitude_degrees, longitude_degrees);
		}

		CRCCOVLIB_API ReceptionPointDetailedResult APIENTRY GenerateReceptionPointDetailedResult(ISimulation* sim, double latitude_degrees, double longitude_degrees)
		{
			return sim->GenerateReceptionPointDetailedResult(latitude_degrees, longitude_degrees);
		}

		CRCCOVLIB_API double APIENTRY GenerateProfileReceptionPointResult(ISimulation* sim, double latitude_degrees, double longitude_degrees, int numSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile, const double* surfaceElevProfile, const ITURadioClimaticZone* ituRadioClimaticZoneProfile)
		{
			return sim->GenerateProfileReceptionPointResult(latitude_degrees, longitude_degrees, numSamples, terrainElevProfile, landCoverClassMappedValueProfile, surfaceElevProfile, ituRadioClimaticZoneProfile);
		}

		CRCCOVLIB_API void APIENTRY GenerateReceptionAreaResults(ISimulation* sim)
		{
			return sim->GenerateReceptionAreaResults();
		}

		CRCCOVLIB_API int APIENTRY GetGenerateStatus(ISimulation* sim)
		{
			return sim->GetGenerateStatus();
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaResultValue(ISimulation* sim, int xIndex, int yIndex)
		{
			return sim->GetReceptionAreaResultValue(xIndex, yIndex);
		}

		CRCCOVLIB_API void APIENTRY SetReceptionAreaResultValue(ISimulation* sim, int xIndex, int yIndex, double value)
		{
			return sim->SetReceptionAreaResultValue(xIndex, yIndex, value);
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaResultValueAtLatLon(ISimulation* sim, double latitude_degrees, double longitude_degrees)
		{
			return sim->GetReceptionAreaResultValueAtLatLon(latitude_degrees, longitude_degrees);
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaResultLatitude(ISimulation* sim, int xIndex, int yIndex)
		{
			return sim->GetReceptionAreaResultLatitude(xIndex, yIndex);
		}

		CRCCOVLIB_API double APIENTRY GetReceptionAreaResultLongitude(ISimulation* sim, int xIndex, int yIndex)
		{
			return sim->GetReceptionAreaResultLongitude(xIndex, yIndex);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaResultsToTextFile(ISimulation* sim, const char* pathname, const char* resultsColumnName)
		{
			return sim->ExportReceptionAreaResultsToTextFile(pathname, resultsColumnName);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaResultsToMifFile(ISimulation* sim, const char* pathname, const char* resultsUnits)
		{
			return sim->ExportReceptionAreaResultsToMifFile(pathname, resultsUnits);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaResultsToKmlFile(ISimulation* sim, const char* pathname, double fillOpacity_percent, double lineOpacity_percent, const char* resultsUnits)
		{
			return sim->ExportReceptionAreaResultsToKmlFile(pathname, fillOpacity_percent, lineOpacity_percent, resultsUnits);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaResultsToBilFile(ISimulation* sim, const char* pathname)
		{
			return sim->ExportReceptionAreaResultsToBilFile(pathname);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaTerrainElevationToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero)
		{
			return sim->ExportReceptionAreaTerrainElevationToBilFile(pathname, numHorizontalPoints, numVerticalPoints, setNoDataToZero);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaLandCoverClassesToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool mapValues)
		{
			return sim->ExportReceptionAreaLandCoverClassesToBilFile(pathname, numHorizontalPoints, numVerticalPoints, mapValues);
		}

		CRCCOVLIB_API bool APIENTRY ExportReceptionAreaSurfaceElevationToBilFile(ISimulation* sim, const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero) 
		{
			return sim->ExportReceptionAreaSurfaceElevationToBilFile(pathname, numHorizontalPoints, numVerticalPoints, setNoDataToZero);
		}

		CRCCOVLIB_API CRCCOVLIB_API bool APIENTRY ExportProfilesToCsvFile(ISimulation* sim, const char* pathname, double latitude_degrees, double longitude_degrees)
		{
			return sim->ExportProfilesToCsvFile(pathname, latitude_degrees, longitude_degrees);
		}

#endif // ifdef CRCCOVLIB_WRAP
	}
}


