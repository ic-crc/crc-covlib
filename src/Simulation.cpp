/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#ifdef _MSC_VER
	#define _CRT_SECURE_NO_DEPRECATE
#endif
#include "Simulation.h"
#include "ITURP_DigitalMaps.h"
#include <algorithm>
#include <GeographicLib/Geodesic.hpp>
#include <climits>


using namespace Crc::Covlib;


Simulation::Simulation(void)
{
	pPropagModelId = LONGLEY_RICE;

	pRxAreaResults.SetBordersCoordinates(44.5, -76, 45.5, -74);
	pRxAreaResults.Clear(60, 60);

	pPrimaryTerrainElevSourceId = TERR_ELEV_NONE;
	pSecondaryTerrainElevSourceId = TERR_ELEV_NONE;
	pTertiaryTerrainElevSourceId = TERR_ELEV_NONE;
	pPrimaryLandCoverSourceId = LAND_COVER_NONE;
	pSecondaryLandCoverSourceId = LAND_COVER_NONE;
	pPrimarySurfElevSourceId = SURF_ELEV_NONE;
	pSecondarySurfElevSourceId = SURF_ELEV_NONE;
	pTertiarySurfElevSourceId = SURF_ELEV_NONE;
	pUpdateTerrainManagerSourcePtrs();
	pTerrainElevDataSamplingResKm = 0.100;

	pSetDefaultEsaWorldcoverToP1812Mappings();
	pSetDefaultEsaWorldcoverToP452v17Mappings();
	pSetDefaultEsaWorldcoverToP452v18Mappings();
	pSetDefaultNrcanLandCoverToP1812Mappings();
	pSetDefaultNrcanLandCoverToP452v17Mappings();
	pSetDefaultNrcanLandCoverToP452v18Mappings();

	pResultType = FIELD_STRENGTH_DBUVM;

	pGenerateStatus = STATUS_OK;

	ContourFillsEngine::FillZone fillZone;
	fillZone.m_fromValue = 45;
	fillZone.m_toValue = 60;
	fillZone.m_color = 0x5555FF;
	pCoverageFills.push_back(fillZone);
	fillZone.m_fromValue = 60;
	fillZone.m_toValue = 75;
	fillZone.m_color = 0x0000FF;
	pCoverageFills.push_back(fillZone);
	fillZone.m_fromValue = 75;
	fillZone.m_toValue = 100;
	fillZone.m_color = 0x000088;
	pCoverageFills.push_back(fillZone);
}

Simulation::Simulation(const Simulation& original)
{
	*this = original;
}

Simulation::~Simulation(void)
{
}

const Simulation& Simulation::operator=(const Simulation& original)
{
	if( &original != this )
	{
		pTx = original.pTx;
		pRx = original.pRx;
		pPropagModelId = original.pPropagModelId;
		pLongleyRiceModel = original.pLongleyRiceModel;
		pIturp1812Model = original.pIturp1812Model;
		pIturp452v17Model = original.pIturp452v17Model;
		pIturp452v18Model = original.pIturp452v18Model;
		pFreeSpaceModel = original.pFreeSpaceModel;
		pEHataModel = original.pEHataModel;
		pCrcMlplModel = original.pCrcMlplModel;
		pCrcPathObscuraModel = original.pCrcPathObscuraModel;
		pIturp2108Model = original.pIturp2108Model;
		pIturp2109Model = original.pIturp2109Model;
		pIturp676Model = original.pIturp676Model;
		pPrimaryTerrainElevSourceId = original.pPrimaryTerrainElevSourceId;
		pSecondaryTerrainElevSourceId = original.pSecondaryTerrainElevSourceId;
		pTertiaryTerrainElevSourceId = original.pTertiaryTerrainElevSourceId;
		pTerrainElevDataSamplingResKm = original.pTerrainElevDataSamplingResKm;
		pTerrainElevSrtm = original.pTerrainElevSrtm;
		pTerrainElevCustom = original.pTerrainElevCustom;
		pTerrainElevCdem = original.pTerrainElevCdem;
		pTerrainElevHrdemDtm = original.pTerrainElevHrdemDtm;
		pTerrainElevGeotiff = original.pTerrainElevGeotiff;
		pTerrainElevMrdemDtm = original.pTerrainElevMrdemDtm;
		pPrimaryLandCoverSourceId = original.pPrimaryLandCoverSourceId;
		pSecondaryLandCoverSourceId = original.pSecondaryLandCoverSourceId;
		pLandCoverEsaWorldcover = original.pLandCoverEsaWorldcover;
		pLandCoverGeotiff = original.pLandCoverGeotiff;
		pLandCoverCustom = original.pLandCoverCustom;
		pLandCoverNrcan = original.pLandCoverNrcan;
		pPrimarySurfElevSourceId = original.pPrimarySurfElevSourceId;
		pSecondarySurfElevSourceId = original.pSecondarySurfElevSourceId;
		pTertiarySurfElevSourceId = original.pTertiarySurfElevSourceId;
		pSurfElevSrtm = original.pSurfElevSrtm;
		pSurfElevCustom = original.pSurfElevCustom;
		pSurfElevCdsm = original.pSurfElevCdsm;
		pSurfElevHrdemDsm = original.pSurfElevHrdemDsm;
		pSurfElevGeotiff = original.pSurfElevGeotiff;
		pSurfElevMrdemDsm = original.pSurfElevMrdemDsm;
		pResultType = original.pResultType;
		pCoverageFills = original.pCoverageFills;
		pRxAreaResults = original.pRxAreaResults;
		pGenerateStatus = original.pGenerateStatus;
		pGenerator = original.pGenerator;
		pTopoManager = original.pTopoManager;

		pUpdateTerrainManagerSourcePtrs();
	}

	return *this;
}

void Simulation::Release()
{
	delete this;
}


// Transmitter parameters

void Simulation::SetTransmitterLocation(double latitude_degrees, double longitude_degrees)
{
	if( latitude_degrees > 90 || latitude_degrees < -90 )
		return;

	if( longitude_degrees > 180 || longitude_degrees < -180 )
		return;

	pTx.lat = latitude_degrees;
	pTx.lon = longitude_degrees;
}

double Simulation::GetTransmitterLatitude() const
{
	return pTx.lat;
}

double Simulation::GetTransmitterLongitude() const
{
	return pTx.lon;
}

void Simulation::SetTransmitterHeight(double height_meters)
{
	// Valid range for Longley-Rice is 0.5 - 3000 m
	// Valid range for ITU-R P.1812 is 1.0 - 3000 m
	// No validity range defined for ITU-R P.452 
	if( height_meters < 0.5 || height_meters > 3000 )
		return;

	pTx.rcagl = height_meters;
}

double Simulation::GetTransmitterHeight() const
{
	return pTx.rcagl;
}

void Simulation::SetTransmitterFrequency(double frequency_MHz)
{
	// Valid range for Longley-Rice (propag model) is              20    - 20000 MHz
	// Valid range for ITU-R P.1812 (propag model) is              30    -  6000 MHz
	// Valid range for ITU-R P.452  (propag model) is             100    - 50000 MHz
	// Valid range for Free Space   (propag model) is               0    -   N/A
	// Valid range for ITU-R P.2108 (clutter loss model) is         0.5  -    67 GHz
	// Valid range for ITU-R P.2109 (bldg entry loss model) is      0.08 -   100 GHz
	// Valid range for ITU-R P.676  (gaseous attenuation model) is  0    -  1000 GHz
	if( frequency_MHz <= 0 )
		return;

	pTx.freqMHz = frequency_MHz;
}

double Simulation::GetTransmitterFrequency() const
{
	return pTx.freqMHz;
}

void Simulation::SetTransmitterPower(double power_watts, PowerType powerType)
{
	if( power_watts < 0.0 )
		return;

	if( powerType != TPO && powerType != ERP && powerType != EIRP )
		return;

	pTx.powerType = powerType;
	pTx.power_watts = power_watts;
}

double Simulation::GetTransmitterPower(PowerType powerType) const
{
	switch (powerType)
	{
	case TPO:
		return pTx.tpo(Transmitter::PowerUnit::WATT);
	case ERP:
		return pTx.erp(Transmitter::PowerUnit::WATT);
	case EIRP:
		return pTx.eirp(Transmitter::PowerUnit::WATT);	
	default:
		return 0;
	}
}

void Simulation::SetTransmitterLosses(double losses_dB)
{
	if( losses_dB < 0.0 )
		return;

	pTx.losses_dB = losses_dB;
}

double Simulation::GetTransmitterLosses() const
{
	return pTx.losses_dB;
}

void Simulation::SetTransmitterPolarization(Polarization polarization)
{
	if( polarization != HORIZONTAL_POL && polarization != VERTICAL_POL )
		return;

	pTx.pol = polarization;
}

Polarization Simulation::GetTransmitterPolarization() const
{
	return pTx.pol;
}


// Receiver parameters

void Simulation::SetReceiverHeightAboveGround(double height_meters)
{
	// Valid range for Longley-Rice is 0.5 - 3000 m
	// Valid range for ITU-R P.1812 is 1.0 - 3000 m
	if( height_meters < 0.5 || height_meters > 3000 )
		return;
	pRx.heightAGL = height_meters;
}

double Simulation::GetReceiverHeightAboveGround() const
{
	return pRx.heightAGL;
}

void Simulation::SetReceiverLosses(double losses_dB)
{
	if( losses_dB < 0.0 )
		return;

	pRx.losses_dB = losses_dB;
}

double Simulation::GetReceiverLosses() const
{
	return pRx.losses_dB;
}


// Antenna parameters

void Simulation::ClearAntennaPatterns(Crc::Covlib::Terminal terminal, bool clearHorizontalPattern, bool clearVerticalPattern)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
	{
		if( clearHorizontalPattern )
			term->antPattern.ClearHPattern();
		if( clearVerticalPattern )
			term->antPattern.ClearVPattern();
	}
}

void Simulation::AddAntennaHorizontalPatternEntry(Terminal terminal, double azimuth_degrees, double gain_dB)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		term->antPattern.AddHPatternEntry(azimuth_degrees, gain_dB);
}

void Simulation::AddAntennaVerticalPatternEntry(Terminal terminal, int azimuth_degrees, double elevAngle_degrees, double gain_dB)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		term->antPattern.AddVPatternSliceEntry(azimuth_degrees, elevAngle_degrees, gain_dB);
}

void Simulation::SetAntennaElectricalTilt(Terminal terminal, double elecricalTilt_degrees)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		term->antPattern.SetElectricalTilt(elecricalTilt_degrees);
}

double Simulation::GetAntennaElectricalTilt(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->antPattern.GetElectricalTilt();

	return pTx.antPattern.GetElectricalTilt();
}

void Simulation::SetAntennaMechanicalTilt(Terminal terminal, double mechanicalTilt_degrees, double azimuth_degrees)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		term->antPattern.SetMechanicalTilt(azimuth_degrees, mechanicalTilt_degrees);
}
	
double Simulation::GetAntennaMechanicalTilt(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->antPattern.GetMechanicalTilt();

	return pTx.antPattern.GetMechanicalTilt();
}

double Simulation::GetAntennaMechanicalTiltAzimuth(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->antPattern.GetMechanicalTiltAzimuth();

	return pTx.antPattern.GetMechanicalTiltAzimuth();
}

void Simulation::SetAntennaMaximumGain(Terminal terminal, double maxGain_dBi)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		term->maxGain_dBi = maxGain_dBi;
}

double Simulation::GetAntennaMaximumGain(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->maxGain_dBi;

	return pTx.maxGain_dBi;
}

void Simulation::SetAntennaBearing(Terminal terminal, BearingReference bearingRef, double bearing_degrees)
{
CommTerminal* term;

	if( bearingRef < TRUE_NORTH || bearingRef > OTHER_TERMINAL )
		return;

	if( bearing_degrees < 0 || bearing_degrees > 360 )
		return;

	term = pGetTerminalObjPtr(terminal);
	if( term != nullptr )
	{
		term->bearingRef = bearingRef;
		term->bearingDeg = bearing_degrees;
	}
}

BearingReference Simulation::GetAntennaBearingReference(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->bearingRef;

	return pTx.bearingRef;
}

double Simulation::GetAntennaBearing(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->bearingDeg;

	return pTx.bearingDeg;
}

double Simulation::NormalizeAntennaHorizontalPattern(Crc::Covlib::Terminal terminal)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		return term->antPattern.NormalizeHPattern();

	return 0;
}

double Simulation::NormalizeAntennaVerticalPattern(Crc::Covlib::Terminal terminal)
{
CommTerminal* term = pGetTerminalObjPtr(terminal);

	if( term != nullptr )
		return term->antPattern.NormalizeVPattern();

	return 0;
}

void Simulation::SetAntennaPatternApproximationMethod(Terminal terminal, PatternApproximationMethod method)
{
static_assert((int)AntennaPattern::INTERPOLATION_ALGORITHM::H_PATTERN_ONLY   == (int)Crc::Covlib::PatternApproximationMethod::H_PATTERN_ONLY, "");
static_assert((int)AntennaPattern::INTERPOLATION_ALGORITHM::V_PATTERN_ONLY   == (int)Crc::Covlib::PatternApproximationMethod::V_PATTERN_ONLY, "");
static_assert((int)AntennaPattern::INTERPOLATION_ALGORITHM::SUMMING          == (int)Crc::Covlib::PatternApproximationMethod::SUMMING, "");
static_assert((int)AntennaPattern::INTERPOLATION_ALGORITHM::WEIGHTED_SUMMING == (int)Crc::Covlib::PatternApproximationMethod::WEIGHTED_SUMMING, "");
static_assert((int)AntennaPattern::INTERPOLATION_ALGORITHM::HYBRID           == (int)Crc::Covlib::PatternApproximationMethod::HYBRID, "");

CommTerminal* term;

	if( method < H_PATTERN_ONLY || method > HYBRID )
		return;

	term = pGetTerminalObjPtr(terminal);
	if( term != nullptr )
		term->patternApproxMethod = method;
}

PatternApproximationMethod Simulation::GetAntennaPatternApproximationMethod(Terminal terminal) const
{
const CommTerminal* term = pGetTerminalConstObjPtr(terminal);

	if( term != nullptr )
		return term->patternApproxMethod;

	return pTx.patternApproxMethod;
}

// azimuth_degrees: from 0 to 360 degrees
// elevAngle_degrees: from -90 (towards sky) to +90 (towards ground)
// return value in dBi
double Simulation::GetAntennaGain(Terminal terminal, double azimuth_degrees, double elevAngle_degrees, 
                                  double receiverLatitude_degrees/*=0*/, double receiverLongitude_degrees/*=0*/) const
{
double gain_dBi = 0;

	if( terminal == TRANSMITTER )
		gain_dBi = pGenerator.GetTransmitterAntennaGain(*this, azimuth_degrees, elevAngle_degrees, receiverLatitude_degrees, receiverLongitude_degrees);
	else if( terminal == RECEIVER )
		gain_dBi = pGenerator.GetReceiverAntennaGain(*this, azimuth_degrees, elevAngle_degrees, receiverLatitude_degrees, receiverLongitude_degrees);

	return gain_dBi;
}


// Propagation model selection

void Simulation::SetPropagationModel(PropagationModel propagationModel)
{
	if( propagationModel < LONGLEY_RICE || propagationModel > CRC_PATH_OBSCURA )
		return;
	pPropagModelId = propagationModel;
}

PropagationModel Simulation::GetPropagationModel() const
{
	return pPropagModelId;
}


// Longley-Rice propagation model parameters

void Simulation::SetLongleyRiceSurfaceRefractivity(double refractivity_NUnits)
{
	pLongleyRiceModel.SetSurfaceRefractivity(refractivity_NUnits);
}

double Simulation::GetLongleyRiceSurfaceRefractivity() const
{
	return pLongleyRiceModel.GetSurfaceRefractivity();
}

void Simulation::SetLongleyRiceGroundDielectricConst(double dielectricConst)
{
	pLongleyRiceModel.SetGroundDielectricConst(dielectricConst);
}

double Simulation::GetLongleyRiceGroundDielectricConst() const
{
	return pLongleyRiceModel.GetGroundDielectricConst();
}

void Simulation::SetLongleyRiceGroundConductivity(double groundConduct_Sm)
{
	pLongleyRiceModel.SetGroundConductivity(groundConduct_Sm);
}

double Simulation::GetLongleyRiceGroundConductivity() const
{
	return pLongleyRiceModel.GetGroundConductivity();
}

void Simulation::SetLongleyRiceClimaticZone(LRClimaticZone climaticZone)
{
	pLongleyRiceModel.SetClimaticZone(climaticZone);
}

LRClimaticZone Simulation::GetLongleyRiceClimaticZone() const
{
	return pLongleyRiceModel.GetClimaticZone();
}

void Simulation::SetLongleyRiceActivePercentageSet(LRPercentageSet percentageSet)
{
	pLongleyRiceModel.SetActivePercentageSet(percentageSet);
}

LRPercentageSet Simulation::GetLongleyRiceActivePercentageSet() const
{
	return pLongleyRiceModel.GetActivePercentageSet();
}

void Simulation::SetLongleyRiceTimePercentage(double time_percent)
{
	pLongleyRiceModel.SetTimePercentage(time_percent);
}

double Simulation::GetLongleyRiceTimePercentage() const
{
	return pLongleyRiceModel.GetTimePercentage();
}

void Simulation::SetLongleyRiceLocationPercentage(double location_percent)
{
	pLongleyRiceModel.SetLocationPercentage(location_percent);
}

double Simulation::GetLongleyRiceLocationPercentage() const
{
	return pLongleyRiceModel.GetLocationPercentage();
}

void Simulation::SetLongleyRiceSituationPercentage(double situation_percent)
{
	pLongleyRiceModel.SetSituationPercentage(situation_percent);
}

double Simulation::GetLongleyRiceSituationPercentage() const
{
	return pLongleyRiceModel.GetSituationPercentage();
}

void Simulation::SetLongleyRiceConfidencePercentage(double confidence_percent)
{
	pLongleyRiceModel.SetConfidencePercentage(confidence_percent);
}

double Simulation::GetLongleyRiceConfidencePercentage() const
{
	return pLongleyRiceModel.GetConfidencePercentage();
}

void Simulation::SetLongleyRiceReliabilityPercentage(double reliability_percent)
{
	pLongleyRiceModel.SetReliabilityPercentage(reliability_percent);
}

double Simulation::GetLongleyRiceReliabilityPercentage() const
{
	return pLongleyRiceModel.GetReliabilityPercentage();
}

void Simulation::SetLongleyRiceModeOfVariability(int mode)
{
	pLongleyRiceModel.SetModeOfVariability(mode);
}

int Simulation::GetLongleyRiceModeOfVariability() const
{
	return pLongleyRiceModel.GetModeOfVariability();
}


// ITU-R P.1812 propagation model parameters

void Simulation::SetITURP1812TimePercentage(double time_percent)
{
	pIturp1812Model.SetTimePercentage(time_percent);
}
	
double Simulation::GetITURP1812TimePercentage() const
{
	return pIturp1812Model.GetTimePercentage();
}

void Simulation::SetITURP1812LocationPercentage(double location_percent)
{
	pIturp1812Model.SetLocationPercentage(location_percent);
}
	
double Simulation::GetITURP1812LocationPercentage() const
{
	return pIturp1812Model.GetLocationPercentage();
}
	
void Simulation::SetITURP1812AverageRadioRefractivityLapseRate(double deltaN_Nunitskm)
{
	pIturp1812Model.SetAverageRadioRefractivityLapseRate(deltaN_Nunitskm);
}

double Simulation::GetITURP1812AverageRadioRefractivityLapseRate() const
{
	return pIturp1812Model.GetAverageRadioRefractivityLapseRate();
}

void Simulation::SetITURP1812SeaLevelSurfaceRefractivity(double N0_Nunits)
{
	pIturp1812Model.SetSeaLevelSurfaceRefractivity(N0_Nunits);
}

double Simulation::GetITURP1812SeaLevelSurfaceRefractivity() const
{
	return pIturp1812Model.GetSeaLevelSurfaceRefractivity();
}

void Simulation::SetITURP1812PredictionResolution(double resolution_meters)
{
	pIturp1812Model.SetPredictionResolution(resolution_meters);
}

double Simulation::GetITURP1812PredictionResolution() const
{
	return pIturp1812Model.GetPredictionResolution();
}
	
void Simulation::SetITURP1812RepresentativeClutterHeight(Crc::Covlib::P1812ClutterCategory clutterCategory, double reprHeight_meters)
{
	pIturp1812Model.SetClutterCategoryReprHeight(clutterCategory, reprHeight_meters);
}

double Simulation::GetITURP1812RepresentativeClutterHeight(Crc::Covlib::P1812ClutterCategory clutterCategory) const
{
	return pIturp1812Model.GetClutterCategoryReprHeight(clutterCategory);
}

void Simulation::SetITURP1812RadioClimaticZonesFile(const char* pathname)
{
	pTopoManager.SetRadioClimaticZonesFile(ITU_R_P_1812, pathname);
}
	
const char* Simulation::GetITURP1812RadioClimaticZonesFile() const
{
	return pTopoManager.GetRadioClimaticZonesFile(ITU_R_P_1812);
}

void Simulation::SetITURP1812LandCoverMappingType(P1812LandCoverMappingType mappingType)
{
	pIturp1812Model.SetLandCoverMappingType(mappingType);
}
	
P1812LandCoverMappingType Simulation::GetITURP1812LandCoverMappingType() const
{
	return pIturp1812Model.GetLandCoverMappingType();
}

void Simulation::SetITURP1812SurfaceProfileMethod(P1812SurfaceProfileMethod method)
{
	pIturp1812Model.SetSurfaceProfileMethod(method);
}

P1812SurfaceProfileMethod Simulation::GetITURP1812SurfaceProfileMethod() const
{
	return pIturp1812Model.GetSurfaceProfileMethod();
}


// ITU-R P.452 propagation model parameters

void Simulation::SetITURP452TimePercentage(double time_percent)
{
	pIturp452v17Model.SetTimePercentage(time_percent);
	pIturp452v18Model.SetTimePercentage(time_percent);
}

double Simulation::GetITURP452TimePercentage() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetTimePercentage();
	else
		return pIturp452v18Model.GetTimePercentage();
}

void Simulation::SetITURP452PredictionType(P452PredictionType predictionType)
{
	pIturp452v17Model.SetPredictionType(predictionType);
	pIturp452v18Model.SetPredictionType(predictionType);
}

P452PredictionType Simulation::GetITURP452PredictionType() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetPredictionType();
	else
		return pIturp452v18Model.GetPredictionType();
}

void Simulation::SetITURP452AverageRadioRefractivityLapseRate(double deltaN_Nunitskm)
{
	pIturp452v17Model.SetAverageRadioRefractivityLapseRate(deltaN_Nunitskm);
	pIturp452v18Model.SetAverageRadioRefractivityLapseRate(deltaN_Nunitskm);
}

double Simulation::GetITURP452AverageRadioRefractivityLapseRate() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetAverageRadioRefractivityLapseRate();
	else
		return pIturp452v18Model.GetAverageRadioRefractivityLapseRate();
}

void Simulation::SetITURP452SeaLevelSurfaceRefractivity(double N0_Nunits)
{
	pIturp452v17Model.SetSeaLevelSurfaceRefractivity(N0_Nunits);
	pIturp452v18Model.SetSeaLevelSurfaceRefractivity(N0_Nunits);
}

double Simulation::GetITURP452SeaLevelSurfaceRefractivity() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetSeaLevelSurfaceRefractivity();
	else
		return pIturp452v18Model.GetSeaLevelSurfaceRefractivity();
}

void Simulation::SetITURP452AirTemperature(double temperature_C)
{
	pIturp452v17Model.SetAirTemperature(temperature_C);
	pIturp452v18Model.SetAirTemperature(temperature_C);
}

double Simulation::GetITURP452AirTemperature() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetAirTemperature();
	else
		return pIturp452v18Model.GetAirTemperature();
}

void Simulation::SetITURP452AirPressure(double pressure_hPa)
{
	pIturp452v17Model.SetAirPressure(pressure_hPa);
	pIturp452v18Model.SetAirPressure(pressure_hPa);
}

double Simulation::GetITURP452AirPressure() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pIturp452v17Model.GetAirPressure();
	else
		return pIturp452v18Model.GetAirPressure();
}

void Simulation::SetITURP452RadioClimaticZonesFile(const char* pathname)
{
	pTopoManager.SetRadioClimaticZonesFile(ITU_R_P_452_V17, pathname);
	pTopoManager.SetRadioClimaticZonesFile(ITU_R_P_452_V18, pathname);
}

const char* Simulation::GetITURP452RadioClimaticZonesFile() const
{
	if( pPropagModelId == ITU_R_P_452_V17 )
		// use v17 as a safeguard here but the value should be the same regardless of version
		return pTopoManager.GetRadioClimaticZonesFile(ITU_R_P_452_V17);
	else
		return pTopoManager.GetRadioClimaticZonesFile(ITU_R_P_452_V18);
}

// specific to P.452-17 version

void Simulation::SetITURP452HeightGainModelClutterValue(P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam, double nominalValue)
{
ITURP452v17PropagModel::ClutterParams clutterParams;

	clutterParams = pIturp452v17Model.GetNominalClutterParams(clutterCategory);
	if( nominalParam == P452_NOMINAL_HEIGHT_M )
		clutterParams.nominalHeight_m = nominalValue;
	else if( nominalParam == P452_NOMINAL_DISTANCE_KM )
		clutterParams.nominalDist_km = nominalValue;
	pIturp452v17Model.SetNominalClutterParams(clutterCategory, clutterParams);
}

double Simulation::GetITURP452HeightGainModelClutterValue(P452HeightGainModelClutterCategory clutterCategory, P452HeightGainModelClutterParam nominalParam) const
{
ITURP452v17PropagModel::ClutterParams clutterParams;

	clutterParams = pIturp452v17Model.GetNominalClutterParams(clutterCategory);
	if( nominalParam == P452_NOMINAL_HEIGHT_M )
		return clutterParams.nominalHeight_m;
	else if( nominalParam == P452_NOMINAL_DISTANCE_KM )
		return clutterParams.nominalDist_km;
	return 0;
}

void Simulation::SetITURP452HeightGainModelMode(Terminal terminal, P452HeightGainModelMode mode)
{
	if( terminal == TRANSMITTER )
		pIturp452v17Model.SetTransmitterHeightGainModelMode(mode);
	else if( terminal == RECEIVER )
		pIturp452v17Model.SetReceiverHeightGainModelMode(mode);
}

P452HeightGainModelMode Simulation::GetITURP452HeightGainModelMode(Terminal terminal) const
{
	if( terminal == RECEIVER )
		return pIturp452v17Model.GetReceiverHeightGainModelMode();
	else
		return pIturp452v17Model.GetTransmitterHeightGainModelMode();
}

// specific to P.452-18 version

void Simulation::SetITURP452RepresentativeClutterHeight(P452ClutterCategory clutterCategory, double reprHeight_meters)
{
	pIturp452v18Model.SetClutterCategoryReprHeight(clutterCategory, reprHeight_meters);
}

double Simulation::GetITURP452RepresentativeClutterHeight(P452ClutterCategory clutterCategory) const
{
	return pIturp452v18Model.GetClutterCategoryReprHeight(clutterCategory);
}

void Simulation::SetITURP452LandCoverMappingType(P452LandCoverMappingType mappingType)
{
	pIturp452v18Model.SetLandCoverMappingType(mappingType);
}

P452LandCoverMappingType Simulation::GetITURP452LandCoverMappingType() const
{
	return pIturp452v18Model.GetLandCoverMappingType();
}

void Simulation::SetITURP452SurfaceProfileMethod(P452SurfaceProfileMethod method)
{
	pIturp452v18Model.SetSurfaceProfileMethod(method);
}

P452SurfaceProfileMethod Simulation::GetITURP452SurfaceProfileMethod() const
{
	return pIturp452v18Model.GetSurfaceProfileMethod();
}


// ITU-R P.2108 statistical clutter loss model for terrestrial paths

void Simulation::SetITURP2108TerrestrialStatModelActiveState(bool active)
{
	pIturp2108Model.SetActiveState(active);
}

bool Simulation::GetITURP2108TerrestrialStatModelActiveState() const
{
	return pIturp2108Model.IsActive();
}

void Simulation::SetITURP2108TerrestrialStatModelLocationPercentage(double location_percent)
{
	pIturp2108Model.SetLocationPercentage(location_percent);
}

double Simulation::GetITURP2108TerrestrialStatModelLocationPercentage() const
{
	return pIturp2108Model.GetLocationPercentage();
}

double Simulation::GetITURP2108TerrestrialStatModelLoss(double frequency_GHz, double distance_km) const
{
	return pIturp2108Model.CalcTerrestrialStatisticalLoss(frequency_GHz, distance_km);
}


// Extended Hata propagation model parameters

void Simulation::SetEHataClutterEnvironment(EHataClutterEnvironment clutterEnvironment)
{
	pEHataModel.SetClutterEnvironment(clutterEnvironment);
}

EHataClutterEnvironment Simulation::GetEHataClutterEnvironment() const
{
	return 	pEHataModel.GetClutterEnvironment();
}

void Simulation::SetEHataReliabilityPercentage(double percent)
{
	pEHataModel.SetReliabilityPercentage(percent);
}

double Simulation::GetEHataReliabilityPercentage() const
{
	return pEHataModel.GetReliabilityPercentage();
}


// ITU-R P.2109 building entry loss model

void Simulation::SetITURP2109ActiveState(bool active)
{
	pIturp2109Model.SetActiveState(active);
}
	
bool Simulation::GetITURP2109ActiveState() const
{
	return pIturp2109Model.IsActive();
}
	
void Simulation::SetITURP2109Probability(double probability_percent)
{
	pIturp2109Model.SetProbability(probability_percent);
}
	
double Simulation::GetITURP2109Probability() const
{
	return pIturp2109Model.GetProbability();
}
	
void Simulation::SetITURP2109DefaultBuildingType(P2109BuildingType buildingType)
{
	pIturp2109Model.SetDefaultBuildingType(buildingType);
}
	
P2109BuildingType Simulation::GetITURP2109DefaultBuildingType() const
{
	return pIturp2109Model.GetDefaultBuildingType();
}
	
double Simulation::GetITURP2109BuildingEntryLoss(double frequency_GHz, double elevAngle_degrees) const
{
	return pIturp2109Model.CalcBuildingEntryLoss(frequency_GHz, elevAngle_degrees);
}


// ITU-R P.676 gaseous attenuation model for terrestrial paths
	
void Simulation::SetITURP676TerrPathGaseousAttenuationActiveState(bool active, double atmPressure_hPa/*=AUTOMATIC*/, double temperature_C/*=AUTOMATIC*/,
                                                                  double waterVapourDensity_gm3/*=AUTOMATIC*/)
{
	pIturp676Model.SetActiveState(active);
	if( active == true )
	{
		pIturp676Model.SetAtmosphericPressure(atmPressure_hPa);
		if( std::isnan(temperature_C) ) // if "AUTOMATIC"
			pIturp676Model.SetTemperature(temperature_C);
		else
			pIturp676Model.SetTemperature(temperature_C+273.15);
		pIturp676Model.SetWaterVapourDensity(waterVapourDensity_gm3);
	}
}

bool Simulation::GetITURP676TerrPathGaseousAttenuationActiveState() const
{
	return pIturp676Model.IsActive();
}
	
double Simulation::GetITURP676GaseousAttenuation(double frequency_GHz, double atmPressure_hPa/*=1013.25*/, double temperature_C/*=15*/,
                                                 double waterVapourDensity_gm3/*=7.5*/) const
{
	return pIturp676Model.GaseousAttenuationPerKm(frequency_GHz, atmPressure_hPa, temperature_C+273.15, waterVapourDensity_gm3);
}


// ITU digial maps

double Simulation::GetITUDigitalMapValue(Crc::Covlib::ITUDigitalMap map, double latitude_degrees, double longitude_degrees) const
{
	if( latitude_degrees < -90 || latitude_degrees > 90 || longitude_degrees < -180 || longitude_degrees > 180 )
		return 0;

	switch (map)
	{
	case ITU_MAP_DN50:
		return ITURP_DigitalMaps::DN50(latitude_degrees, longitude_degrees);
	case ITU_MAP_N050:
		return ITURP_DigitalMaps::N050(latitude_degrees, longitude_degrees);
	case ITU_MAP_T_ANNUAL:
		return ITURP_DigitalMaps::T_Annual(latitude_degrees, longitude_degrees);
	case ITU_MAP_SURFWV_50:
		return ITURP_DigitalMaps::Surfwv_50(latitude_degrees, longitude_degrees);
	default:
		return 0;
	}
}


// Terrain elevation data parameters

void Simulation::SetPrimaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource)
{
	if( pIsTerrainElevDataSourceValid(terrainElevSource) == false )
		return;
	pPrimaryTerrainElevSourceId = terrainElevSource;
	pUpdateTerrainManagerSourcePtrs();
}
	
TerrainElevDataSource Simulation::GetPrimaryTerrainElevDataSource() const
{
	return pPrimaryTerrainElevSourceId;
}

void Simulation::SetSecondaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource)
{
	if( pIsTerrainElevDataSourceValid(terrainElevSource) == false )
		return;
	pSecondaryTerrainElevSourceId = terrainElevSource;
	pUpdateTerrainManagerSourcePtrs();
}

TerrainElevDataSource Simulation::GetSecondaryTerrainElevDataSource() const
{
	return pSecondaryTerrainElevSourceId;
}

void Simulation::SetTertiaryTerrainElevDataSource(TerrainElevDataSource terrainElevSource)
{
	if( pIsTerrainElevDataSourceValid(terrainElevSource) == false )
		return;
	pTertiaryTerrainElevSourceId = terrainElevSource;
	pUpdateTerrainManagerSourcePtrs();
}
	
TerrainElevDataSource Simulation::GetTertiaryTerrainElevDataSource() const
{
	return pTertiaryTerrainElevSourceId;
}

void Simulation::SetTerrainElevDataSourceDirectory(TerrainElevDataSource terrainElevSource, const char* directory, 
                                                   bool useIndexFile/*=false*/, bool overwriteIndexFile/*=false*/)
{
	switch (terrainElevSource)
	{
	case TERR_ELEV_SRTM:
		pTerrainElevSrtm.SetDirectory(directory);
		break;
	case TERR_ELEV_NRCAN_CDEM:
		pTerrainElevCdem.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case TERR_ELEV_NRCAN_HRDEM_DTM:
		pTerrainElevHrdemDtm.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case TERR_ELEV_GEOTIFF:
		pTerrainElevGeotiff.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case TERR_ELEV_NRCAN_MRDEM_DTM:
		pTerrainElevMrdemDtm.SetDirectory(directory, false, false); // has online 1 file, so no index use
		break;
	default:
		break;
	}
}

const char* Simulation::GetTerrainElevDataSourceDirectory(TerrainElevDataSource terrainElevSource) const
{
	switch (terrainElevSource)
	{
	case TERR_ELEV_SRTM:
		return pTerrainElevSrtm.GetDirectory();
	case TERR_ELEV_NRCAN_CDEM:
		return pTerrainElevCdem.GetDirectory();
	case TERR_ELEV_NRCAN_HRDEM_DTM:
		return pTerrainElevHrdemDtm.GetDirectory();
	case TERR_ELEV_GEOTIFF:
		return pTerrainElevGeotiff.GetDirectory();
	case TERR_ELEV_NRCAN_MRDEM_DTM:
		return pTerrainElevMrdemDtm.GetDirectory();
	default:
		break;
	}
	return nullptr;
}

bool Simulation::AddCustomTerrainElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees,
                                          int numHorizSamples, int numVertSamples, const float* terrainElevData_meters, bool defineNoDataValue/*=false*/, float noDataValue/*=0*/)
{
	if( numHorizSamples < 2 || numVertSamples < 2 )
		return false;

	return pTerrainElevCustom.AddData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees,
	                                  (unsigned int)numHorizSamples, (unsigned int)numVertSamples, terrainElevData_meters, defineNoDataValue, noDataValue);
}

void Simulation::ClearCustomTerrainElevData()
{
	pTerrainElevCustom.ClearData();
}

void Simulation::SetTerrainElevDataSourceSamplingMethod(TerrainElevDataSource terrainElevSource, SamplingMethod samplingMethod)
{
TerrainElevSource* source;

	source = pGetTerrainElevSourceObjPtr(terrainElevSource);
	if( source != nullptr )
	{
		if( samplingMethod == BILINEAR_INTERPOLATION )
			source->SetInterpolationType(TerrainElevSource::BILINEAR);
		else if( samplingMethod == NEAREST_NEIGHBOR )
			source->SetInterpolationType(TerrainElevSource::CLOSEST_POINT);
	}
}

SamplingMethod Simulation::GetTerrainElevDataSourceSamplingMethod(TerrainElevDataSource terrainElevSource) const
{
const TerrainElevSource* source;

	source = pGetTerrainElevSourceConstObjPtr(terrainElevSource);
	if( source != nullptr )
		if( source->GetInterpolationType() == TerrainElevSource::CLOSEST_POINT )
			return NEAREST_NEIGHBOR;

	return BILINEAR_INTERPOLATION;
}

void Simulation::SetTerrainElevDataSamplingResolution(double samplingResolution_meters)
{
	if( samplingResolution_meters < 0.5 )
		return;

	pTerrainElevDataSamplingResKm = samplingResolution_meters / 1000.0;
}

double Simulation::GetTerrainElevDataSamplingResolution() const
{
	return pTerrainElevDataSamplingResKm * 1000.0;
}

double Simulation::GetTerrainElevation(double latitude_degrees, double longitude_degrees, double noDataValue/*=0*/)
{
bool success;
float terrainElevation_meters = noDataValue;

	success = pTopoManager.GetTerrainElevation(latitude_degrees, longitude_degrees, &terrainElevation_meters);
	if( success )
		return terrainElevation_meters;
	else
		return noDataValue;
}

// Returns required size if buffer too small, otherwise returns items written
int Simulation::GetTerrainElevationProfile(double latitude_degrees, double longitude_degrees, double* outputProfile, int sizeOutputProfile)
{
std::vector<std::pair<double,double>> latLonProfile;
std::vector<double> distKmProfile;

	pTopoManager.GetLatLonProfileByRes(pTx.lat, pTx.lon, latitude_degrees, longitude_degrees, pTerrainElevDataSamplingResKm,
	                                   TopographicDataManager::ITU_GREAT_CIRCLE, &latLonProfile, &distKmProfile);

	size_t latLonSize = latLonProfile.size();
	if( latLonSize <= static_cast<size_t>(INT_MAX) )
	{
		int requiredSize = static_cast<int>(latLonSize);
		if( sizeOutputProfile >= requiredSize )
		{
			std::vector<double> terrainElevProfile;
			pTopoManager.GetTerrainElevProfile(latLonProfile, &terrainElevProfile);
			std::memcpy(outputProfile, terrainElevProfile.data(), latLonSize*sizeof(double));
		}
		return requiredSize;
	}
	else
		return -1;
}


// Land cover data parameters

void Simulation::SetPrimaryLandCoverDataSource(LandCoverDataSource landCoverSource)
{
	if( pIsLandCoverDataSourceValid(landCoverSource) == false )
		return;
	pPrimaryLandCoverSourceId = landCoverSource;
	pUpdateTerrainManagerSourcePtrs();
}
	
LandCoverDataSource Simulation::GetPrimaryLandCoverDataSource() const
{
	return pPrimaryLandCoverSourceId;
}

void Simulation::SetSecondaryLandCoverDataSource(LandCoverDataSource landCoverSource)
{
	if( pIsLandCoverDataSourceValid(landCoverSource) == false )
		return;
	pSecondaryLandCoverSourceId = landCoverSource;
	pUpdateTerrainManagerSourcePtrs();
}
	
LandCoverDataSource Simulation::GetSecondaryLandCoverDataSource() const
{
	return pSecondaryLandCoverSourceId;
}

void Simulation::SetLandCoverDataSourceDirectory(Crc::Covlib::LandCoverDataSource landCoverSource, const char* directory, bool useIndexFile, bool overwriteIndexFile)
{
	switch (landCoverSource)
	{
	case LAND_COVER_ESA_WORLDCOVER:
		pLandCoverEsaWorldcover.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case LAND_COVER_GEOTIFF:
		pLandCoverGeotiff.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case LAND_COVER_NRCAN:
		pLandCoverNrcan.SetDirectory(directory, false, false); // only 1 file so don't use index file
		break;
	default:
		break;
	}
}
	
const char* Simulation::GetLandCoverDataSourceDirectory(Crc::Covlib::LandCoverDataSource landCoverSource) const
{
	switch (landCoverSource)
	{
	case LAND_COVER_ESA_WORLDCOVER:
		return pLandCoverEsaWorldcover.GetDirectory();
	case LAND_COVER_GEOTIFF:
		return pLandCoverGeotiff.GetDirectory();
	case LAND_COVER_NRCAN:
		return pLandCoverNrcan.GetDirectory();
	default:
		break;
	}
	return nullptr;
}

bool Simulation::AddCustomLandCoverData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees, 
                                        int numHorizSamples, int numVertSamples, const short* landCoverData, bool defineNoDataValue/*=false*/, short noDataValue/*=0*/)
{
	if( numHorizSamples < 2 || numVertSamples < 2 )
		return false;

	return pLandCoverCustom.AddData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees,
	                                (unsigned int)numHorizSamples, (unsigned int)numVertSamples, landCoverData, defineNoDataValue, noDataValue);
}

void Simulation::ClearCustomLandCoverData()
{
	pLandCoverCustom.ClearData();
}

int Simulation::GetLandCoverClass(double latitude_degrees, double longitude_degrees)
{
int landCoverClass = -1;

	if( pTopoManager.GetLandCover(latitude_degrees, longitude_degrees, &landCoverClass) == true )
		return landCoverClass;
	else
		return -1;
}

// Returns required size if buffer too small, otherwise returns items written
int Simulation::GetLandCoverClassProfile(double latitude_degrees, double longitude_degrees, int* outputProfile, int sizeOutputProfile)
{
std::vector<std::pair<double,double>> latLonProfile;
std::vector<double> distKmProfile;

	pTopoManager.GetLatLonProfileByRes(pTx.lat, pTx.lon, latitude_degrees, longitude_degrees, pTerrainElevDataSamplingResKm,
	                                   TopographicDataManager::ITU_GREAT_CIRCLE, &latLonProfile, &distKmProfile);

	size_t latLonSize = latLonProfile.size();
	if( latLonSize <= static_cast<size_t>(INT_MAX) )
	{
		int requiredSize = static_cast<int>(latLonSize);
		if( sizeOutputProfile >= requiredSize )
		{
			std::vector<int> landCoverClassProfile;
			pTopoManager.GetLandCoverProfile(latLonProfile, &landCoverClassProfile);
			std::memcpy(outputProfile, landCoverClassProfile.data(), latLonSize*sizeof(int));
		}
		return requiredSize;
	}
	else
		return -1;
}

int Simulation::GetLandCoverClassMappedValue(double latitude_degrees, double longitude_degrees, PropagationModel propagationModel)
{
int modelValue = -1;

	if( pTopoManager.GetLandCoverMappedValue(latitude_degrees, longitude_degrees, propagationModel, &modelValue) == true )
		return modelValue;
	else
		return pGetPropagModelPtr(propagationModel)->DefaultMappedLandCoverValue();
}

// Returns required size if buffer too small, otherwise returns items written
int Simulation::GetLandCoverClassMappedValueProfile(double latitude_degrees, double longitude_degrees, PropagationModel propagationModel, int* outputProfile, int sizeOutputProfile)
{
std::vector<std::pair<double,double>> latLonProfile;
std::vector<double> distKmProfile;

	pTopoManager.GetLatLonProfileByRes(pTx.lat, pTx.lon, latitude_degrees, longitude_degrees, pTerrainElevDataSamplingResKm,
	                                   TopographicDataManager::ITU_GREAT_CIRCLE, &latLonProfile, &distKmProfile);

	size_t latLonSize = latLonProfile.size();
	if( latLonSize <= static_cast<size_t>(INT_MAX) )
	{
		int requiredSize = static_cast<int>(latLonSize);
		if( sizeOutputProfile >= requiredSize )
		{
			std::vector<int> landCoverClassMappedValueProfile;
			int defaultValue = pGetPropagModelPtr(propagationModel)->DefaultMappedLandCoverValue();
			pTopoManager.GetMappedLandCoverProfile(latLonProfile, propagationModel, defaultValue, &landCoverClassMappedValueProfile);
			std::memcpy(outputProfile, landCoverClassMappedValueProfile.data(), latLonSize*sizeof(int));
		}
		return requiredSize;
	}
	else
		return -1;
}

void Simulation::SetLandCoverClassMapping(LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel, int modelValue)
{
LandCoverSource* source;

	source = pGetLandCoverSourceObjPtr(landCoverSource);
	if( source != nullptr )
		source->SetMapping(sourceClass, propagationModel, modelValue);
}

int Simulation::GetLandCoverClassMapping(LandCoverDataSource landCoverSource, int sourceClass, PropagationModel propagationModel) const
{
const LandCoverSource* source;
int modelValue;

	source = pGetLandCoverSourceConstObjPtr(landCoverSource);
	if( source != nullptr )
		if( source->GetMapping(sourceClass, propagationModel, &modelValue) == true )
			return modelValue;

	return -1;
}

void Simulation::SetDefaultLandCoverClassMapping(LandCoverDataSource landCoverSource, PropagationModel propagationModel, int modelValue)
{
LandCoverSource* source;

	source = pGetLandCoverSourceObjPtr(landCoverSource);
	if( source != nullptr )
		source->SetDefaultMapping(propagationModel, modelValue);
}

int Simulation::GetDefaultLandCoverClassMapping(LandCoverDataSource landCoverSource, PropagationModel propagationModel) const
{
const LandCoverSource* source;
int modelValue;

	source = pGetLandCoverSourceConstObjPtr(landCoverSource);
	if( source != nullptr )
		if( source->GetDefaultMapping(propagationModel, &modelValue) == true )
			return modelValue;

	return -1;
}

void Simulation::ClearLandCoverClassMappings(LandCoverDataSource landCoverSource, PropagationModel propagationModel)
{
LandCoverSource* source;

	source = pGetLandCoverSourceObjPtr(landCoverSource);
	if( source != nullptr )
		source->ClearMappings(propagationModel);
}


// Surface elevation data parameters

void Simulation::SetPrimarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource)
{
	if( pIsSurfaceElevDataSourceValid(surfaceElevSource) == false )
		return;
	pPrimarySurfElevSourceId = surfaceElevSource;
	pUpdateTerrainManagerSourcePtrs();
}

SurfaceElevDataSource Simulation::GetPrimarySurfaceElevDataSource() const
{
	return pPrimarySurfElevSourceId;
}

void Simulation::SetSecondarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource)
{
	if( pIsSurfaceElevDataSourceValid(surfaceElevSource) == false )
		return;
	pSecondarySurfElevSourceId = surfaceElevSource;
	pUpdateTerrainManagerSourcePtrs();
}

SurfaceElevDataSource Simulation::GetSecondarySurfaceElevDataSource() const
{
	return pSecondarySurfElevSourceId;
}

void Simulation::SetTertiarySurfaceElevDataSource(SurfaceElevDataSource surfaceElevSource)
{
	if( pIsSurfaceElevDataSourceValid(surfaceElevSource) == false )
		return;
	pTertiarySurfElevSourceId = surfaceElevSource;
	pUpdateTerrainManagerSourcePtrs();
}

SurfaceElevDataSource Simulation::GetTertiarySurfaceElevDataSource() const
{
	return pTertiarySurfElevSourceId;
}

void Simulation::SetSurfaceElevDataSourceDirectory(SurfaceElevDataSource surfaceElevSource, const char* directory, bool useIndexFile/*=false*/, bool overwriteIndexFile/*=false*/)
{
	switch (surfaceElevSource)
	{
	case SURF_ELEV_SRTM:
		pSurfElevSrtm.SetDirectory(directory);
		break;
	case SURF_ELEV_NRCAN_CDSM:
		pSurfElevCdsm.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case SURF_ELEV_NRCAN_HRDEM_DSM:
		pSurfElevHrdemDsm.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case SURF_ELEV_GEOTIFF:
		pSurfElevGeotiff.SetDirectory(directory, useIndexFile, overwriteIndexFile);
		break;
	case SURF_ELEV_NRCAN_MRDEM_DSM:
		pSurfElevMrdemDsm.SetDirectory(directory, false, false); // only 1 file so don't use index
		break;
	default:
		break;
	}
}

const char* Simulation::GetSurfaceElevDataSourceDirectory(SurfaceElevDataSource surfaceElevSource) const
{
	switch (surfaceElevSource)
	{
	case SURF_ELEV_SRTM:
		return pSurfElevSrtm.GetDirectory();
	case SURF_ELEV_NRCAN_CDSM:
		return pSurfElevCdsm.GetDirectory();
	case SURF_ELEV_NRCAN_HRDEM_DSM:
		return pSurfElevHrdemDsm.GetDirectory();
	case SURF_ELEV_GEOTIFF:
		return pSurfElevGeotiff.GetDirectory();
	case SURF_ELEV_NRCAN_MRDEM_DSM:
		return pSurfElevMrdemDsm.GetDirectory();
	default:
		break;
	}
	return nullptr;
}

void Simulation::SetSurfaceAndTerrainDataSourcePairing(bool usePairing)
{
	pTopoManager.UsePairedTerrainAndSurfaceElevSources(usePairing);
	pUpdateTerrainManagerSourcePtrs();
}

bool Simulation::GetSurfaceAndTerrainDataSourcePairing() const
{
	return pTopoManager.UsePairedTerrainAndSurfaceElevSources();
}

void Simulation::SetSurfaceElevDataSourceSamplingMethod(SurfaceElevDataSource surfaceElevSource, SamplingMethod samplingMethod)
{
SurfaceElevSource* source;

	source = pGetSurfaceElevSourceObjPtr(surfaceElevSource);
	if( source != nullptr )
	{
		if( samplingMethod == BILINEAR_INTERPOLATION )
			source->SetInterpolationType(SurfaceElevSource::BILINEAR);
		else if( samplingMethod == NEAREST_NEIGHBOR )
			source->SetInterpolationType(SurfaceElevSource::CLOSEST_POINT);
	}
}

SamplingMethod Simulation::GetSurfaceElevDataSourceSamplingMethod(SurfaceElevDataSource surfaceElevSource) const
{
const SurfaceElevSource* source;

	source = pGetSurfaceElevSourceConstObjPtr(surfaceElevSource);
	if( source != nullptr )
		if( source->GetInterpolationType() == SurfaceElevSource::CLOSEST_POINT )
			return NEAREST_NEIGHBOR;

	return BILINEAR_INTERPOLATION;
}

bool Simulation::AddCustomSurfaceElevData(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees,
                                          double upperRightCornerLat_degrees, double upperRightCornerLon_degrees,
                                          int numHorizSamples, int numVertSamples, const float* surfaceElevData_meters,
                                          bool defineNoDataValue/*=false*/, float noDataValue/*=0*/)
{
	if( numHorizSamples < 2 || numVertSamples < 2 )
		return false;

	return pSurfElevCustom.AddData(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees,
	                               (unsigned int)numHorizSamples, (unsigned int)numVertSamples, surfaceElevData_meters, defineNoDataValue, noDataValue);
}

void Simulation::ClearCustomSurfaceElevData()
{
	pSurfElevCustom.ClearData();
}

double Simulation::GetSurfaceElevation(double latitude_degrees, double longitude_degrees, double noDataValue/*=0*/)
{
bool success;
float surfaceElevation_meters = noDataValue;

	success = pTopoManager.GetSurfaceElevation(latitude_degrees, longitude_degrees, &surfaceElevation_meters);
	if( success )
		return surfaceElevation_meters;
	else
		return noDataValue;
}

// Returns required size if buffer too small, otherwise returns items written
int Simulation::GetSurfaceElevationProfile(double latitude_degrees, double longitude_degrees, double* outputProfile, int sizeOutputProfile)
{
std::vector<std::pair<double,double>> latLonProfile;
std::vector<double> distKmProfile;

	pTopoManager.GetLatLonProfileByRes(pTx.lat, pTx.lon, latitude_degrees, longitude_degrees, pTerrainElevDataSamplingResKm,
	                                   TopographicDataManager::ITU_GREAT_CIRCLE, &latLonProfile, &distKmProfile);

	size_t latLonSize = latLonProfile.size();
	if( latLonSize <= static_cast<size_t>(INT_MAX) )
	{
		int requiredSize = static_cast<int>(latLonSize);
		if( sizeOutputProfile >= requiredSize )
		{
			std::vector<double> surfaceElevProfile;
			pTopoManager.GetSurfaceElevProfile(latLonProfile, &surfaceElevProfile);
			std::memcpy(outputProfile, surfaceElevProfile.data(), latLonSize*sizeof(double));
		}
		return requiredSize;
	}
	else
		return -1;
}


// Reception area parameters

void Simulation::SetReceptionAreaCorners(double lowerLeftCornerLat_degrees, double lowerLeftCornerLon_degrees, double upperRightCornerLat_degrees, double upperRightCornerLon_degrees)
{
	if( lowerLeftCornerLat_degrees > 90 || lowerLeftCornerLat_degrees < -90 )
		return;
	if( lowerLeftCornerLon_degrees > 180 || lowerLeftCornerLon_degrees < -180 )
		return;
	if( upperRightCornerLat_degrees > 90 || upperRightCornerLat_degrees < -90 )
		return;
	if( upperRightCornerLon_degrees > 180 || upperRightCornerLon_degrees < -180 )
		return;

	pRxAreaResults.SetBordersCoordinates(lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees,
										 upperRightCornerLat_degrees, upperRightCornerLon_degrees);
}

double Simulation::GetReceptionAreaLowerLeftCornerLatitude() const
{
double minLat, minLon, maxLat, maxLon;

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	return minLat;
}

double Simulation::GetReceptionAreaLowerLeftCornerLongitude() const
{
double minLat, minLon, maxLat, maxLon;

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	return minLon;
}

double Simulation::GetReceptionAreaUpperRightCornerLatitude() const
{
double minLat, minLon, maxLat, maxLon;

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	return maxLat;
}

double Simulation::GetReceptionAreaUpperRightCornerLongitude() const
{
double minLat, minLon, maxLat, maxLon;

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	return maxLon;
}

void Simulation::SetReceptionAreaNumHorizontalPoints(int numPoints)
{
	if( numPoints < 2 )
		return;

	if( (unsigned int)numPoints == pRxAreaResults.SizeX() )
		return;

	pRxAreaResults.Clear((unsigned int)numPoints, pRxAreaResults.SizeY());
}

int Simulation::GetReceptionAreaNumHorizontalPoints() const
{
	return (int)pRxAreaResults.SizeX();
}

void Simulation::SetReceptionAreaNumVerticalPoints(int numPoints)
{
	if( numPoints < 2 )
		return;

	if( (unsigned int)numPoints == pRxAreaResults.SizeY() )
		return;

	pRxAreaResults.Clear(pRxAreaResults.SizeX(), (unsigned int)numPoints);
}

int Simulation::GetReceptionAreaNumVerticalPoints() const
{
	return (int)pRxAreaResults.SizeY();
}


// Result type parameters

void Simulation::SetResultType(ResultType resultType)
{
	if( pResultType < FIELD_STRENGTH_DBUVM || pResultType > RECEIVED_POWER_DBM )
		return;
	pResultType = resultType;
}

ResultType Simulation::GetResultType() const
{
	return pResultType;
}


// Coverage display parameters for vector files (.mif and .kml)

void Simulation::ClearCoverageDisplayFills()
{
	pCoverageFills.clear();
}

void Simulation::AddCoverageDisplayFill(double fromValue, double toValue, int rgbColor)
{
ContourFillsEngine::FillZone fillZone;

	if( toValue >= fromValue )
	{
		fillZone.m_fromValue = fromValue;
		fillZone.m_toValue = toValue;
	}
	else
	{
		fillZone.m_fromValue = toValue;
		fillZone.m_toValue = fromValue;
	}
	fillZone.m_color = (unsigned int) rgbColor;
	pCoverageFills.push_back(fillZone);
}

int Simulation::GetCoverageDisplayNumFills() const
{
	return (int) pCoverageFills.size();
}

double Simulation::GetCoverageDisplayFillFromValue(int index) const
{
	if( index >= 0 && index < GetCoverageDisplayNumFills() )
		return pCoverageFills[(size_t)index].m_fromValue;
	return 0;
}

double Simulation::GetCoverageDisplayFillToValue(int index) const
{
	if( index >= 0 && index < GetCoverageDisplayNumFills() )
		return pCoverageFills[(size_t)index].m_toValue;
	return 0;
}

int Simulation::GetCoverageDisplayFillColor(int index) const
{
	if( index >= 0 && index < GetCoverageDisplayNumFills() )
		return (int) pCoverageFills[(size_t)index].m_color;
	return 0;
}


// Generating and accessing results

double Simulation::GenerateReceptionPointResult(double latitude_degrees, double longitude_degrees)
{
	return pGenerator.RunPointCalculation(*this, latitude_degrees, longitude_degrees);
}

ReceptionPointDetailedResult Simulation::GenerateReceptionPointDetailedResult(double latitude_degrees, double longitude_degrees)
{
ReceptionPointDetailedResult detailedResult = {0,0,0,0,0,0,0,0,0,0,0};

	pGenerator.RunPointCalculation(*this, latitude_degrees, longitude_degrees, 0, nullptr, nullptr, nullptr, nullptr, &detailedResult);

	return detailedResult;
}

double Simulation::GenerateProfileReceptionPointResult(double latitude_degrees, double longitude_degrees, int numSamples,
                                                       const double* terrainElevProfile,
                                                       const int* landCoverClassMappedValueProfile/*=nullptr*/,
                                                       const double* surfaceElevProfile/*=nullptr*/,
                                                       const ITURadioClimaticZone* ituRadioClimaticZoneProfile/*=nullptr*/)
{
	if( numSamples < 0 )
		numSamples = 0;

	return pGenerator.RunPointCalculation(*this, latitude_degrees, longitude_degrees, static_cast<unsigned int>(numSamples), terrainElevProfile,
	                                      landCoverClassMappedValueProfile, surfaceElevProfile, ituRadioClimaticZoneProfile);
}

ReceptionPointDetailedResult Simulation::GenerateProfileReceptionPointDetailedResult(double latitude_degrees, double longitude_degrees, int numSamples,
                                                                                     const double* terrainElevProfile,
																					 const int* landCoverClassMappedValueProfile/*=nullptr*/,
																					 const double* surfaceElevProfile/*=nullptr*/,
																					 const ITURadioClimaticZone* ituRadioClimaticZoneProfile/*=nullptr*/)
{
ReceptionPointDetailedResult detailedResult = {0,0,0,0,0,0,0,0,0,0,0};

	if( numSamples < 0 )
		numSamples = 0;

	pGenerator.RunPointCalculation(*this, latitude_degrees, longitude_degrees, static_cast<unsigned int>(numSamples), terrainElevProfile,
	                               landCoverClassMappedValueProfile, surfaceElevProfile, ituRadioClimaticZoneProfile, &detailedResult);

	return detailedResult;
}

void Simulation::GenerateReceptionAreaResults()
{
	pGenerator.RunAreaCalculation(*this);
}

int Simulation::GetGenerateStatus() const
{
	return pGenerateStatus;
}

double Simulation::GetReceptionAreaResultValue(int xIndex, int yIndex) const
{
	return pRxAreaResults.GetData((unsigned int)xIndex, (unsigned int)yIndex);
}

void Simulation::SetReceptionAreaResultValue(int xIndex, int yIndex, double value)
{
	pRxAreaResults.SetData((unsigned int)xIndex, (unsigned int)yIndex, (float)value);
}

double Simulation::GetReceptionAreaResultValueAtLatLon(double latitude_degrees, double longitude_degrees) const
{
float result;

	pRxAreaResults.GetInterplData(latitude_degrees, longitude_degrees, &result);
	return result;
}

double Simulation::GetReceptionAreaResultLatitude(int xIndex, int yIndex) const
{
	return pRxAreaResults.GetPos((unsigned int)xIndex, (unsigned int)yIndex).m_lat;
}

double Simulation::GetReceptionAreaResultLongitude(int xIndex, int yIndex) const
{
	return pRxAreaResults.GetPos((unsigned int)xIndex, (unsigned int)yIndex).m_lon;
}

bool Simulation::ExportReceptionAreaResultsToTextFile(const char* pathname, const char* resultsColumnName/*=nullptr*/) const
{
	if( resultsColumnName != nullptr )
		return pRxAreaResults.ExportToTextFile(pathname, resultsColumnName);
	else
	{
		std::string colNameStr  = pRxAreaResults.GetDataDescription();
		colNameStr.append(" (").append(pRxAreaResults.GetDataUnit()).append(")");
		return pRxAreaResults.ExportToTextFile(pathname, colNameStr.c_str());
	}
}

bool Simulation::ExportReceptionAreaResultsToMifFile(const char* pathname, const char* resultsUnits/*=nullptr*/) const
{
ContourFillsEngine contourEngine;
std::vector<ContourFillsEngine::PolyPolygon> polyPolygons;

	polyPolygons = contourEngine.GeneratePolyPolygons(pRxAreaResults, pCoverageFills);
	return contourEngine.ExportToMifFile(pathname, (resultsUnits != nullptr) ? resultsUnits : pRxAreaResults.GetDataUnit(), polyPolygons);
}

bool Simulation::ExportReceptionAreaResultsToKmlFile(const char* pathname, double fillOpacity_percent/*=50*/, double lineOpacity_percent/*=50*/, const char* resultsUnits/*=nullptr*/) const
{
ContourFillsEngine contourEngine;
std::vector<ContourFillsEngine::PolyPolygon> polyPolygons;

	polyPolygons = contourEngine.GeneratePolyPolygons(pRxAreaResults, pCoverageFills);
	return contourEngine.ExportToKmlFile(pathname, fillOpacity_percent, lineOpacity_percent, (resultsUnits != nullptr) ? resultsUnits : pRxAreaResults.GetDataUnit(), polyPolygons);
}

bool Simulation::ExportReceptionAreaResultsToBilFile(const char* pathname) const
{
	return pRxAreaResults.ExportToBILFile(pathname);
}

bool Simulation::ExportReceptionAreaTerrainElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero/*=false*/)
{
	if( numHorizontalPoints < 2 || numVerticalPoints < 2 )
		return false;

GeoDataGrid<float> grid((unsigned int)numHorizontalPoints, (unsigned int)numVerticalPoints);
float terrainElev;
Position pos;
double minLat, minLon, maxLat, maxLon;
float noDataValue;

	if( setNoDataToZero == true )
	{
		noDataValue = 0;
		grid.UndefineNoDataValue();
	}
	else
	{
		noDataValue = INT16_MIN;
		grid.DefineNoDataValue(noDataValue);
	}

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	grid.SetBordersCoordinates(minLat, minLon, maxLat, maxLon);
	for(unsigned int i=0 ; i<grid.SizeX() ; i++)
	{
		for(unsigned int j=0 ; j<grid.SizeY() ; j++)
		{
			pos = grid.GetPos(i, j);
			if( pTopoManager.GetTerrainElevation(pos.m_lat, pos.m_lon, &terrainElev) == true )
				grid.SetData(i, j, terrainElev);
			else
				grid.SetData(i, j, noDataValue);
		}
	}
	pTopoManager.ReleaseResources(true);

	return grid.ExportToBILFile(pathname);
}

bool Simulation::ExportReceptionAreaLandCoverClassesToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool mapValues)
{
	if( numHorizontalPoints < 2 || numVerticalPoints < 2 )
		return false;

GeoDataGrid<short> grid((unsigned int)numHorizontalPoints, (unsigned int)numVerticalPoints);
int landCover;
Position pos;
double minLat, minLon, maxLat, maxLon;
short noDataValue = -1;
bool success;

	grid.DefineNoDataValue(noDataValue);
	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	grid.SetBordersCoordinates(minLat, minLon, maxLat, maxLon);
	for(unsigned int i=0 ; i<grid.SizeX() ; i++)
	{
		for(unsigned int j=0 ; j<grid.SizeY() ; j++)
		{
			pos = grid.GetPos(i, j);
			if( mapValues == false )
				success = pTopoManager.GetLandCover(pos.m_lat, pos.m_lon, &landCover);
			else
				success = pTopoManager.GetLandCoverMappedValue(pos.m_lat, pos.m_lon, pPropagModelId, &landCover);
			if( success == true )
				grid.SetData(i, j, (short)landCover);
			else
				grid.SetData(i, j, noDataValue);
		}
	}
	pTopoManager.ReleaseResources(true);

	return grid.ExportToBILFile(pathname);
}

bool Simulation::ExportReceptionAreaSurfaceElevationToBilFile(const char* pathname, int numHorizontalPoints, int numVerticalPoints, bool setNoDataToZero)
{
	if( numHorizontalPoints < 2 || numVerticalPoints < 2 )
		return false;

GeoDataGrid<float> grid((unsigned int)numHorizontalPoints, (unsigned int)numVerticalPoints);
float surfaceElev;
Position pos;
double minLat, minLon, maxLat, maxLon;
float noDataValue;

	if( setNoDataToZero == true )
	{
		noDataValue = 0;
		grid.UndefineNoDataValue();
	}
	else
	{
		noDataValue = INT16_MIN;
		grid.DefineNoDataValue(noDataValue);
	}

	pRxAreaResults.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	grid.SetBordersCoordinates(minLat, minLon, maxLat, maxLon);
	for(unsigned int i=0 ; i<grid.SizeX() ; i++)
	{
		for(unsigned int j=0 ; j<grid.SizeY() ; j++)
		{
			pos = grid.GetPos(i, j);
			if( pTopoManager.GetSurfaceElevation(pos.m_lat, pos.m_lon, &surfaceElev) == true )
				grid.SetData(i, j, surfaceElev);
			else
				grid.SetData(i, j, noDataValue);
		}
	}
	pTopoManager.ReleaseResources(true);

	return grid.ExportToBILFile(pathname);
}

bool Simulation::ExportProfilesToCsvFile(const char* pathname, double latitude_degrees, double longitude_degrees)
{
	return pGenerator.ExportProfilesToCsvFile(*this, pathname, latitude_degrees, longitude_degrees);
}



// private methods

bool Simulation::pIsTerrainElevDataSourceValid(TerrainElevDataSource terrainElevSource)
{
	if( terrainElevSource >= TERR_ELEV_NONE && terrainElevSource <= TERR_ELEV_NRCAN_MRDEM_DTM )
		return true;
	return false;
}

bool Simulation::pIsLandCoverDataSourceValid(LandCoverDataSource landCoverSource)
{
	if( landCoverSource >= LAND_COVER_NONE && landCoverSource <= LAND_COVER_NRCAN )
		return true;
	return false;
}

bool Simulation::pIsSurfaceElevDataSourceValid(SurfaceElevDataSource surfaceElevSource)
{
	if( surfaceElevSource >= SURF_ELEV_NONE && surfaceElevSource <= SURF_ELEV_NRCAN_MRDEM_DSM )
		return true;
	return false;
}

void Simulation::pUpdateTerrainManagerSourcePtrs()
{
TerrainElevSource *terrainElevSource1, *terrainElevSource2, *terrainElevSource3;
SurfaceElevSource *surfaceElevSource1, *surfaceElevSource2, *surfaceElevSource3;
LandCoverSource *landCoverSource1, *landCoverSource2;

	pTopoManager.ClearTerrainElevSources();
	pTopoManager.ClearLandCoverSources();
	pTopoManager.ClearSurfaceElevSources();
	pTopoManager.ClearPairedTerrainAndSurfaceElevSources();

	terrainElevSource1 = pGetTerrainElevSourceObjPtr(pPrimaryTerrainElevSourceId);
	terrainElevSource2 = pGetTerrainElevSourceObjPtr(pSecondaryTerrainElevSourceId);
	terrainElevSource3 = pGetTerrainElevSourceObjPtr(pTertiaryTerrainElevSourceId);

	landCoverSource1 = pGetLandCoverSourceObjPtr(pPrimaryLandCoverSourceId);
	landCoverSource2 = pGetLandCoverSourceObjPtr(pSecondaryLandCoverSourceId);

	surfaceElevSource1 = pGetSurfaceElevSourceObjPtr(pPrimarySurfElevSourceId);
	surfaceElevSource2 = pGetSurfaceElevSourceObjPtr(pSecondarySurfElevSourceId);
	surfaceElevSource3 = pGetSurfaceElevSourceObjPtr(pTertiarySurfElevSourceId);

	if( terrainElevSource1 != nullptr ) pTopoManager.AddTerrainElevSource(terrainElevSource1);
	if( terrainElevSource2 != nullptr ) pTopoManager.AddTerrainElevSource(terrainElevSource2);
	if( terrainElevSource3 != nullptr ) pTopoManager.AddTerrainElevSource(terrainElevSource3);

	if( landCoverSource1 != nullptr ) pTopoManager.AddLandCoverSource(landCoverSource1);
	if( landCoverSource2 != nullptr ) pTopoManager.AddLandCoverSource(landCoverSource2);

	if( surfaceElevSource1 != nullptr ) pTopoManager.AddSurfaceElevSource(surfaceElevSource1);
	if( surfaceElevSource2 != nullptr ) pTopoManager.AddSurfaceElevSource(surfaceElevSource2);
	if( surfaceElevSource3 != nullptr ) pTopoManager.AddSurfaceElevSource(surfaceElevSource3);

	if( pTopoManager.UsePairedTerrainAndSurfaceElevSources() )
	{
		if( terrainElevSource1 != nullptr && surfaceElevSource1 != nullptr )
			pTopoManager.AddPairedTerrainAndSurfaceElevSources(terrainElevSource1, surfaceElevSource1);

		if( terrainElevSource2 != nullptr && surfaceElevSource2 != nullptr )
			pTopoManager.AddPairedTerrainAndSurfaceElevSources(terrainElevSource2, surfaceElevSource2);

		if( terrainElevSource3 != nullptr && surfaceElevSource3 != nullptr )
			pTopoManager.AddPairedTerrainAndSurfaceElevSources(terrainElevSource3, surfaceElevSource3);
	}
}

TerrainElevSource* Simulation::pGetTerrainElevSourceObjPtr(TerrainElevDataSource terrainElevSource)
{
	return const_cast<TerrainElevSource*>(pGetTerrainElevSourceConstObjPtr(terrainElevSource));
}

const TerrainElevSource* Simulation::pGetTerrainElevSourceConstObjPtr(TerrainElevDataSource terrainElevSource) const
{
	switch (terrainElevSource)
	{
	case TERR_ELEV_NONE:
		return nullptr;
	case TERR_ELEV_SRTM:
		return &pTerrainElevSrtm;
	case TERR_ELEV_CUSTOM:
		return &pTerrainElevCustom;
	case TERR_ELEV_NRCAN_CDEM:
		return &pTerrainElevCdem;
	case TERR_ELEV_NRCAN_HRDEM_DTM:
		return &pTerrainElevHrdemDtm;
	case TERR_ELEV_GEOTIFF:
		return &pTerrainElevGeotiff;
	case TERR_ELEV_NRCAN_MRDEM_DTM:
		return &pTerrainElevMrdemDtm;
	}
	return nullptr;
}

LandCoverSource* Simulation::pGetLandCoverSourceObjPtr(LandCoverDataSource landCoverSource)
{
	return const_cast<LandCoverSource*>(pGetLandCoverSourceConstObjPtr(landCoverSource));
}

const LandCoverSource* Simulation::pGetLandCoverSourceConstObjPtr(LandCoverDataSource landCoverSource) const
{
	switch (landCoverSource)
	{
	case LAND_COVER_NONE:
		return nullptr;
	case LAND_COVER_GEOTIFF:
		return &pLandCoverGeotiff;
	case LAND_COVER_ESA_WORLDCOVER:
		return &pLandCoverEsaWorldcover;
	case LAND_COVER_CUSTOM:
		return &pLandCoverCustom;
	case LAND_COVER_NRCAN:
		return &pLandCoverNrcan;
	}
	return nullptr;
}

SurfaceElevSource* Simulation::pGetSurfaceElevSourceObjPtr(SurfaceElevDataSource surfaceElevSource)
{
	return const_cast<SurfaceElevSource*>(pGetSurfaceElevSourceConstObjPtr(surfaceElevSource));
}
	
const SurfaceElevSource* Simulation::pGetSurfaceElevSourceConstObjPtr(SurfaceElevDataSource surfaceElevSource) const
{
	switch (surfaceElevSource)
	{
	case SURF_ELEV_NONE:
		return nullptr;
	case SURF_ELEV_SRTM:
		return &pSurfElevSrtm;
	case SURF_ELEV_CUSTOM:
		return &pSurfElevCustom;
	case SURF_ELEV_NRCAN_CDSM:
		return &pSurfElevCdsm;
	case SURF_ELEV_NRCAN_HRDEM_DSM:
		return &pSurfElevHrdemDsm;
	case SURF_ELEV_GEOTIFF:
		return &pSurfElevGeotiff;
	case SURF_ELEV_NRCAN_MRDEM_DSM:
		return &pSurfElevMrdemDsm;
	}
	return nullptr;
}

void Simulation::pSetDefaultEsaWorldcoverToP1812Mappings()
{
	pLandCoverEsaWorldcover.ClearMappings(ITU_R_P_1812);
	pLandCoverEsaWorldcover.SetDefaultMapping(ITU_R_P_1812, (int) P1812_OPEN_RURAL);
	pLandCoverEsaWorldcover.SetMapping(10 /*tree cover*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverEsaWorldcover.SetMapping(50 /*built-up*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverEsaWorldcover.SetMapping(80 /*permanent water bodies*/, ITU_R_P_1812, (int) P1812_WATER_SEA);
}
	
void Simulation::pSetDefaultEsaWorldcoverToP452v17Mappings()
{
	pLandCoverEsaWorldcover.ClearMappings(ITU_R_P_452_V17);
	pLandCoverEsaWorldcover.SetDefaultMapping(ITU_R_P_452_V17, (int) P452_HGM_OTHER);

	pLandCoverEsaWorldcover.SetMapping(10 /*tree cover*/, ITU_R_P_452_V17, (int) P452_HGM_MIXED_TREE_FOREST);
	pLandCoverEsaWorldcover.SetMapping(20 /*shrubland*/, ITU_R_P_452_V17, (int) P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES);
	pLandCoverEsaWorldcover.SetMapping(30 /*grassland*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(40 /*cropland*/, ITU_R_P_452_V17, (int) P452_HGM_HIGH_CROP_FIELDS);
	pLandCoverEsaWorldcover.SetMapping(50 /*built-up*/, ITU_R_P_452_V17, (int) P452_HGM_SUBURBAN);
	pLandCoverEsaWorldcover.SetMapping(60 /*bare / sparse vegetation*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(70 /*snow and ice*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(80 /*permanent water bodies*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(90 /*herbaceous wetland*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(95 /*mangroves*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverEsaWorldcover.SetMapping(100 /*moss and lichen*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
}

void Simulation::pSetDefaultEsaWorldcoverToP452v18Mappings()
{
	pLandCoverEsaWorldcover.ClearMappings(ITU_R_P_452_V18);
	pLandCoverEsaWorldcover.SetDefaultMapping(ITU_R_P_452_V18, (int) P452_OPEN_RURAL);
	pLandCoverEsaWorldcover.SetMapping(10 /*tree cover*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverEsaWorldcover.SetMapping(50 /*built-up*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverEsaWorldcover.SetMapping(80 /*permanent water bodies*/, ITU_R_P_452_V18, (int) P452_WATER_SEA);
}

void Simulation::pSetDefaultNrcanLandCoverToP1812Mappings()
{
	pLandCoverNrcan.ClearMappings(ITU_R_P_1812);
	pLandCoverNrcan.SetDefaultMapping(ITU_R_P_1812, (int) P1812_OPEN_RURAL);
	pLandCoverNrcan.SetMapping(1 /*Temperate or sub-polar needleleaf forest*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(2 /*Sub-polar taiga needleleaf forest*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(5 /*Temperate or sub-polar broadleaf deciduous forest*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(6 /*Forêt feuillue tempérée ou subpolaire*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(17 /*Urban*/, ITU_R_P_1812, (int) P1812_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(18 /*Water*/, ITU_R_P_1812, (int) P1812_WATER_SEA);
}

void Simulation::pSetDefaultNrcanLandCoverToP452v17Mappings()
{
	pLandCoverNrcan.ClearMappings(ITU_R_P_452_V17);
	pLandCoverNrcan.SetDefaultMapping(ITU_R_P_452_V17, (int) P452_HGM_OTHER);

	pLandCoverNrcan.SetMapping(1 /*Temperate or sub-polar needleleaf forest*/, ITU_R_P_452_V17, (int) P452_HGM_CONIFEROUS_TREES_IRREGULARLY_SPACED);
	pLandCoverNrcan.SetMapping(2 /*Sub-polar taiga needleleaf forest*/, ITU_R_P_452_V17, (int) P452_HGM_CONIFEROUS_TREES_IRREGULARLY_SPACED);
	pLandCoverNrcan.SetMapping(5 /*Temperate or sub-polar broadleaf deciduous forest*/, ITU_R_P_452_V17, (int) P452_HGM_DECIDUOUS_TREES_IRREGULARLY_SPACED);
	pLandCoverNrcan.SetMapping(6 /*Forêt feuillue tempérée ou subpolaire*/, ITU_R_P_452_V17, (int) P452_HGM_DECIDUOUS_TREES_IRREGULARLY_SPACED);
	pLandCoverNrcan.SetMapping(8 /*Temperate or sub-polar Shrubland*/, ITU_R_P_452_V17, (int) P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES);
	pLandCoverNrcan.SetMapping(10 /*Temperate or sub-polar grassland*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(11 /*Sub-polar or polar shrubland-lichen-moss*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(12 /*Sub-polar or polar grassland-lichen-moss*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(13 /*Sub-polar or polar barren-lichen-moss*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(14 /*Wetland*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(15 /*Cropland*/, ITU_R_P_452_V17, (int) P452_HGM_HIGH_CROP_FIELDS);
	pLandCoverNrcan.SetMapping(16 /*Barren lands*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(17 /*Urban*/, ITU_R_P_452_V17, (int) P452_HGM_URBAN);
	pLandCoverNrcan.SetMapping(18 /*Water*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
	pLandCoverNrcan.SetMapping(19 /*Snow and ice*/, ITU_R_P_452_V17, (int) P452_HGM_OTHER);
}
	
void Simulation::pSetDefaultNrcanLandCoverToP452v18Mappings()
{
	pLandCoverNrcan.ClearMappings(ITU_R_P_452_V18);
	pLandCoverNrcan.SetDefaultMapping(ITU_R_P_452_V18, (int) P452_OPEN_RURAL);
	pLandCoverNrcan.SetMapping(1 /*Temperate or sub-polar needleleaf forest*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(2 /*Sub-polar taiga needleleaf forest*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(5 /*Temperate or sub-polar broadleaf deciduous forest*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(6 /*Forêt feuillue tempérée ou subpolaire*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(17 /*Urban*/, ITU_R_P_452_V18, (int) P452_URBAN_TREES_FOREST);
	pLandCoverNrcan.SetMapping(18 /*Water*/, ITU_R_P_452_V18, (int) P452_WATER_SEA);
}

const CommTerminal* Simulation::pGetTerminalConstObjPtr(Terminal terminal) const
{
	switch (terminal)
	{
	case TRANSMITTER:
		return &pTx;
	case RECEIVER:
		return &pRx;
	}
	return nullptr;
}

CommTerminal* Simulation::pGetTerminalObjPtr(Crc::Covlib::Terminal terminal)
{
	return const_cast<CommTerminal*>(pGetTerminalConstObjPtr(terminal));
}

PropagModel* Simulation::pGetPropagModelPtr(PropagationModel propagModelId)
{
	switch (propagModelId)
	{
	case LONGLEY_RICE:
		return &pLongleyRiceModel;
	case ITU_R_P_1812:
		return &pIturp1812Model;
	case ITU_R_P_452_V17:
		return &pIturp452v17Model;
	case ITU_R_P_452_V18:
		return &pIturp452v18Model;
	case FREE_SPACE:
		return &pFreeSpaceModel;
	case EXTENDED_HATA:
		return &pEHataModel;
	case CRC_MLPL:
		return &pCrcMlplModel;
	case CRC_PATH_OBSCURA:
		return &pCrcPathObscuraModel;
	default:
		return nullptr;
	}
}
