/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "ITURP_452_v17.h"
#include "PropagModel.h"
#include "CRC-COVLIB.h"


class ITURP452v17PropagModel : public ITURP_452_v17, public PropagModel
{
public:
	ITURP452v17PropagModel();
	~ITURP452v17PropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m, 
	                    double txAntGain_dBi, double rxAntGain_dBi, Crc::Covlib::Polarization pol, unsigned int sizeProfiles,
	                    double* distKmProfile, double* elevProfile, Crc::Covlib::ITURadioClimaticZone* radioClimaticZoneProfile,
	                    Crc::Covlib::P452HeightGainModelClutterCategory* clutterCatProfile=nullptr);

	void SetTimePercentage(double percent);
	double GetTimePercentage() const;

	void SetPredictionType(Crc::Covlib::P452PredictionType predictionType);
	Crc::Covlib::P452PredictionType GetPredictionType() const;

	void SetAverageRadioRefractivityLapseRate(double deltaN);
	double GetAverageRadioRefractivityLapseRate() const;

	void SetSeaLevelSurfaceRefractivity(double N0);
	double GetSeaLevelSurfaceRefractivity() const;

	void SetAirTemperature(double temperature_C);
	double GetAirTemperature() const;

	void SetAirPressure(double pressure_hPa);
	double GetAirPressure() const;

	struct ClutterParams
	{
		double nominalHeight_m;
		double nominalDist_km;
	};
	void SetNominalClutterParams(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory, double nominalHeight_m, double nominalDist_km);
	void SetNominalClutterParams(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory, ClutterParams nominalClutterParams);
	ClutterParams GetNominalClutterParams(Crc::Covlib::P452HeightGainModelClutterCategory clutterCategory) const;

	void SetTransmitterHeightGainModelMode(Crc::Covlib::P452HeightGainModelMode mode);
	Crc::Covlib::P452HeightGainModelMode GetTransmitterHeightGainModelMode() const;

	void SetReceiverHeightGainModelMode(Crc::Covlib::P452HeightGainModelMode mode);
	Crc::Covlib::P452HeightGainModelMode GetReceiverHeightGainModelMode() const;

	double GetClutterProfileUsageLimitKm() const;

private:
	bool pIsAutomatic(double param);

	double pTimePercent;
	Crc::Covlib::P452PredictionType pPredictionType;
	double pDeltaN;
	double pN0;
	double pTemperature_C;
	double pPressure_hPa;
	std::map<Crc::Covlib::P452HeightGainModelClutterCategory, ClutterParams> pClutterParams;
	Crc::Covlib::P452HeightGainModelMode pHeightGainModelModeAtTransmitter;
	Crc::Covlib::P452HeightGainModelMode pHeightGainModelModeAtReceiver;
	double pClutterProfileUsageLimitKm;
};