/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "CRC-COVLIB.h"
#include "PropagModel.h"


class LongleyRicePropagModel : public PropagModel
{
public:
	LongleyRicePropagModel();
	~LongleyRicePropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_mHz, double txHeight_meters, double rxHeight_meters, Crc::Covlib::Polarization pol, unsigned int sizeProfile, double deltaDist_km, double* elevProfile);

	void SetSurfaceRefractivity(double refractivity_NUnits);
	double GetSurfaceRefractivity() const;
	void SetGroundDielectricConst(double dielectricConst);
	double GetGroundDielectricConst() const;
	void SetGroundConductivity(double groundConduct_Sm);
	double GetGroundConductivity() const;
	void SetClimaticZone(Crc::Covlib::LRClimaticZone climaticZone);
	Crc::Covlib::LRClimaticZone GetClimaticZone() const;
	void SetActivePercentageSet(Crc::Covlib::LRPercentageSet percentageSet);
	Crc::Covlib::LRPercentageSet GetActivePercentageSet() const;
	void SetTimePercentage(double percent);
	double GetTimePercentage() const;
	void SetLocationPercentage(double percent);
	double GetLocationPercentage() const;
	void SetSituationPercentage(double percent);
	double GetSituationPercentage() const;
	void SetConfidencePercentage(double percent);
	double GetConfidencePercentage() const;
	void SetReliabilityPercentage(double percent);
	double GetReliabilityPercentage() const;
	void SetModeOfVariability(int mode);
	int GetModeOfVariability() const;

private:
	double pSurfaceRefractivity;
	double pGroundDielectricConst;
	double pGroundConductivity;
	Crc::Covlib::LRClimaticZone pClimaticZone;
	Crc::Covlib::LRPercentageSet pActivePercentageSet;
	double pTimePercent;
	double pLocationPercent;
	double pSituationPercent;
	double pConfidencePercent;
	double pReliabilityPercent;
	int pModeOfVariability;

	static const double MIN_PERCENT;
	static const double MAX_PERCENT;
};
