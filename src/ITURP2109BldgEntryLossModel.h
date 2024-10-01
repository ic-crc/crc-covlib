#pragma once
#include "CRC-COVLIB.h"


class ITURP2109BldgEntryLossModel
{
public:
	ITURP2109BldgEntryLossModel();
	~ITURP2109BldgEntryLossModel();

	void SetActiveState(bool active);
	bool IsActive() const;

	void SetProbability(double percent);
	double GetProbability() const;

	void SetDefaultBuildingType(Crc::Covlib::P2109BuildingType buildingType);
	Crc::Covlib::P2109BuildingType GetDefaultBuildingType() const;

	double CalcBuildingEntryLoss(double frequency_GHz, double elevAngle_degrees) const;

private:
	bool pIsActive;
	double pProbability;
	Crc::Covlib::P2109BuildingType pDefaultBuildingType;
};