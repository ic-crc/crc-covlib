#pragma once
#include "CRC-COVLIB.h"
#include "PropagModel.h"

class EHataPropagModel : PropagModel
{
public:
	EHataPropagModel();
	~EHataPropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_mHz, double txHeight_meters, double rxHeight_meters, unsigned int sizeProfile, double deltaDist_km, double* elevProfile);

	void SetClutterEnvironment(Crc::Covlib::EHataClutterEnvironment clutterEnvironment);
	Crc::Covlib::EHataClutterEnvironment GetClutterEnvironment() const;
	void SetReliabilityPercentage(double percent);
	double GetReliabilityPercentage() const;

private:
	Crc::Covlib::EHataClutterEnvironment pClutterEnvironment;
	double pReliabilityPercent;
};