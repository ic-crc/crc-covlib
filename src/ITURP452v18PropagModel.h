#pragma once
#include "ITURP_452_v18.h"
#include "PropagModel.h"
#include "CRC-COVLIB.h"


class ITURP452v18PropagModel : public ITURP_452_v18, public PropagModel
{
public:
	ITURP452v18PropagModel();
	~ITURP452v18PropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m, 
	                    double txAntGain_dBi, double rxAntGain_dBi, Crc::Covlib::Polarization pol, unsigned int sizeProfiles,
	                    double* distKmProfile, double* elevProfile, int* mappedLandCoverProfile, double* surfaceHeightProfile,
	                    Crc::Covlib::ITURadioClimaticZone* radioClimaticZoneProfile);

	void GetReprClutterHeightProfile(unsigned int sizeProfile, int* mappedLandCoverProfile, std::vector<double>* reprClutterHeightProfile);

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

	void SetClutterCategoryReprHeight(Crc::Covlib::P452ClutterCategory clutterCategory, double representativeHeight_m);
	double GetClutterCategoryReprHeight(Crc::Covlib::P452ClutterCategory clutterCategory) const;

	virtual void SetLandCoverMappingType(Crc::Covlib::P452LandCoverMappingType mappingType);
	virtual Crc::Covlib::P452LandCoverMappingType GetLandCoverMappingType() const;

	virtual void SetSurfaceProfileMethod(Crc::Covlib::P452SurfaceProfileMethod method);
	virtual Crc::Covlib::P452SurfaceProfileMethod GetSurfaceProfileMethod() const;

private:
	bool pIsAutomatic(double param);

	double pTimePercent;
	Crc::Covlib::P452PredictionType pPredictionType;
	double pDeltaN;
	double pN0;
	double pTemperature_C;
	double pPressure_hPa;
	Crc::Covlib::P452LandCoverMappingType pMappingType;
	Crc::Covlib::P452SurfaceProfileMethod pSurfaceProfileMethod;
};
