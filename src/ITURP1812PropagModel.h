#pragma once
#include "ITURP_1812.h"
#include "PropagModel.h"
#include "CRC-COVLIB.h"
#include <vector>


class ITURP1812PropagModel : public ITURP_1812, public PropagModel
{
public:
	ITURP1812PropagModel();
	~ITURP1812PropagModel();

	virtual Crc::Covlib::PropagationModel Id();
	virtual bool IsUsingTerrainElevData();
	virtual bool IsUsingMappedLandCoverData();
	virtual bool IsUsingItuRadioClimZoneData();
	virtual bool IsUsingSurfaceElevData();
	virtual int DefaultMappedLandCoverValue();

	double CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
	                    Crc::Covlib::Polarization pol, unsigned int sizeProfiles, double* distKmProfile, double* elevProfile,
	                    int* mappedLandCoverProfile, double* surfaceHeightProfile, Crc::Covlib::ITURadioClimaticZone* radioClimaticZoneProfile);

	void GetReprClutterHeightProfile(unsigned int sizeProfile, int* mappedLandCoverProfile, std::vector<double>* reprClutterHeightProfile);

	void SetTimePercentage(double percent);
	double GetTimePercentage() const;

	void SetLocationPercentage(double percent);
	double GetLocationPercentage() const;

	void SetAverageRadioRefractivityLapseRate(double deltaN);
	double GetAverageRadioRefractivityLapseRate() const;

	void SetSeaLevelSurfaceRefractivity(double N0);
	double GetSeaLevelSurfaceRefractivity() const;

	void SetClutterCategoryReprHeight(Crc::Covlib::P1812ClutterCategory clutterCategory, double representativeHeight_m);
	double GetClutterCategoryReprHeight(Crc::Covlib::P1812ClutterCategory clutterCategory) const;

	void SetLandCoverMappingType(Crc::Covlib::P1812LandCoverMappingType mappingType);
	Crc::Covlib::P1812LandCoverMappingType GetLandCoverMappingType() const;

	void SetSurfaceProfileMethod(Crc::Covlib::P1812SurfaceProfileMethod method);
	Crc::Covlib::P1812SurfaceProfileMethod GetSurfaceProfileMethod() const;

private:
	bool pIsAutomatic(double param);

	double pTimePercent;
	double pLocationPercent;
	double pDeltaN;
	double pN0;
	Crc::Covlib::P1812LandCoverMappingType pMappingType;
	Crc::Covlib::P1812SurfaceProfileMethod pSurfaceProfileMethod;
};