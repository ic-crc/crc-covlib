#include "ITURP1812PropagModel.h"
#include <cmath>

using namespace Crc::Covlib;


ITURP1812PropagModel::ITURP1812PropagModel()
{
	pTimePercent = 50;
	pLocationPercent = 50;
	pDeltaN = AUTOMATIC;
	pN0 = AUTOMATIC;
	pMappingType = P1812_MAP_TO_CLUTTER_CATEGORY;
	pSurfaceProfileMethod = P1812_ADD_REPR_CLUTTER_HEIGHT;
}
	
ITURP1812PropagModel::~ITURP1812PropagModel()
{

}

PropagationModel ITURP1812PropagModel::Id()
{
	return ITU_R_P_1812;
}

bool ITURP1812PropagModel::IsUsingTerrainElevData()
{
	return true;
}

bool ITURP1812PropagModel::IsUsingMappedLandCoverData()
{
	return (pSurfaceProfileMethod == P1812_ADD_REPR_CLUTTER_HEIGHT);
}

bool ITURP1812PropagModel::IsUsingItuRadioClimZoneData()
{
	return true;
}

bool ITURP1812PropagModel::IsUsingSurfaceElevData()
{
	return (pSurfaceProfileMethod == P1812_USE_SURFACE_ELEV_DATA);
}

int ITURP1812PropagModel::DefaultMappedLandCoverValue()
{
	if( pMappingType == P1812_MAP_TO_CLUTTER_CATEGORY )
		return P1812_OPEN_RURAL;
	else // P1812_MAP_TO_REPR_CLUTTER_HEIGHT
		return 0; // 0 meters
}

double ITURP1812PropagModel::CalcPathLoss(double freq_Ghz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
                                          Polarization pol, unsigned int sizeProfiles, double* distKmProfile, double* elevProfile, 
                                          int* mappedLandCoverProfile, double* surfaceHeightProfile, ITURadioClimaticZone* radioClimaticZoneProfile)
{
std::vector<double> reprClutterHeightVector;
double* reprClutterHeightProfile = nullptr;

	if( sizeProfiles < 3 )
		return 0;
	if( distKmProfile[sizeProfiles-1] < 1E-5 )
		return 0;

	if( IsUsingMappedLandCoverData() )
	{
		GetReprClutterHeightProfile(sizeProfiles, mappedLandCoverProfile, &reprClutterHeightVector);
		reprClutterHeightProfile = reprClutterHeightVector.data();
	}

	if( IsUsingSurfaceElevData() == false )
		surfaceHeightProfile = nullptr;

	static_assert(std::isnan(Crc::Covlib::AUTOMATIC) && std::isnan(ITURP_452_1812_common::AUTO), "");

	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_COASTAL_LAND == (int)ITURP_452_1812_common::RadioClimaticZone::COASTAL_LAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_INLAND == (int)ITURP_452_1812_common::RadioClimaticZone::INLAND, "");
	static_assert((int)Crc::Covlib::ITURadioClimaticZone::ITU_SEA == (int)ITURP_452_1812_common::RadioClimaticZone::SEA, "");

	return BasicTransmissionloss(freq_Ghz, pTimePercent, pLocationPercent, txLat, txLon, rxLat, rxLon, txRcagl_m, rxRcagl_m,
	                             pol==VERTICAL_POL, sizeProfiles, distKmProfile, elevProfile, nullptr, reprClutterHeightProfile,
	                             surfaceHeightProfile, (RadioClimaticZone*)radioClimaticZoneProfile, pDeltaN, pN0);
}

void ITURP1812PropagModel::GetReprClutterHeightProfile(unsigned int sizeProfile, int* mappedLandCoverProfile, std::vector<double>* reprClutterHeightProfile)
{
	reprClutterHeightProfile->resize(sizeProfile);
	if( pMappingType == P1812_MAP_TO_CLUTTER_CATEGORY )
	{
		for(unsigned int i=0 ; i<sizeProfile ; i++)
			(*reprClutterHeightProfile)[i] = GetClutterCategoryReprHeight((P1812ClutterCategory)mappedLandCoverProfile[i]);
	}
	else // P1812_MAP_TO_REPR_CLUTTER_HEIGHT
	{
		for(unsigned int i=0 ; i<sizeProfile ; i++)
			(*reprClutterHeightProfile)[i] = (double)mappedLandCoverProfile[i];
	}
}

void ITURP1812PropagModel::SetTimePercentage(double percent)
{
	if( percent < 1.0 || percent > 50.0 )
		return;
	pTimePercent = percent;
}
	
double ITURP1812PropagModel::GetTimePercentage() const
{
	return pTimePercent;
}
	
void ITURP1812PropagModel::SetLocationPercentage(double percent)
{
	if( percent < 1.0 || percent > 99.0 )
		return;
	pLocationPercent = percent;
}

void ITURP1812PropagModel::SetAverageRadioRefractivityLapseRate(double deltaN)
{
	if( pIsAutomatic(deltaN) || deltaN > 0 )
		pDeltaN = deltaN;
}
	
double ITURP1812PropagModel::GetAverageRadioRefractivityLapseRate() const
{
	return pDeltaN;
}

void ITURP1812PropagModel::SetSeaLevelSurfaceRefractivity(double N0)
{
	if( pIsAutomatic(N0) || N0 > 0 )
		pN0 = N0;
}
	
double ITURP1812PropagModel::GetSeaLevelSurfaceRefractivity() const
{
	return pN0;
}

double ITURP1812PropagModel::GetLocationPercentage() const
{
	return pLocationPercent;
}

void ITURP1812PropagModel::SetClutterCategoryReprHeight(P1812ClutterCategory clutterCategory, double representativeHeight_m)
{
static_assert((int)ITURP_452_1812_common::ClutterCategory::WATER_SEA == (int)Crc::Covlib::P1812ClutterCategory::P1812_WATER_SEA, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::OPEN_RURAL == (int)Crc::Covlib::P1812ClutterCategory::P1812_OPEN_RURAL, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::SUBURBAN == (int)Crc::Covlib::P1812ClutterCategory::P1812_SUBURBAN, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::URBAN_TREES_FOREST == (int)Crc::Covlib::P1812ClutterCategory::P1812_URBAN_TREES_FOREST, "");
static_assert((int)ITURP_452_1812_common::ClutterCategory::DENSE_URBAN == (int)Crc::Covlib::P1812ClutterCategory::P1812_DENSE_URBAN, "");

	ITURP_1812::SetDefaultRepresentativeHeight((ITURP_452_1812_common::ClutterCategory) clutterCategory, representativeHeight_m);
}

double ITURP1812PropagModel::GetClutterCategoryReprHeight(P1812ClutterCategory clutterCategory) const
{
	return ITURP_1812::GetDefaultRepresentativeHeight((ITURP_452_1812_common::ClutterCategory) clutterCategory);
}

void ITURP1812PropagModel::SetLandCoverMappingType(P1812LandCoverMappingType mappingType)
{
	if( mappingType == P1812_MAP_TO_CLUTTER_CATEGORY || mappingType == P1812_MAP_TO_REPR_CLUTTER_HEIGHT )
		pMappingType = mappingType;
}
	
P1812LandCoverMappingType ITURP1812PropagModel::GetLandCoverMappingType() const
{
	return pMappingType;
}

void ITURP1812PropagModel::SetSurfaceProfileMethod(P1812SurfaceProfileMethod method)
{
	if( method == P1812_ADD_REPR_CLUTTER_HEIGHT || method == P1812_USE_SURFACE_ELEV_DATA )
		pSurfaceProfileMethod = method;
}

P1812SurfaceProfileMethod ITURP1812PropagModel::GetSurfaceProfileMethod() const
{
	return pSurfaceProfileMethod;
}

bool ITURP1812PropagModel::pIsAutomatic(double param)
{
	return std::isnan(param);
}