#pragma once
#include <vector>
#include <map>
#include <limits>


class ITURP_452_1812_common
{
public:
	ITURP_452_1812_common();
	~ITURP_452_1812_common();

	inline static constexpr double AUTO = std::numeric_limits<double>::quiet_NaN();

	enum RadioClimaticZone
	{
		COASTAL_LAND = 3,
		INLAND       = 4,
		SEA          = 1
	};

	enum ClutterCategory
	{
		WATER_SEA           = 1,
		OPEN_RURAL          = 2,
		SUBURBAN            = 3,
		URBAN_TREES_FOREST  = 4,
		DENSE_URBAN         = 5
	};

protected:
	void _LengthOfPathSectionsProfile(unsigned int n, double* d, std::vector<double>& lopsVector);
	void _EffectiveEarthRadius(double deltaN, double& ae, double& aB);
	double _OverWaterPercent(double dn, unsigned int n, RadioClimaticZone* rcz, double* lops);
	void _LongestContinuousLand(double dn, unsigned int n, RadioClimaticZone* rcz, double* lops, double &dtm, double& dlm);
	double _B0(double lat, double dtm, double dlm);
	void _LoSPropagationLoss(double f, double p, double B0, double hts, double hrs, double dn, double dlt, double dlr, double& Lbfs, double& Lb0p, double& Lb0B);
	void _DiffractionLoss(double f, double p, double B0, double htc, double hrc, bool vpol, double ae, double aB, unsigned int n, double* d, double* g, double omega, double hstd, double hsrd, double Lbfs, double Lb0p, double& Ldp, double& Lbd50, double& Lbd);
	double _DuctingLayerReflectionLoss(double f, double p, double ae, double B0, double dlm, double dn, double hts, double hrs, double dlt, double dlr, double thetat, double thetar, double hte, double hre, double hm, double omega, double dct, double dcr);
	double _50PLocBasicTransmissionLoss(double p, double B0, double dn, double Fj, double omega, double Lb0p, double Lb0B, double Ldp, double Lbd50, double Lba, double Lbd, double Lbs);
	void _PathProfileParameters(double f, double htg, double hrg, unsigned int n, double* d, double* h, double ae, double& dlt, double& dlr, double& thetat, double& thetar, double& theta, double& hstd, double& hsrd, double& hte, double& hre, double& hm);
	double _I(double x);
	double _Dct(unsigned int n, RadioClimaticZone* rcz, double* lops);
	double _Dcr(unsigned int n, RadioClimaticZone* rcz, double* lops);
	bool _IsAUTO(double param);
	void _SetDefaultRepresentativeHeight(ClutterCategory clutterCategory, double representativeHeight);
	double _GetDefaultRepresentativeHeight(ClutterCategory clutterCategory) const;

	std::map<ClutterCategory, double> _defaultRepClutterHeights;

private:
	double _BullingtonDiffractionLoss(double f, double htc, double hrc, double Ce, unsigned int n, double* d, double* g);
	double _SphericalEarthDiffractionLoss(double f, bool vpol, double dn, double ap, double omega, double htesph, double hresph);
	double _FirstTermSphericalEarthDiffractionLoss(double f, bool vpol, double dn, double adft, double omega, double htesph, double hresph);
	double _FirstTermSubCalculation(double f, bool vpol, double dn, double adft, double perm, double conduct, double htesph, double hresph);
	double _DeltaBullingtonDiffractionLoss(double f, double htc, double hrc, bool vpol, double ap, double Ce, unsigned int n, double* d, double* g, double omega, double hstd, double hsrd);

	double _ElevationAngle(double hamsl_from, double hamsl_to, double distKm, double ae);
	void _MaxTxToTerrainElevationAngle(double hts, unsigned int n, double* d, double* h, double ae, double& theta_max, unsigned int& dindex);
	unsigned int _LoSMaxDiffractionDist(double f, double hts, double hrs, unsigned int n, double* d, double* h, double Ce);
	void _MaxRxToTerrainElevationAngle(double hts, unsigned int n, double* d, double* h, double ae, double& theta_max, unsigned int& dindex);
	void _SmoothEarthHeights(unsigned int n, double* d, double* h, double& hst, double& hsr);
	void _SmoothEarthHeightsForDiffractionModel(double hts, double hrs, unsigned int n, double* d, double* h, double hst, double hsr, double& hstd, double& hsrd);
	void _EffectiveHeightsAndTerrainRoughness(double htg, double hrg, unsigned int n, double* d, double* h, double hst, double hsr, unsigned int ilt, unsigned int ilr, double& hte, double& hre, double& hm);
};