#pragma once


class ITURP676GaseousAttenuationModel
{
public:
	ITURP676GaseousAttenuationModel();
	~ITURP676GaseousAttenuationModel();

	void SetActiveState(bool active);
	bool IsActive() const;

	void SetAtmosphericPressure(double pressure_hPa);
	double GetAtmosphericPressure() const;

	void SetTemperature(double temperature_K);
	double GetTemperature() const;

	void SetWaterVapourDensity(double density_gm3);
	double GetWaterVapourDensity();

	static double GaseousAttenuationPerKm(double frequency_GHz, double atmPressure_hPa, double temperature_K, double waterVapourDensity_gm3);

	double CalcGaseousAttenuation(double frequency_GHz, double txLat, double txLon, double rxLat, double rxLon, double txRcagl_m, double rxRcagl_m,
	                              unsigned int sizeProfiles, double* distKmProfile, double* elevProfile) const;


private:
	static bool pIsAutomatic(double param);

	bool pIsActive;
	double pAtmPressure_hPa;
	double pTemperature_K;
	double pWaterVapourDensity_gm3;
};
