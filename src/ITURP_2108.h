#pragma once


class ITURP_2108
{
public:
	ITURP_2108();
	~ITURP_2108();

	static double StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent);

private:
	static double _StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent);
};