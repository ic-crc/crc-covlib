#pragma once


class ITURP_2109
{
public:
	ITURP_2109();
	~ITURP_2109();

    enum BuildingType
	{
		TRADITIONAL         = 1,
		THERMALLY_EFFICIENT = 2
	};

	static double BuildingEntryLoss(double f_GHz, double probability_percent, BuildingType bldgType, double elevAngle_degrees);

protected:
    static const double _BLDG_COEFFS[2][9];
};