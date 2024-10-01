#pragma once


class ITURP_835
{
public:
	ITURP_835();
	~ITURP_835();

    static double StandardPressure(double geometricHeight_km);

protected:
    static double _ToGeopotentialHeight(double geometricHeight_km);
};