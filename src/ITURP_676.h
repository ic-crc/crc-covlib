#pragma once


class ITURP_676
{
public:
	ITURP_676();
	~ITURP_676();

	static double AttenuationDueToDryAir(double f, double P=1013.25, double TK=288.15, double rho=7.5);
	static double AttenuationDueToWaterVapour(double f, double P=1013.25, double TK=288.15, double rho=7.5);

protected:
	static double _NppOxygen(double f, double P, double TK, double rho);
	static double _NppWaterVapour(double f, double P, double TK, double rho);
	static double _NppD(double f, double P, double TK, double rho);

	static const double _TABLE1[44][7];
	static const double _TABLE2[35][7];
};