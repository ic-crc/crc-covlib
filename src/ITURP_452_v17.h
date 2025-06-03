/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "ITURP_452_common.h"


class ITURP_452_v17 : public ITURP_452_common
{
public:
	ITURP_452_v17();
	~ITURP_452_v17();

	double ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double lat, double lon,
	                                     double htg, double hrg, double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h,
	                                     RadioClimaticZone* rcz=nullptr, double dct=AUTO, double dcr=AUTO, double P=AUTO, double TC=AUTO,
	                                     double deltaN=AUTO, double N0=AUTO, double dkt=0, double dkr=0, double hat=0, double har=0);

	double ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double latt, double lont, double latr, double lonr,
	                                     double htg, double hrg, double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h,
	                                     RadioClimaticZone* rcz=nullptr, double dct=AUTO, double dcr=AUTO, double P=AUTO, double TC=AUTO,
	                                     double deltaN=AUTO, double N0=AUTO, double dkt=0, double dkr=0, double hat=0, double har=0);

protected:
	double _AdditionalClutterLosses(double f, double h, double dk, double ha);
	void _AdditionalClutterLosses(double f, double htg, double hrg, unsigned int n, double* d, double* h, double dkt, double dkr, double hat, double har, double& Aht, double& Ahr, double& htgc, double& hrgc, std::vector<double>& dc, std::vector<double>& hc);
};