/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "ITURP_452_common.h"


class ITURP_452_v18 : public ITURP_452_common
{
public:
	ITURP_452_v18();
	~ITURP_452_v18();

	void SetDefaultRepresentativeHeight(ClutterCategory clutterCategory, double representativeHeight);
	double GetDefaultRepresentativeHeight(ClutterCategory clutterCategory) const;

	double ClearAirBasicTransmissionLoss(double f, double p, bool worstMonth, double latt, double lont, double latr, double lonr,
	                                     double htg, double hrg, double Gt, double Gr, bool vpol, unsigned int n, double* d, double* h,
	                                     ClutterCategory* cc=nullptr, double* rch=nullptr, RadioClimaticZone* rcz=nullptr,
	                                     double dct=AUTO, double dcr=AUTO, double P=AUTO, double TC=AUTO, double deltaN=AUTO, double N0=AUTO);

protected:
	void _gProfile(unsigned int n, double* d, double* h, ClutterCategory* cc, double* rch, std::vector<double>& gVector);
};