/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "ITURP_452_1812_common.h"


class ITURP_1812 : public ITURP_452_1812_common
{
public:
	ITURP_1812();
	~ITURP_1812();

	void SetDefaultRepresentativeHeight(ClutterCategory clutterCategory, double representativeHeight);
	double GetDefaultRepresentativeHeight(ClutterCategory clutterCategory) const;

	void SetPredictionResolution(double resolution_meters);
	double GetPredictionResolution() const;

	double BasicTransmissionloss(double f, double p, double pL, double latt, double lont, double latr, double lonr, 
	                             double htg, double hrg, bool vpol, unsigned int n, double* d, double* h,
	                             ClutterCategory* cc=nullptr, double* rch=nullptr, double* sh=nullptr, RadioClimaticZone* rcz=nullptr,
	                             double dct=AUTO, double dcr=AUTO, double deltaN=AUTO, double N0=AUTO);

	static double FieldStrength(double f, double Lb);

protected:
	double _TroposcatterLoss(double f, double p, double dn, double theta, double N0);
	double _TroposcatterLoss(double f, double p, double latt, double lont, double latr, double lonr, double hts, double hrs, double Gt, double Gr, unsigned int n, double* d, double* h, double ae, double thetat, double thetar);
	double _50PLocBasicTransmissionLoss(double p, double B0, double dn, double theta, double omega, double Lb0p, double Lb0B, double Ldp, double Lbd50, double Lba, double Lbd, double Lbs);
	double _BasicTransmissionLoss(double f, double pL, double hrg, double R, double Lb0p, double Lbc);
	void _gProfile(unsigned int n, double* h, ClutterCategory* cc, double* rch, double* sh, std::vector<double>& gVector, double& R);
	double _TerrainHeight(unsigned int n, double* d, double* h, double distKm);

	double _wa;
};