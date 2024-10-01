#pragma once
#include "ITURP_452_1812_common.h"


class ITURP_452_common : public ITURP_452_1812_common
{
public:
	ITURP_452_common();
	~ITURP_452_common();

protected:
	double _AnnualEquivalentTimePercent(double pw, double lat, double omega);
	double _TotalGaseousAbsorption(double f, double hts, double hrs, double dn, double omega, double P, double TK);
	double _TroposcatterLoss(double f, double p, double Gt, double Gr, double dn, double theta, double N0, double P, double TK);    
	void _LoSPropagationLoss(double f, double p, double B0, double hts, double hrs, double dn, double dlt, double dlr, double omega, double P, double TK, double& Lbfsg, double& Lb0p, double& Lb0B);
	double _DuctingLayerReflectionLoss(double f, double p, double ae, double B0, double dlm, double dn, double hts, double hrs, double dlt, double dlr, double thetat, double thetar, double hte, double hre, double hm, double omega, double dct, double dcr, double P, double TK);
	double _OverallPrediction(double p, double B0, double dn, double omega, double htc, double hrc, double ae, unsigned int n, double* d, double* h, double Lb0p, double Lb0B, double Ldp, double Lbd50, double Lba, double Lbd, double Lbs);
};