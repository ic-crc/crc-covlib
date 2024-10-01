#pragma once

// 2024-05-06 Modifications to original NTIA version:
//   - Changed definition of DLLEXPORT to avoid exporting functions.

#ifdef _WIN32
// Export the DLL functions as "C" and not C++
//#define DLLEXPORT extern "C" __declspec(dllexport)
#define DLLEXPORT
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#ifdef __linux__
//#define DLLEXPORT extern "C"
#define DLLEXPORT
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

struct InterValues
{
    double d_bp__km;
    double att_1km;
    double att_100km;

    double h_b_eff__meter;
    double h_m_eff__meter;

    // Terrain Stats
    double pfl10__meter;
    double pfl50__meter;
    double pfl90__meter;
    double deltah__meter;

    // Path Geometry
    double d__km;
    double d_hzn__meter[2];
    double h_avg__meter[2];
    double theta_m__mrad;
    double beta;
    int iend_ov_sea;
    double hedge_tilda;
    bool single_horizon;

    // Misc
    double slope_max;
    double slope_min;
};

#define PI 3.14159265358979323846

// public
DLLEXPORT void ExtendedHata(double pfl[], double f__mhz, double h_b__meter, double h_m__meter, int environment, double reliability, double *plb);
DLLEXPORT void ExtendedHata_DBG(double pfl[], double f__mhz, double h_b__meter, double h_m__meter, int environment, double reliability, double *plb, InterValues *interValues);

// private
void FindAverageGroundHeight(double *pfl, double h_m__meter, double h_b__meter, InterValues *interValues);
void MobileTerrainSlope(double *pfl, InterValues *interValues);
void LeastSquares(double *pfl_segment, double x1, double x2, double *z0, double *zn);
void AnalyzeSeaPath(double* pfl, InterValues *interValues);
void FindHorizons(double *pfl, double gme, double d__meter, double h_1__meter, double h_2__meter, double *d_hzn__meter);
void SingleHorizonTest(double *pfl, double h_m__meter, double h_b__meter, InterValues *interValues);
void ComputeTerrainStatistics(double *pfl, InterValues *interValues);
double FindQuantile(const int &nn, double *apfl, const int &ir);
void PreprocessTerrainPath(double *pfl, double h_b__meter, double h_m__meter, InterValues *interValues);
double AverageTerrainHeight(double *pfl);
double GeneralSlopeCorrectionFactor(double theta_m__mrad, double d__km);
double FineRollingHillyTerrainCorrectionFactor(InterValues *interValues, double h_m_gnd__meter);
double MixedPathCorrectionFactor(double d__km, InterValues *interValues);
double MedianRollingHillyTerrainCorrectionFactor(double deltah);
void MedianBasicPropLoss(double f__mhz, double h_b__meter, double h_m__meter, double d__km, int environment, double* plb_med__db, InterValues *interValues);
double IsolatedRidgeCorrectionFactor(double d1_hzn__km, double d2_hzn__km, double h_edge__meter);
double Variability(double plb_med__db, double f__mhz, int enviro_code, double reliability);
double Sigma_u(double f__mhz);
double Sigma_r(double f__mhz);
double ierf(double q);

