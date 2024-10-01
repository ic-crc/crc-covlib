#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#else
	#define APIENTRY
#endif

#include "iturp528.h"
#include "./include/p528.h"

P528LIB_API int APIENTRY P528(double d__km, double h_1__meter, double h_2__meter, double f__mhz, int T_pol, double p,
    int* out_propagation_mode, int* out_warnings, double* out_d__km, double* out_A__db, double* out_A_fs__db,
    double* out_A_a__db, double* out_theta_h1__rad)
{
    Result result;
    int ret;
    ret = P528(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p, &result);
    *out_propagation_mode = result.propagation_mode;
    *out_warnings = result.warnings;
    *out_d__km = result.d__km;
    *out_A__db = result.A__db;
    *out_A_fs__db = result.A_fs__db;
    *out_A_a__db = result.A_a__db;
    *out_theta_h1__rad = result.theta_h1__rad;

    return ret;
}
