#pragma once

#ifdef _WIN32
	#ifdef P528LIB_EXPORTS
		#define P528LIB_API __declspec(dllexport)
	#else
		#define P528LIB_API __declspec(dllimport)
	#endif
#else
	#define P528LIB_API
	#define __stdcall
#endif

extern "C" P528LIB_API int __stdcall P528(double d__km, double h_1__meter, double h_2__meter, double f__mhz, int T_pol, double p,
    int* out_propagation_mode, int* out_warnings, double* out_d__km, double* out_A__db, double* out_A_fs__db, double* out_A_a__db,
    double* out_theta_h1__rad);
