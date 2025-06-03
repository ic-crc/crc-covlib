/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include <vector>


class ITURP_DigitalMaps
{
public:
	ITURP_DigitalMaps();
	~ITURP_DigitalMaps();

	static bool Init_DN50(const char* pathname);
	static bool Init_N050(const char* pathname);
	static bool Init_T_Annual(const char* pathname);
	static bool Init_Surfwv_50(const char* pathname);

	static double DN50(double lat, double lon);
	static double N050(double lat, double lon);
	static double T_Annual(double lat, double lon);
	static double Surfwv_50(double lat, double lon);

private:
	static bool _InitDigitalMap(std::vector<double>& digitalMapVect, int expectedSize, const char* pathname);
	static bool _ReadDigitalMapFile(double* mapArray, int mapArraySize, const char* pathname);
	static double _SquareGridBilinearInterpolation(const double* mapArray, int numRows, int rowSize, double r, double c);

	static const int _DN50_SIZE = 29161;
	static std::vector<double> _DN50_Vect;
	static const int _N050_SIZE = 29161;
	static std::vector<double> _N050_Vect;
	static const int _T_ANNUAL_SIZE = 115921;
	static std::vector<double> _T_ANNUAL_Vect;
	static const int _SURFWV_50_FIXED_SIZE = 29161;
	static std::vector<double> _SURFWV_50_FIXED_Vect;
};