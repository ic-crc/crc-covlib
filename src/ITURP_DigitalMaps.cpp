#include "ITURP_DigitalMaps.h"
#include <algorithm>
#include <fstream>
#include <cstring>


std::vector<double> ITURP_DigitalMaps::_DN50_Vect;
std::vector<double> ITURP_DigitalMaps::_N050_Vect;
std::vector<double> ITURP_DigitalMaps::_T_ANNUAL_Vect;
std::vector<double> ITURP_DigitalMaps::_SURFWV_50_FIXED_Vect;


ITURP_DigitalMaps::ITURP_DigitalMaps()
{

}

ITURP_DigitalMaps::~ITURP_DigitalMaps()
{

}

// Initialize from DN50.TXT ITU file
bool ITURP_DigitalMaps::Init_DN50(const char* pathname)
{
	return _InitDigitalMap(_DN50_Vect, _DN50_SIZE, pathname);
}

// Initialize from N050.TXT ITU file
bool ITURP_DigitalMaps::Init_N050(const char* pathname)
{
	return _InitDigitalMap(_N050_Vect, _N050_SIZE, pathname);
}

// Initialize from T_Annual.TXT ITU file
bool ITURP_DigitalMaps::Init_T_Annual(const char* pathname)
{
	return _InitDigitalMap(_T_ANNUAL_Vect, _T_ANNUAL_SIZE, pathname);
}

// Initialize from surfwv_50_fixed.txt ITU file
bool ITURP_DigitalMaps::Init_Surfwv_50(const char* pathname)
{
	return _InitDigitalMap(_SURFWV_50_FIXED_Vect, _SURFWV_50_FIXED_SIZE, pathname);
}

// Get value from the DN50 digital map.
// ITU-R P.1812-6, Annex 1, Section 3.5
// ITU-R P.452-17/18, Attachment 1 to Annex 1, Section 2
// lat:      latitude (degrees) [-90.0, 90.0] 
// lon:      longitude (degrees) [-180.0, 180.0]
// [return]: median annual radio-refractivity lapse-rate through the lowest 1 km of the atmosphere (N-units/km)
double ITURP_DigitalMaps::DN50(double lat, double lon)
{
// lat from +90 to -90, lon from 0 to 360 in _DN50 array
const double latInterval = 1.5;
const double lonInterval = 1.5;
const int numRows = 121; // number of points from [-90, 90] latitude deg range, at 1.5 deg intervals: (180/1.5)+1
const int rowSize = 241; // number of points from [0, 360] longitude deg range, at 1.5 deg intervals: (360/1.5)+1
double r,c;

	if( _DN50_Vect.size() != _DN50_SIZE )
		return 45.0; // use reasonable default value

	if( lon < 0 )
		lon += 360;
	r = (90.0 - lat) / latInterval;
	c = lon / lonInterval;
	return _SquareGridBilinearInterpolation(_DN50_Vect.data(), numRows, rowSize, r, c);
}

// Get value from the N050 digital map.
// ITU-R P.1812-6, Annex 1, Section 3.5
// ITU-R P.452-17/18, Attachment 1 to Annex 1, Section 2
// lat:      latitude (degrees) [-90.0, 90.0] 
// lon:      longitude (degrees) [-180.0, 180.0]
// [return]: median annual sea-level surface refractivity (N-units)
double ITURP_DigitalMaps::N050(double lat, double lon)
{
// lat from +90 to -90, lon from 0 to 360 in _N050 array
const double latInterval = 1.5;
const double lonInterval = 1.5;
const int numRows = 121; // number of points from [-90, 90] latitude deg range, at 1.5 deg intervals: (180/1.5)+1
const int rowSize = 241; // number of points from [0, 360] longitude deg range, at 1.5 deg intervals: (360/1.5)+1
double r,c;

	if( _N050_Vect.size() != _N050_SIZE )
		return 325.0; // use reasonable default value

	if( lon < 0 )
		lon += 360;
	r = (90.0 - lat) / latInterval;
	c = lon / lonInterval;
	return _SquareGridBilinearInterpolation(_N050_Vect.data(), numRows, rowSize, r, c);
}

// Get value from the T_Annual digital map.
// ITU-R P.1510
// lat:      latitude (degrees) [-90.0, 90.0] 
// lon:      longitude (degrees) [-180.0, 180.0]
// [return]: the annual mean surface temperature at 2 meters above the surface of the Earth (Kelvins)
double ITURP_DigitalMaps::T_Annual(double lat, double lon)
{
// lat from -90 to +90, lon from -180 to +180 in _T_ANNUAL array
const double latInterval = 0.75;
const double lonInterval = 0.75;
const int numRows = 241; // number of points from [-90, 90] latitude deg range, at 0.75 deg intervals: (180/0.75)+1
const int rowSize = 481; // number of points from [-180, 180] longitude deg range, at 0.75 deg intervals: (360/0.75)+1
double r,c;

	if( _T_ANNUAL_Vect.size() != _T_ANNUAL_SIZE )
		return 288.15; // use reasonable default value

	r = (90.0 + lat) / latInterval;
	c = (180.0 + lon) / lonInterval;
	return _SquareGridBilinearInterpolation(_T_ANNUAL_Vect.data(), numRows, rowSize, r, c);
}

// Get value from the survwv_50_fixed digital map.
// ITU-R P.2001
// lat:      latitude (degrees) [-90.0, 90.0] 
// lon:      longitude (degrees) [-180.0, 180.0]
// [return]: the surface water-vapour density under non-rain conditions,
//           exceeded for 50% of an average year (g/m^3)
double ITURP_DigitalMaps::Surfwv_50(double lat, double lon)
{
// lat from +90 to -90, lon from 0 to 360 in _SURFWV_50 array
const double latInterval = 1.5;
const double lonInterval = 1.5;
const int numRows = 121; // number of points from [-90, 90] latitude deg range, at 1.5 deg intervals: (180/1.5)+1
const int rowSize = 241; // number of points from [0, 360] longitude deg range, at 1.5 deg intervals: (360/1.5)+1
double r,c;

	if( _SURFWV_50_FIXED_Vect.size() != _SURFWV_50_FIXED_SIZE )
		return 7.5; // use reasonable default value

	if( lon < 0 )
		lon += 360;
	r = (90.0 - lat) / latInterval;
	c = lon / lonInterval;
	return _SquareGridBilinearInterpolation(_SURFWV_50_FIXED_Vect.data(), numRows, rowSize, r, c);
}

// Bi-linear interpolation as documented in ITU-R P.1144-11, Annex 1, section 1b
double ITURP_DigitalMaps::_SquareGridBilinearInterpolation(const double* mapArray, int numRows, int rowSize, double r, double c)
{
int R, C;
double irc, I00, I01, I10, I11;

	R = (int)r;
	R = std::max(R, 0);
	R = std::min(R, numRows-2);
	C = (int)c;
	C = std::max(C, 0);
	C = std::min(C, rowSize-2);
	I00 = mapArray[ R    * rowSize +  C   ];
	I10 = mapArray[(R+1) * rowSize +  C   ];
	I01 = mapArray[ R    * rowSize + (C+1)];
	I11 = mapArray[(R+1) * rowSize + (C+1)];
	irc = I00*((R+1-r)*(C+1-c)) + I10*((r-R)*(C+1-c)) + I01*((R+1-r)*(c-C)) + I11*((r-R)*(c-C));

	return irc;
}

bool ITURP_DigitalMaps::_InitDigitalMap(std::vector<double>& digitalMapVect, int expectedSize, const char* pathname)
{
	digitalMapVect.resize((size_t)expectedSize);
	if( _ReadDigitalMapFile(digitalMapVect.data(), expectedSize, pathname) == false )
	{
		digitalMapVect.resize(0);
		return false;
	}
	return true;
}

bool ITURP_DigitalMaps::_ReadDigitalMapFile(double* mapArray, int mapArraySize, const char* pathname)
{
std::fstream txtFile;
std::string line;
char* token;
int arrayIndex = 0;
const char* delims = " \t";

	txtFile.open(pathname, std::ios::in);
	if(txtFile)
	{
		while( std::getline(txtFile, line) )
		{
			token = std::strtok((char*)line.c_str(), delims);
			while( token != NULL )
			{
				if( arrayIndex < mapArraySize )
					mapArray[arrayIndex] = atof(token);
				arrayIndex++;
				token = strtok(NULL, delims);
			}
		}
	}
	txtFile.close();

	return (arrayIndex == mapArraySize);
}