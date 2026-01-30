/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

// This test compares path loss results from crc-covlib's ITURP_1812 class with results
// from the MATLAB/Octabe reference version approved by ITU-R Working Party 3K.
// The validation profiles' csv files were obtained from:
// https://github.com/eeveetza/p1812

#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../../src/ITURP_1812.h"


std::vector<std::string> tokenize(const char* text, const char* separators)
{
int i=0;
std::string str;
std::vector<std::string> strList;
bool separatorFound;

	while( text[i] != '\0' )
	{
		separatorFound = false;
		for(int j=0 ; separators[j] != '\0' ; j++)
		{
			if( text[i] == separators[j] )
			{
				separatorFound = true;
				break;
			}
		}

		if( separatorFound == false )
			str.push_back(text[i]);
		else
		{
			strList.push_back(str);
			str.clear();
		}
		i++;
	}
	strList.push_back(str);

	return strList;
}

struct P1812Input
{
	double f;
	double p;
	double pL;
	double latt;
	double lont;
	double latr;
	double lonr;
	double htg;
	double hrg;
	bool vpol;
	std::vector<double> d;
	std::vector<double> h;
	std::vector<ITURP_1812::ClutterCategory> cc;
	std::vector<ITURP_1812::RadioClimaticZone> rcz;
	std::vector<double> rch;
	double deltaN;
	double N0;
	double Lb;
};

double readDouble(std::fstream& f)
{
std::string line;
std::vector<std::string> tokens;

	getline(f, line);
	tokens = tokenize(line.c_str(), ",");
	return atof(tokens[1].c_str());
}

int readInt(std::fstream& f)
{
std::string line;
std::vector<std::string> tokens;

	getline(f, line);
	tokens = tokenize(line.c_str(), ",");
	return atoi(tokens[1].c_str());
}

void readProfilesLine(std::fstream& f, P1812Input& in)
{
std::string line;
std::vector<std::string> tokens;

	getline(f, line);
	tokens = tokenize(line.c_str(), ",");
	in.d.push_back( atof(tokens[0].c_str()) );
	in.h.push_back( atof(tokens[1].c_str()) );
	in.cc.push_back( (ITURP_1812::ClutterCategory) atoi(tokens[2].c_str()) );
	in.rcz.push_back( (ITURP_1812::RadioClimaticZone) atoi(tokens[4].c_str()) );
	in.rch.push_back( atof(tokens[3].c_str()) );
}

bool readMeasurementLine(std::fstream& f, P1812Input& in)
{
std::string line;
std::vector<std::string> tokens;
int pol;

	getline(f, line);
	if( line.rfind("{End of Measurements}", 0) != 0 )
	{
		tokens = tokenize(line.c_str(), ",");
		in.f = atof( tokens[0].c_str() ) / 1000.0;
		in.htg = atof( tokens[1].c_str() );
		in.hrg = atof( tokens[3].c_str() );
		pol = atoi( tokens[4].c_str() );
		if( pol == 1 )
			in.vpol = false;
		else if( pol == 2 )
			in.vpol = true;
		else
		{
			std::cout << "polarization=" << pol << std::endl;
			in.vpol = false;
		}
		in.p = atof( tokens[14].c_str() );
		in.pL = 50.0;
		in.Lb = atof( tokens[17].c_str() );
		return true;
	}
	else
		return false;
}

std::vector<struct P1812Input> readValidationProfileFile(const char* pathname)
{
std::fstream csvFile;
std::string line;
int numPoints;
P1812Input in;
std::vector<struct P1812Input> inputs;

	csvFile.open(pathname, std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		in.latt = readDouble(csvFile);
		in.lont = readDouble(csvFile);
		in.latr = readDouble(csvFile);
		in.lonr = readDouble(csvFile);
		for(int i=0 ; i<16 ; i++)
			getline(csvFile, line);
		in.deltaN = readDouble(csvFile);
		in.N0 = readDouble(csvFile);
		for(int i=0 ; i<14 ; i++)
			getline(csvFile, line);
		numPoints = readInt(csvFile);
		for(int i=0 ; i<numPoints ; i++)
			readProfilesLine(csvFile, in);
		for(int i=0 ; i<5 ; i++)
			getline(csvFile, line);
		while( readMeasurementLine(csvFile, in) )
			inputs.push_back(in);
	}
	csvFile.close();

	return inputs;
}

std::vector<std::string> getCsvPathnames(const char* directory)
{
std::vector<std::string> result;

	for (const auto& p : fs::recursive_directory_iterator(directory))
	{
		if (!fs::is_directory(p))
		{
			std::string ext(p.path().extension().string());
			if (ext == ".csv")
				result.push_back(p.path().string());
		}
	}
	return result;
}

int main(int argc, char* argv[])
{
ITURP_1812 p1812;
std::vector<struct P1812Input> inputs;
std::vector<std::string> csvPathnames;
double Lb, diff;
struct P1812Input* in;
std::string testResult;
int numPassed = 0;
int numFailed = 0;
double maxDiff = 0;

	csvPathnames = getCsvPathnames("./validation_profiles/");

	std::cout << std::endl;
	std::cout << std::setprecision(9);
	std::cout << std::setw(16) << std::left << "C++ (covlib)";
	std::cout << std::setw(16) << std::left <<"MATLAB/Octave";
	std::cout << std::setw(20) << std::left << "Diff";
	std::cout << std::setw(16) << std::left << "PASSED/FAILED";
	std::cout << std::left << "Filename" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;

	for(unsigned int i=0 ; i<csvPathnames.size() ; i++)
	{
		inputs = readValidationProfileFile(csvPathnames[i].c_str());
		for(unsigned int j=0 ; j<inputs.size() ; j++)
		{
			in = &(inputs[j]);

			Lb = p1812.BasicTransmissionloss(in->f, in->p, in->pL, in->latt, in->lont, in->latr, in->lonr, in->htg, in->hrg,
											in->vpol, in->d.size(), in->d.data(), in->h.data(), in->cc.data(), in->rch.data(),
											nullptr, in->rcz.data(), 500.0, 500.0, in->deltaN, in->N0);

			diff = fabs(Lb-in->Lb);
			maxDiff = std::max(maxDiff, diff);
			if( fabs(diff) < 0.001 )
			{
				testResult = "PASSED";
				numPassed++;
			}
			else
			{
				testResult = "FAILED";
				numFailed++;
			}
			std::cout << std::setw(16) << std::left << Lb;
			std::cout << std::setw(16) << in->Lb;
			std::cout << std::setw(20) << diff;
			std::cout << std::setw(16) << testResult;
			std::cout << fs::path(csvPathnames[i]).filename().string() << " [" << j+1 << "]" << std::endl;
		}
	}

	std::cout << std::endl;
	std::cout << "PASSED: " << numPassed << std:: endl;
	std::cout << "FAILED: " << numFailed << std:: endl;
	std::cout << "Max diff: " << maxDiff << std:: endl;

	return 0;
}