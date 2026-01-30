/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

// This test compares path loss results from crc-covlib's ITURP_452v17 class with results
// from the MATLAB/Octabe reference version approved by ITU-R Working Party 3M.
// The validation profiles' csv files were obtained from:
// https://github.com/eeveetza/p452

#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../../src/ITURP_452_v17.h"


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

typedef ITURP_452_1812_common::RadioClimaticZone RadioClimaticZone;

struct Profile
{
	std::string filename;
	std::vector<double> d;
	std::vector<double> h;
	std::vector<RadioClimaticZone> rcz;
};

RadioClimaticZone toRCZ(std::string& s)
{
	if( s=="B" )
		return RadioClimaticZone::SEA;
	else if( s=="A1" )
		return RadioClimaticZone::COASTAL_LAND;
	return RadioClimaticZone::INLAND;
}

Profile readProfileFile(const char* pathname)
{
Profile profile;
std::fstream csvFile;
std::string line;
std::vector<std::string> tokens;

	profile.filename = fs::path(pathname).filename().string();
	csvFile.open(pathname, std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		getline(csvFile, line);
		while(!csvFile.fail())
		{
			tokens = tokenize(line.c_str(), ",");
			if( tokens.size() == 4 )
			{
				profile.d.push_back(atof(tokens[0].c_str()));
				profile.h.push_back(atof(tokens[1].c_str()));
				profile.rcz.push_back(toRCZ(tokens[2]));
			}
			getline(csvFile, line);
		}
	}
	csvFile.close();

	return profile;
}

std::vector<Profile> readProfileFiles(const char* directory)
{
std::vector<Profile> profiles;

	for (const auto& p : fs::recursive_directory_iterator(directory))
	{
		if (!fs::is_directory(p))
		{
			std::string ext(p.path().extension().string());
			if (ext == ".csv")
				profiles.push_back(readProfileFile(p.path().string().c_str()));
		}
	}

	return profiles;
}

struct ResultInput
{
	std::string filename;
	std::string profilesFilename;
	int lineNo;
	double f;
	double p;
	double htg;
	double hrg;
	double phi_path;
	double Gt;
	double Gr;
	int pol;
	double dct;
	double dcr;
	double DN;
	double N0;
	double press;
	double temp;
	double ha_t;
	double ha_r;
	double dk_t;
	double dk_r;
};

struct ResultOutput
{
	double Lb;
};

struct Result
{
	ResultInput in;
	ResultOutput out;
};

std::vector<Result> readResultFile(const char* pathname)
{
std::vector<Result> results;
std::fstream csvFile;
std::string line;
int lineNo = 0;
std::vector<std::string> tokens;
std::string filename = fs::path(pathname).filename().string();

	csvFile.open(pathname, std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		lineNo++;
		getline(csvFile, line);
		lineNo++;
		while(!csvFile.fail())
		{
			tokens = tokenize(line.c_str(), ",");
			if( tokens.size() >= 47 )
			{
			Result result;

				result.in.filename = filename;
				result.in.profilesFilename = tokens[0];
				result.in.lineNo = lineNo;
				result.in.f = atof(tokens[1].c_str());
				result.in.p = atof(tokens[2].c_str());
				result.in.htg = atof(tokens[3].c_str());
				result.in.hrg = atof(tokens[4].c_str());
				result.in.phi_path = atof(tokens[5].c_str());
				result.in.Gt = atof(tokens[6].c_str());
				result.in.Gr = atof(tokens[7].c_str());
				result.in.pol = atoi(tokens[8].c_str());
				result.in.dct = atof(tokens[9].c_str());
				result.in.dcr = atof(tokens[10].c_str());
				result.in.DN = atof(tokens[11].c_str());
				result.in.N0 = atof(tokens[12].c_str());
				result.in.press = atof(tokens[13].c_str());
				result.in.temp = atof(tokens[14].c_str());
				result.in.ha_t = atof(tokens[15].c_str());
				result.in.ha_r = atof(tokens[16].c_str());
				result.in.dk_t = atof(tokens[17].c_str());
				result.in.dk_r = atof(tokens[18].c_str());

				result.out.Lb = atof(tokens[38].c_str());

				results.push_back(result);
			}
			getline(csvFile, line);
			lineNo++;
		}
	}
	csvFile.close();

	return results;
}

std::vector<Result> readResultFiles(const char* directory)
{
std::vector<Result> results;

	for (const auto& p : fs::recursive_directory_iterator(directory))
	{
		if (!fs::is_directory(p))
		{
			std::string ext(p.path().extension().string());
			if (ext == ".csv")
			{
				std::vector<Result> fileResults = readResultFile(p.path().string().c_str());
				results.insert(results.end(), fileResults.begin(), fileResults.end());
			}
		}
	}

	return results;
}

int main()
{
ITURP_452_v17 p452v17;
std::vector<Profile> profiles;
std::vector<Result> results;
double diff;
std::string testResult;
int numPassed = 0;
int numFailed = 0;
double maxDiff = 0;

	profiles = readProfileFiles("./validation_examples/profiles/");
	results = readResultFiles("./validation_examples/results/");

	std::cout << std::endl;
	std::cout << std::setprecision(9);
	std::cout << std::setw(16) << std::left << "C++ (covlib)";
	std::cout << std::setw(16) << std::left <<"MATLAB/Octave";
	std::cout << std::setw(20) << std::left << "Diff";
	std::cout << std::setw(16) << std::left << "PASSED/FAILED";
	std::cout << std::left << "Filename [line no]" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;

	for(size_t i=0 ; i<results.size() ; i++)
	{
	Result* r = &(results[i]);
	Profile* p = nullptr;
	double Lb;

		for(size_t j=0 ; j<profiles.size() ; j++)
		{
			if( r->in.profilesFilename == profiles[j].filename)
			{
				p = &(profiles[j]);
				break;
			}
		}

		Lb = p452v17.ClearAirBasicTransmissionLoss(r->in.f, r->in.p, false, r->in.phi_path, 0,
		                                           r->in.htg, r->in.hrg, r->in.Gt, r->in.Gr, (r->in.pol==2), p->d.size(),
		                                           p->d.data(), p->h.data(), p->rcz.data(), r->in.dct, r->in.dcr, r->in.press,
		                                           r->in.temp, r->in.DN, r->in.N0, r->in.dk_t, r->in.dk_r, r->in.ha_t, r->in.ha_r);

		diff = fabs(Lb - r->out.Lb);
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
		std::cout << std::setw(16) << r->out.Lb;
		std::cout << std::setw(20) << diff;
		std::cout << std::setw(16) << testResult;
		std::cout << r->in.filename << "[" << r->in.lineNo << "]" << std::endl;
	}

	std::cout << std::endl;
	std::cout << "PASSED: " << numPassed /*<< "/" << numPassed+numFailed*/ << std:: endl;
	std::cout << "FAILED: " << numFailed /*<< "/" << numPassed+numFailed*/ << std:: endl;
	std::cout << "Max diff: " << maxDiff << std:: endl;

	return 0;
}