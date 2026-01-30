/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

// Both ITM (i.e. Longley-Rice) source code and validation profiles from:
// https://github.com/NTIA/itm
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "../../src/ntia_itm/include/itm.h"


std::vector<double> toDblVect(const char* text, char separator)
{
int i=0;
std::string str;
std::vector<double> dblVect;

	while( text[i] != '\0' )
	{
		if( text[i] != separator )
			str.push_back(text[i]);
		else
		{
			dblVect.push_back(atof(str.c_str()));
			str.clear();
		}
		i++;
	}
	dblVect.push_back(atof(str.c_str()));

	return dblVect;
}

std::vector<std::vector<double>> readElevProfiles(const char* pathname)
{
std::vector<std::vector<double>> elevProfiles;
std::fstream csvFile;
std::string line;

	csvFile.open(pathname, std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		while(!csvFile.fail())
		{
			elevProfiles.push_back( toDblVect(line.c_str(), ',') );
			getline(csvFile, line);
		}
	}
	csvFile.close();

	return elevProfiles;
}

struct ValidationProfile
{
	double h_tx__meter;
	double h_rx__meter;
	double epsilon;
	double sigma;
	double N_0;
	double  f__mhz;
	int	pol;
	int climate;
	double time;
	double location;
	double situation;
	int mdvar;
	double A__db;
};

std::vector<std::string> toStrVect(const char* text, char separator)
{
int i=0;
std::string str;
std::vector<std::string> strVect;

	while( text[i] != '\0' )
	{
		if( text[i] != separator )
			str.push_back(text[i]);
		else
		{
			strVect.push_back(str);
			str.clear();
		}
		i++;
	}
	strVect.push_back(str);

	return strVect;
}

std::vector<ValidationProfile> readValidationProfiles(const char* pathname)
{
std::vector<ValidationProfile> profiles;
std::fstream csvFile;
std::string line;
std::vector<std::string> tokens;

	csvFile.open(pathname, std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		getline(csvFile, line);
		while(!csvFile.fail())
		{
			tokens = toStrVect(line.c_str(), ',');
			if( tokens.size() >= 13 )
			{
			ValidationProfile profile;

				profile.h_tx__meter = atof(tokens[0].c_str());
				profile.h_rx__meter = atof(tokens[1].c_str());
				profile.epsilon = atof(tokens[2].c_str());
				profile.sigma = atof(tokens[3].c_str());
				profile.N_0 = atof(tokens[4].c_str());
				profile.f__mhz = atof(tokens[5].c_str());
				profile.pol = atoi(tokens[6].c_str());
				profile.climate = atoi(tokens[7].c_str());
				profile.time = atof(tokens[8].c_str());
				profile.location = atof(tokens[9].c_str());
				profile.situation = atof(tokens[10].c_str());
				profile.mdvar = atoi(tokens[11].c_str());
				profile.A__db = atof(tokens[12].c_str());

				profiles.push_back(profile);
			}
			getline(csvFile, line);
		}
	}
	csvFile.close();

	return profiles;
}


int main()
{
std::vector<ValidationProfile> validationProfiles;
std::vector<std::vector<double>> elevProfiles;
long warnings;
double A__db;
double diff;

	validationProfiles = readValidationProfiles("./p2p.csv");
	elevProfiles = readElevProfiles("./pfls.csv");

	std::cout << std::endl;
	std::cout << std::setprecision(9);
	std::cout << std::setw(22) << std::left << "From validaton file";
	std::cout << std::setw(16) << std::left <<"Computed";
	std::cout << std::setw(16) << std::left << "PASSED/FAILED";
	std::cout << std::endl << "----------------------------------------------------" << std::endl;

	for(size_t i=0 ; i<validationProfiles.size() ; i++)
	{
	ValidationProfile* vp = &(validationProfiles[i]);

		ITM_P2P_TLS(vp->h_tx__meter, vp->h_rx__meter, elevProfiles[i].data(), vp->climate, vp->N_0, vp->f__mhz,
		            vp->pol, vp->epsilon, vp->sigma, vp->mdvar, vp->time, vp->location, vp->situation, &A__db, &warnings);
		diff = fabs(vp->A__db-A__db);


		std::cout << std::setw(22) << std::left << vp->A__db;
		std::cout << std::setw(16) << A__db;
		if( diff < 0.005 )
			std::cout << std::setw(16) << "PASSED" << std::endl;
		else
			std::cout << std::setw(16) << "FAILED" << std::endl;
	}

	return 0;
}
