/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;
using namespace std;


template<typename T>
void initProfile(vector<T>* profile)
{
	if(profile != nullptr)
	{
		fill(profile->begin(), profile->end(), 0);
		profile->clear();
	}
}

void readProfilesCsv(const char* pathname, vector<double>* resultProfile, vector<double>* elevProfile,
                     vector<int>* clutterCatProfile, vector<int>* reprClutterHeightProfile, vector<double>* surfHeightProfile)
{
ifstream csvFile;
string line, field;
int resultColIndex=-1, elevColIndex=-1, clutterCatColIndex=-1, reprClutterHeightColIndex=-1, surfHeightProfileColIndex=-1;

	initProfile(resultProfile);
	initProfile(elevProfile);
	initProfile(clutterCatProfile);
	initProfile(reprClutterHeightProfile);
	initProfile(surfHeightProfile);

	csvFile.open(pathname, ios::in);

	getline(csvFile, line, '\n'); // read header row
	stringstream ss(line);
	for(int i=0 ; getline(ss, field, ',') ; i++)
	{
		if( field == "path loss (dB)" || field == "field strength (dBuV/m)" ||
		    field =="transmission loss (dB)" || field == "received power (dBm)")
			resultColIndex = i;
		else if( field == "terrain elevation (m)")
			elevColIndex = i;
		else if( field.find("clutter category") != string::npos )
			clutterCatColIndex = i;
		else if( field.find("representative clutter height (m)") != string::npos )
			reprClutterHeightColIndex = i;
		else if( field == "surface elevation (m)")
			surfHeightProfileColIndex = i;
	}
	
	while( csvFile.good() )
	{
		getline(csvFile, line, '\n');
		stringstream ss(line);
		for(int i=0 ; getline(ss, field, ',') ; i++)
		{
			if( i==resultColIndex && resultProfile!=nullptr)
			{
				if( field != "" )
					resultProfile->push_back(stod(field));
				else
					resultProfile->push_back(-9999);
			}
			if( i==elevColIndex && elevProfile!=nullptr)
				elevProfile->push_back(stod(field));
			if( i==clutterCatColIndex && clutterCatProfile!=nullptr)
				clutterCatProfile->push_back(stoi(field));
			if( i==reprClutterHeightColIndex && reprClutterHeightProfile!=nullptr)
				reprClutterHeightProfile->push_back((int)stof(field));
			if( i==surfHeightProfileColIndex && surfHeightProfile!=nullptr)
				surfHeightProfile->push_back(stod(field));
		}
	}
	csvFile.close();
}

bool isEqual(double a, double b)
{
	return fabs(a-b) < 0.001;
}

int gNumTests = 0;
int gNumPasses = 0;
void printResult(ISimulation* sim, bool usedSameProfiles, double result0, double result1)
{
PropagationModel propagModel = sim->GetPropagationModel();
bool passed = false;

	cout << setw(15) << left;
	if( propagModel == LONGLEY_RICE )
		 cout << "Longley Rice";
	else if( propagModel == ITU_R_P_1812 )
		cout << "ITU-R P.1812";
	else if( propagModel == ITU_R_P_452_V17 )
		cout << "ITU-R P.452-17";
	else if( propagModel == ITU_R_P_452_V18 )
		cout << "ITU-R P.452-18";

	if( usedSameProfiles == true )
	{
		if( isEqual(result0, result1) == true )
			passed = true;
	}
	else
	{
		if( isEqual(result0, result1) == false )
			passed = true;
	}

	cout << setw(19) << left;
	cout << (usedSameProfiles == true ? "same profiles":"different profiles") << result0 << " " << result1 << " "
	     << (passed ? "PASSED":"FAILED") << endl;

	gNumTests++;
	if(passed)
		gNumPasses++;
}

int main(int argc, char* argv[])
{
ISimulation* sim = NewSimulation();
double rxLat = 45.51353;
double rxLon = -75.45785;
vector<double> csvResultProfile;
vector<double> csvElevProfile;
vector<int> csvClutterCatProfile;
vector<int> csvReprClutterHeightProfile;
vector<double> csvSurfElevProfile;
double customProfilesResult;

	SetITUProprietaryDataDirectory("../../data/itu-proprietary");

	sim->SetTransmitterLocation(45.557573,-75.506452);
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_HRDEM_DTM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_HRDEM_DTM, "../../data/terrain-elev-samples/NRCAN_HRDEM_DTM");
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER);
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_HRDEM_DSM);
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_HRDEM_DSM, "../../data/surface-elev-samples/NRCAN_HRDEM_DSM");
	sim->SetTerrainElevDataSamplingResolution(25);


	// LONGLEY-RICE
	
	sim->SetPropagationModel(LONGLEY_RICE);
	sim->ExportProfilesToCsvFile("lr_profiles.csv", rxLat, rxLon);
	readProfilesCsv("lr_profiles.csv", &csvResultProfile, &csvElevProfile, nullptr, nullptr, nullptr);

	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data());
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	vector<double> zeroElevProfile(csvElevProfile.size(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, zeroElevProfile.size(), zeroElevProfile.data());
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);


	// ITU-R P.1812

	sim->SetPropagationModel(ITU_R_P_1812);
	sim->SetITURP1812SurfaceProfileMethod(P1812_ADD_REPR_CLUTTER_HEIGHT);
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");
	sim->ExportProfilesToCsvFile("P1812_profiles.csv", rxLat, rxLon);
	readProfilesCsv("P1812_profiles.csv", &csvResultProfile, &csvElevProfile, &csvClutterCatProfile, &csvReprClutterHeightProfile, nullptr);
	
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	zeroElevProfile.resize(csvElevProfile.size());
	fill(zeroElevProfile.begin(), zeroElevProfile.end(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, zeroElevProfile.size(), zeroElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	vector<int> openRuralClutterProfile(csvClutterCatProfile.size(), P1812_OPEN_RURAL);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), openRuralClutterProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	sim->SetITURP1812LandCoverMappingType(P1812_MAP_TO_REPR_CLUTTER_HEIGHT);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), csvReprClutterHeightProfile.data(), nullptr, nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	sim->SetITURP1812SurfaceProfileMethod(P1812_USE_SURFACE_ELEV_DATA);
	sim->ExportProfilesToCsvFile("P1812_profiles_2.csv", rxLat, rxLon);
	readProfilesCsv("P1812_profiles_2.csv", &csvResultProfile, &csvElevProfile, nullptr, nullptr, &csvSurfElevProfile);

	vector<double> zeroSurfHeightProfile(csvSurfElevProfile.size(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), nullptr, zeroSurfHeightProfile.data(), nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), nullptr, csvSurfElevProfile.data(), nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);


	// ITU-R P.452-17

	sim->SetPropagationModel(ITU_R_P_452_V17);
	sim->SetITURP452RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");
	sim->ExportProfilesToCsvFile("P452v17_profiles.csv", rxLat, rxLon);
	readProfilesCsv("P452v17_profiles.csv", &csvResultProfile, &csvElevProfile, &csvClutterCatProfile, nullptr, nullptr);

	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	zeroElevProfile.resize(csvElevProfile.size());
	fill(zeroElevProfile.begin(), zeroElevProfile.end(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, zeroElevProfile.size(), zeroElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	vector<int> otherClutterProfile(csvClutterCatProfile.size(), P452_HGM_OTHER);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), otherClutterProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);


	// ITU-R P.452-18

	sim->SetPropagationModel(ITU_R_P_452_V18);
	sim->SetITURP452SurfaceProfileMethod(P452_ADD_REPR_CLUTTER_HEIGHT);
	sim->ExportProfilesToCsvFile("P452v18_profiles.csv", rxLat, rxLon);
	readProfilesCsv("P452v18_profiles.csv", &csvResultProfile, &csvElevProfile, &csvClutterCatProfile, &csvReprClutterHeightProfile, nullptr);
	
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	zeroElevProfile.resize(csvElevProfile.size());
	fill(zeroElevProfile.begin(), zeroElevProfile.end(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, zeroElevProfile.size(), zeroElevProfile.data(), csvClutterCatProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	openRuralClutterProfile.resize(csvClutterCatProfile.size());
	fill(openRuralClutterProfile.begin(), openRuralClutterProfile.end(), P452_OPEN_RURAL);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), openRuralClutterProfile.data(), nullptr, nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	sim->SetITURP452LandCoverMappingType(P452_MAP_TO_REPR_CLUTTER_HEIGHT);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), csvReprClutterHeightProfile.data(), nullptr, nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);

	sim->SetITURP452SurfaceProfileMethod(P452_EXPERIMENTAL_USE_OF_SURFACE_ELEV_DATA);
	sim->ExportProfilesToCsvFile("P452v18_profiles_2.csv", rxLat, rxLon);
	readProfilesCsv("P452v18_profiles_2.csv", &csvResultProfile, &csvElevProfile, nullptr, nullptr, &csvSurfElevProfile);

	zeroSurfHeightProfile.resize(csvSurfElevProfile.size());
	fill(zeroSurfHeightProfile.begin(), zeroSurfHeightProfile.end(), 0);
	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), nullptr, zeroSurfHeightProfile.data(), nullptr);
	printResult(sim, false, csvResultProfile.back(), customProfilesResult);

	customProfilesResult = sim->GenerateProfileReceptionPointResult(rxLat, rxLon, csvElevProfile.size(), csvElevProfile.data(), nullptr, csvSurfElevProfile.data(), nullptr);
	printResult(sim, true, csvResultProfile.back(), customProfilesResult);


	std::cout << std::endl;
	std::cout << "PASSED: " << gNumPasses << std:: endl;
	std::cout << "FAILED: " << gNumTests-gNumPasses << std:: endl;

	sim->Release();

	return 0;
}