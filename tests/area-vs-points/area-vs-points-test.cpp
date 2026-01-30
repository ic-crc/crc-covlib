/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;
using namespace std::chrono;

void runComparision(ISimulation* sim)
{
	std::cout << std::endl << "  Generating rx area results...";
	steady_clock::time_point begin = steady_clock::now();
	sim->GenerateReceptionAreaResults();
	steady_clock::time_point end = steady_clock::now();
	std::cout << " completed in " << duration_cast<milliseconds> (end - begin).count() << " msecs" << std::endl;

	std::cout << "  Generating points  results...";
	begin = steady_clock::now();
	double lat, lon;
	double areaResult, pointResult, diff, maxDiff=0;
	int xMax = sim->GetReceptionAreaNumHorizontalPoints();
	int yMax = sim->GetReceptionAreaNumVerticalPoints();
	for(int x=0 ; x<xMax ; x++)
	{
		for(int y=0 ; y<yMax ; y++)
		{
			lat = sim->GetReceptionAreaResultLatitude(x, y);
			lon = sim->GetReceptionAreaResultLongitude(x, y);
			areaResult = sim->GetReceptionAreaResultValue(x, y);
			pointResult = sim->GenerateReceptionPointResult(lat, lon);
			diff = fabs(areaResult-pointResult);
			maxDiff = std::max(diff, maxDiff);
		}
	}
	end = steady_clock::now();
	std::cout << " completed in " << duration_cast<milliseconds> (end - begin).count() << " msecs" << std::endl;

	std::cout << "  Max result diff: " << maxDiff;
	if( maxDiff < 0.0001 )
		std::cout << " PASSED" << std::endl;
	else
		std::cout << " FAILED" << std::endl;
}


int main(int argc, char* argv[])
{
ISimulation* sim;

	SetITUProprietaryDataDirectory("../../data/itu-proprietary");

	sim = NewSimulation();

	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	sim->SetReceiverHeightAboveGround(1.5);

	sim->SetPropagationModel(ITU_R_P_1812);

	sim->SetITURP1812TimePercentage(50);
	sim->SetITURP1812LocationPercentage(50);
	sim->SetITURP1812LandCoverMappingType(P1812_MAP_TO_CLUTTER_CATEGORY);
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");

	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	sim->SetTerrainElevDataSamplingResolution(25);

	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER);
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");

	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(180);
	sim->SetReceptionAreaNumVerticalPoints(200);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	runComparision(sim);

	sim->SetITURP1812SurfaceProfileMethod(P1812_USE_SURFACE_ELEV_DATA);
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_CDSM);
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_CDSM, "../../data/surface-elev-samples/NRCAN_CDSM");
	//sim->SetSurfaceElevDataSourceSamplingMethod(SURF_ELEV_NRCAN_CDSM, NEAREST_NEIGHBOR);

	runComparision(sim);

	return 0;
}