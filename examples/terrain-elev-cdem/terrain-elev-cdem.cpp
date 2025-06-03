/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <stdio.h>
#include <iostream>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - NRCAN CDEM" << std::endl;

	sim = NewSimulation();

	// Set transmitter parameters
	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters
	sim->SetReceiverHeightAboveGround(1.0);

	// Set propagation model parameters
	sim->SetLongleyRiceTimePercentage(50);
	sim->SetLongleyRiceLocationPercentage(50);
	sim->SetLongleyRiceSituationPercentage(50);

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	// One terrain elevation value every 25m in the terrain profiles that will be provided to Longley-Rice
	sim->SetTerrainElevDataSamplingResolution(25);

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(200);
	sim->SetReceptionAreaNumVerticalPoints(200);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	// Set contour values and colors when exporting results to .mif or .kml files
	sim->ClearCoverageDisplayFills();
	sim->AddCoverageDisplayFill(45, 60, 0x5555FF);
	sim->AddCoverageDisplayFill(60, 75, 0x0000FF);
	sim->AddCoverageDisplayFill(75, 300, 0x000088);

	std::cout << "Generating and exporting coverage results (high resolution)..." << std::endl;

	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToTextFile("terrain-elev-cdem-hires.txt");
	sim->ExportReceptionAreaResultsToMifFile("terrain-elev-cdem-hires.mif");
	sim->ExportReceptionAreaResultsToKmlFile("terrain-elev-cdem-hires.kml");
	sim->ExportReceptionAreaResultsToBilFile("terrain-elev-cdem-hires.bil");

	// Run same simulation again but at a lower resolution (much faster)
	std::cout << "Generating and exporting coverage results (low resolution)..." << std::endl;
	sim->SetTerrainElevDataSamplingResolution(200); // one value every 200m for terrain elevation profiles
	sim->SetReceptionAreaNumHorizontalPoints(60); // 3600 reception points (60x60)
	sim->SetReceptionAreaNumVerticalPoints(60);
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToTextFile("terrain-elev-cdem-lowres.txt");
	sim->ExportReceptionAreaResultsToMifFile("terrain-elev-cdem-lowres.mif");
	sim->ExportReceptionAreaResultsToKmlFile("terrain-elev-cdem-lowres.kml");
	sim->ExportReceptionAreaResultsToBilFile("terrain-elev-cdem-lowres.bil");

	sim->Release();

	std::cout << "Simulations completed" << std::endl;

	return 0;
}