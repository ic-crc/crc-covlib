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

	std::cout << std::endl << "crc-covlib - NRCAN HRDEM" << std::endl;

	sim = NewSimulation();

	// Set transmitter parameters
	sim->SetTransmitterLocation(45.533030, -75.496224);
	sim->SetTransmitterHeight(15);
	sim->SetTransmitterPower(0.2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters
	sim->SetReceiverHeightAboveGround(1.0);

	// Set propagation model parameters
	sim->SetLongleyRiceTimePercentage(50);
	sim->SetLongleyRiceLocationPercentage(50);
	sim->SetLongleyRiceSituationPercentage(50);

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_HRDEM_DTM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_HRDEM_DTM, "../../data/terrain-elev-samples/NRCAN_HRDEM_DTM");
	sim->SetTerrainElevDataSourceSamplingMethod(TERR_ELEV_NRCAN_HRDEM_DTM, BILINEAR_INTERPOLATION); // alternately, could use NEAREST_NEIGHBOR
	// One terrain elevation value every 10m in the terrain profiles that will be provided to Longley-Rice
	sim->SetTerrainElevDataSamplingResolution(10);

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.5140, -75.5110, 45.5477, -75.4619);
	sim->SetReceptionAreaNumHorizontalPoints(100);
	sim->SetReceptionAreaNumVerticalPoints(100);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	// Set contour values and colors when exporting results to .mif or .kml files
	sim->ClearCoverageDisplayFills();
	sim->AddCoverageDisplayFill(45, 60, 0x5555FF);
	sim->AddCoverageDisplayFill(60, 75, 0x0000FF);
	sim->AddCoverageDisplayFill(75, 300, 0x000088);

	std::cout << "Generating and exporting coverage results..." << std::endl;

	sim->GenerateReceptionAreaResults();

	// Export results in various formats
	sim->ExportReceptionAreaResultsToTextFile("terrain-elev-hrdem.txt");
	sim->ExportReceptionAreaResultsToMifFile("terrain-elev-hrdem.mif");
	sim->ExportReceptionAreaResultsToKmlFile("terrain-elev-hrdem.kml");
	sim->ExportReceptionAreaResultsToBilFile("terrain-elev-hrdem.bil");

	sim->Release();

	std::cout << "Simulation completed" << std::endl;

	return 0;
}