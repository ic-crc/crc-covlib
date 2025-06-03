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

void outputStatusMsg(int status)
{
	if( status == STATUS_OK )
		std::cout << "STATUS_OK" << std::endl;
	else
	{
		if( (status & STATUS_SOME_TERRAIN_ELEV_DATA_MISSING) != 0 )
			std::cout << "STATUS_SOME_TERRAIN_ELEV_DATA_MISSING" << std::endl;
		if( (status & STATUS_NO_TERRAIN_ELEV_DATA) != 0 )
			std::cout << "STATUS_NO_TERRAIN_ELEV_DATA" << std::endl;
		if( (status & STATUS_SOME_LAND_COVER_DATA_MISSING) != 0 )
			std::cout << "STATUS_SOME_LAND_COVER_DATA_MISSING" << std::endl;
		if( (status & STATUS_NO_LAND_COVER_DATA) != 0 )
			std::cout << "STATUS_NO_LAND_COVER_DATA" << std::endl;
	}
}

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - Secondary Terrain Elevation Source" << std::endl;

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
	sim->SetReceptionAreaCorners(45.5140, -75.5110, 45.5653, -75.44194);
	sim->SetReceptionAreaNumHorizontalPoints(100);
	sim->SetReceptionAreaNumVerticalPoints(100);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	// Set contour values and colors when exporting results to .mif or .kml files
	sim->ClearCoverageDisplayFills();
	sim->AddCoverageDisplayFill(45, 60, 0x5555FF);
	sim->AddCoverageDisplayFill(60, 75, 0x0000FF);
	sim->AddCoverageDisplayFill(75, 300, 0x000088);

	std::cout << "Generating coverage results using HRDEM only..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToKmlFile("hrdem-only-coverage.kml");
	sim->ExportReceptionAreaResultsToBilFile("hrdem-only-coverage.bil");
	sim->ExportReceptionAreaTerrainElevationToBilFile("hrdem-only-terrain.bil", 1000, 1000, false);
	outputStatusMsg(sim->GetGenerateStatus()); // use GetGenerateStatus() to check if any terrain elevation data is missing

	std::cout << "Adding CDEM as secondary terrain elevation source" << std::endl;
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");

	std::cout << "Generating coverage results using HRDEM as primary source and CDEM as secondary source..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToKmlFile("hrdem-and-cdem-coverage.kml");
	sim->ExportReceptionAreaResultsToBilFile("hrdem-and-cdem-coverage.bil");
	sim->ExportReceptionAreaTerrainElevationToBilFile("hrdem-and-cdem-terrain.bil", 1000, 1000, false);
	outputStatusMsg(sim->GetGenerateStatus());

	// Optionally, a third terrain elevation source could be added using sim->SetTertiaryElevationDataSource()

	sim->Release();

	std::cout << "Simulations completed" << std::endl;

	return 0;
}