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


void SetReceptionAreaParams(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(120);
	sim->SetReceptionAreaNumVerticalPoints(120);
}

void SetSimulationParams(ISimulation* sim, double sampleResolution_m)
{
	// Set transmitter parameters
	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters
	sim->SetReceiverHeightAboveGround(1.5);

	// Propagation model selection
	sim->SetPropagationModel(ITU_R_P_1812);

	// Specify file to get ITU radio climate zones from
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	sim->SetTerrainElevDataSamplingResolution(sampleResolution_m);

	// Set land cover data parameters
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER);
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");

	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	SetReceptionAreaParams(sim);
}

int main(int argc, char* argv[])
{
ISimulation* sim25m;
ISimulation* sim50m;
ISimulation* simsDiff;
int numHPoints, numVPoints;
double diff;

	std::cout << std::endl << "crc-covlib - Area results comparison" << std::endl;

	if( SetITUProprietaryDataDirectory("../../data/itu-proprietary") != true )
		std::cout << std::endl << "*** Warning: failed to read ITU data" << std::endl;

	sim25m = NewSimulation();
	SetSimulationParams(sim25m, 25);

	sim50m = NewSimulation();
	SetSimulationParams(sim50m, 50);

	std::cout << std::endl << "Running simulation with 25 meters terrain sampling resolution..." << std::endl;
	sim25m->GenerateReceptionAreaResults();

	std::cout << std::endl << "Running simulation with 50 meters terrain sampling resolution..." << std::endl;
	sim50m->GenerateReceptionAreaResults();

	// The simsDiff simulation object will be used to hold comparison results (i.e. the difference) between the
	// two previous simulations
	simsDiff = NewSimulation();
	SetReceptionAreaParams(simsDiff);

	std::cout << std::endl << "Calculating difference between simulations..." << std::endl;
	numHPoints = simsDiff->GetReceptionAreaNumHorizontalPoints();
	numVPoints = simsDiff->GetReceptionAreaNumVerticalPoints();
	for(int x=0 ; x<numHPoints ; x++)
	{
		for(int y=0 ; y<numVPoints ; y++)
		{
			diff = sim25m->GetReceptionAreaResultValue(x, y) - sim50m->GetReceptionAreaResultValue(x, y);
			simsDiff->SetReceptionAreaResultValue(x, y, diff);
		}
	}

	// Export simulation results and difference between simulations as raster files
	sim25m->ExportReceptionAreaResultsToBilFile("sim25m.bil");
	sim50m->ExportReceptionAreaResultsToBilFile("sim50m.bil");
	simsDiff->ExportReceptionAreaResultsToBilFile("difference.bil");

	// Export difference between simulations as vector file
	//   - shades of red when sim25m result is greater than corresponding sim50m result
	//   - shades of blue when sim25m result is smaller than corresponding sim50m result
	simsDiff->ClearCoverageDisplayFills();
	simsDiff->AddCoverageDisplayFill(0, 1, 0xFFD0D0);
	simsDiff->AddCoverageDisplayFill(1, 3, 0xFF9090);
	simsDiff->AddCoverageDisplayFill(3, 10, 0xFF4444);
	simsDiff->AddCoverageDisplayFill(10, 100, 0xFF0000);
	simsDiff->AddCoverageDisplayFill(0, -1, 0xD0D0FF);
	simsDiff->AddCoverageDisplayFill(-1, -3, 0x9090FF);
	simsDiff->AddCoverageDisplayFill(-3, -10, 0x4444FF);
	simsDiff->AddCoverageDisplayFill(-10, -100, 0x0000FF);
	simsDiff->ExportReceptionAreaResultsToKmlFile("difference.kml", 50, 50, "dB");

	sim25m->Release();
	sim50m->Release();
	simsDiff->Release();

	std::cout << std::endl << "Simulations completed" << std::endl;

	return 0;
}