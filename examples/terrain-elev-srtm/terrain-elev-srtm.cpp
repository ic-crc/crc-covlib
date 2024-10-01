#include <stdio.h>
#include <iostream>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - SRTM" << std::endl;

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
	// NOTE: SRTM data samples contain surface elevation data (terrain elevation + clutter height with no 
	//       distinction between the two). Therefore the calculated field strength should be generally lesser
	//       than when using true terrain elevation (ground level) data.
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_SRTM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_SRTM, "../../data/surface-elev-samples/SRTMGL1");
	// One terrain elevation value every 50m in the terrain profiles that will be provided to Longley-Rice
	sim->SetTerrainElevDataSamplingResolution(50);

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(140);
	sim->SetReceptionAreaNumVerticalPoints(140);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	// Set contour values and colors when exporting results to .mif or .kml files
	sim->ClearCoverageDisplayFills();
	sim->AddCoverageDisplayFill(45, 60, 0x5555FF);
	sim->AddCoverageDisplayFill(60, 75, 0x0000FF);
	sim->AddCoverageDisplayFill(75, 300, 0x000088);

	std::cout << "Generating and exporting coverage results..." << std::endl;

	sim->GenerateReceptionAreaResults();

	// Export results in various formats
	sim->ExportReceptionAreaResultsToTextFile("terrain-elev-srtm.txt");
	sim->ExportReceptionAreaResultsToMifFile("terrain-elev-srtm.mif");
	sim->ExportReceptionAreaResultsToKmlFile("terrain-elev-srtm.kml");
	sim->ExportReceptionAreaResultsToBilFile("terrain-elev-srtm.bil");

	sim->Release();

	std::cout << "Simulation completed" << std::endl;

	return 0;
}