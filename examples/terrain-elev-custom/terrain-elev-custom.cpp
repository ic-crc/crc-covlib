#include <iostream>
#include <fstream>
#include "../../src/CRC-COVLIB.h"

using namespace std;
using namespace Crc::Covlib;

int main(int argc, char* argv[])
{
	cout << endl << "crc-covlib - Custom terrain elevation data file" << endl;

	// Preparing the custom terrain elevation data to provide to crc-covlib
	const int SIZE_X = 5448;
	const int SIZE_Y = 2733;
	unsigned int dataSize = SIZE_X*SIZE_Y*sizeof(float);
	ifstream file;
	float* terrainElevData;

	try
	{
		terrainElevData = new float[SIZE_X*SIZE_Y];
	}
	catch(const bad_alloc &)
	{
		cout <<  "Failed to allocate memory for custom terrain elevation data, program terminated" << endl;
		return 1;
	}

	file.open("../../data/terrain-elev-samples/custom/cdem_ottawa_075asecs.float", ios::in|ios::binary);
	if (file.is_open())
	{
		file.read((char*)terrainElevData, dataSize);
		file.close();
	}
	if (file.gcount() != dataSize)
	{
		delete[] terrainElevData;
		cout <<  "Failed to read custom terrain elevation data, program terminated" << endl;
		return 1;
	}

	ISimulation* sim;
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
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_CUSTOM);
	sim->AddCustomTerrainElevData(44.95875000, -76.37583333, 45.52791667, -75.24104167, SIZE_X, SIZE_Y, terrainElevData);

	// terrainElevData can be discarded to free up memory, the sim object keeps a copy of it
	delete[] terrainElevData;
	terrainElevData = NULL;
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

	cout << "Generating and exporting coverage results..." << endl;

	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToTextFile("terrain-elev-custom.txt");
	sim->ExportReceptionAreaResultsToMifFile("terrain-elev-custom.mif");
	sim->ExportReceptionAreaResultsToKmlFile("terrain-elev-custom.kml");
	sim->ExportReceptionAreaResultsToBilFile("terrain-elev-custom.bil");

	sim->Release();

	cout << "Simulation completed" << endl;

	return 0;
}