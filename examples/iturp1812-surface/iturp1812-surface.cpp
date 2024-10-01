#include <stdio.h>
#include <iostream>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - ITU-R P. 1812 propagation model - using surface elevation data" << std::endl;

	if( SetITUProprietaryDataDirectory("../../data/itu-proprietary") != true )
		std::cout << std::endl << "*** Warning: failed to read ITU data" << std::endl;

	sim = NewSimulation();

	// Set transmitter parameters
	sim->SetTransmitterLocation(45.536, -75.493);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters
	sim->SetReceiverHeightAboveGround(1.5);

	// Propagation model selection
	sim->SetPropagationModel(ITU_R_P_1812);

	// Set ITU-R P.1812 propagation model parameters
	sim->SetITURP1812TimePercentage(50);
	sim->SetITURP1812LocationPercentage(50);
	sim->SetITURP1812AverageRadioRefractivityLapseRate(AUTOMATIC); // use ITU digital map (DN50.TXT)
	sim->SetITURP1812SeaLevelSurfaceRefractivity(AUTOMATIC); // use ITU digital map (N050.TXT)
	sim->SetITURP1812PredictionResolution(100); // Width (in meters) of the square area over which the variability applies (see Annex 1, Section 4.7 of ITU-R P.1812 recommendation)
	sim->SetITURP1812SurfaceProfileMethod(P1812_USE_SURFACE_ELEV_DATA); // using surface elevation data rather than clutter data

	// Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_HRDEM_DTM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_HRDEM_DTM, "../../data/terrain-elev-samples/NRCAN_HRDEM_DTM");
	sim->SetTerrainElevDataSamplingResolution(10);

	// Set surface elevation data parameters (see Annex 1, section 3.2.2 of ITU-R P.1812-7)
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_HRDEM_DSM);
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_HRDEM_DSM, "../../data/surface-elev-samples/NRCAN_HRDEM_DSM");

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.515, -75.512, 45.557, -75.474);
	sim->SetReceptionAreaNumHorizontalPoints(200);
	sim->SetReceptionAreaNumVerticalPoints(200);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	std::cout << "Generating and exporting coverage results ..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToBilFile("iturp1812-surface.bil");

	sim->Release();

	std::cout << "Simulation completed" << std::endl;

	return 0;
}