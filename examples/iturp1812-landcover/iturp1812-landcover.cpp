#include <stdio.h>
#include <iostream>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - ITU-R P. 1812 propagation model - using land cover data" << std::endl;

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

	// Set ITU-R P.1812 propagation model parameters.
	// Note that specified values below correspond to default crc-covlib's values. Those methods don't need to
    // be called unless default values need to be changed. They are called here as a demonstration.
	sim->SetITURP1812TimePercentage(50);
	sim->SetITURP1812LocationPercentage(50);
	sim->SetITURP1812AverageRadioRefractivityLapseRate(AUTOMATIC); // use ITU digital map (DN50.TXT)
	sim->SetITURP1812SeaLevelSurfaceRefractivity(AUTOMATIC); // use ITU digital map (N050.TXT)
	sim->SetITURP1812PredictionResolution(100); // Width (in meters) of the square area over which the variability applies (see Annex 1, Section 4.7 of ITU-R P.1812 recommendation)
	sim->SetITURP1812LandCoverMappingType(P1812_MAP_TO_CLUTTER_CATEGORY);
	sim->SetITURP1812RepresentativeClutterHeight(P1812_WATER_SEA, 0); // Values from TABLE 2, Section 3.2 of ITU-R P.1812 recommendation
	sim->SetITURP1812RepresentativeClutterHeight(P1812_OPEN_RURAL, 0);
	sim->SetITURP1812RepresentativeClutterHeight(P1812_SUBURBAN, 10);
	sim->SetITURP1812RepresentativeClutterHeight(P1812_URBAN_TREES_FOREST, 15);
	sim->SetITURP1812RepresentativeClutterHeight(P1812_DENSE_URBAN, 20);

	// Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_HRDEM_DTM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_HRDEM_DTM, "../../data/terrain-elev-samples/NRCAN_HRDEM_DTM");
	sim->SetTerrainElevDataSamplingResolution(25);

	// Set land cover data parameters
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER); // data from https://esa-worldcover.org/
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");

	// Define mapping of ESA WorldCover's land cover classes to ITU-R P.1812's clutter categories.
	// Mapping below corresponds to default crc-covlib mapping, so again those methods don't really
	// need to be called unless these default values need to be changed.
	// See https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf for a definition of ESA WorldCover classes.
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812); // delete existing default mapping
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 10, ITU_R_P_1812, P1812_URBAN_TREES_FOREST); // map 'Tree cover' (10) to 'urban/trees/forest' (4)
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 50, ITU_R_P_1812, P1812_URBAN_TREES_FOREST); // map 'Built-up' (50) to 'urban/trees/forest' (4)
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 80, ITU_R_P_1812, P1812_WATER_SEA); // map 'Permanent water bodies' (80) to 'water/sea' (1)
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812, P1812_OPEN_RURAL); // map all other ESA WorldCover classes to 'open/rural' (2)

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.515, -75.512, 45.557, -75.474);
	sim->SetReceptionAreaNumHorizontalPoints(200);
	sim->SetReceptionAreaNumVerticalPoints(200);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	std::cout << "Generating and exporting first coverage results ..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToBilFile("iturp1812-cluttercat.bil");

	// Alternately, we can directly map ESA WorldCover's land cover classes to representative clutter heights
	// and avoid using ITU-R P.1812's clutter categories.
	sim->SetITURP1812LandCoverMappingType(P1812_MAP_TO_REPR_CLUTTER_HEIGHT);
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812);
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 10, ITU_R_P_1812, 15); // map 'Tree cover' (10) to a representative clutter height of 15m
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 50, ITU_R_P_1812, 15); // map 'Built-up' (50) to a representative clutter height of 15m
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 80, ITU_R_P_1812, 0); // map 'Permanent water bodies' to a representative clutter height of 0m
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812, 0); // map all other ESA WorldCover classes to a representative clutter height of 0m

	std::cout << "Generating and exporting second coverage results ..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToBilFile("iturp1812-reprheight.bil");

	sim->Release();

	std::cout << "Simulations completed" << std::endl;

	return 0;
}