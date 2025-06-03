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

	std::cout << std::endl << "crc-covlib - ITU-R P. 452-18 propagation model" << std::endl;

	if( SetITUProprietaryDataDirectory("../../data/itu-proprietary") != true )
		std::cout << std::endl << "*** Warning: failed to read ITU data" << std::endl;

	sim = NewSimulation();

	// Set transmitter parameters (i.e. interfering station's parameters)
	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters (i.e. interfered-with station's parameters)
	sim->SetReceiverHeightAboveGround(1.5);

	// Propagation model selection
	sim->SetPropagationModel(ITU_R_P_452_V18);

	// Set ITU-R P.452-18 propagation model parameters
	// Note that specified values below correspond to default crc-covlib's values. Those methods don't need to
    // be called unless default values need to be changed. They are called here as a demonstration.
	sim->SetITURP452TimePercentage(50);
	sim->SetITURP452PredictionType(P452_AVERAGE_YEAR);
	sim->SetITURP452AverageRadioRefractivityLapseRate(AUTOMATIC); // use ITU digital map (DN50.TXT)
	sim->SetITURP452SeaLevelSurfaceRefractivity(AUTOMATIC); // use ITU digital map (N050.TXT)
	sim->SetITURP452AirTemperature(AUTOMATIC); // use annual mean surface temperature from ITU-R P.1510 (T_Annual.TXT)
	sim->SetITURP452AirPressure(AUTOMATIC); // use mean annual global reference atmosphere pressure from ITU-R P.835
	sim->SetITURP452LandCoverMappingType(P452_MAP_TO_CLUTTER_CATEGORY);
	sim->SetITURP452RepresentativeClutterHeight(P452_WATER_SEA, 0); // Values from TABLE 3, Section 3.2.1 of ITU-R P.452-18 recommendation
	sim->SetITURP452RepresentativeClutterHeight(P452_OPEN_RURAL, 0);
	sim->SetITURP452RepresentativeClutterHeight(P452_SUBURBAN, 10);
	sim->SetITURP452RepresentativeClutterHeight(P452_URBAN_TREES_FOREST, 15);
	sim->SetITURP452RepresentativeClutterHeight(P452_DENSE_URBAN, 20);

	// Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
	sim->SetITURP452RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	sim->SetTerrainElevDataSamplingResolution(25); // Note: this sampling resolution also applies to landcover/clutter data.

	// Set land cover data parameters
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER); // data from https://esa-worldcover.org/
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");

	// Some default mappings between ESA WorldCover's land cover classes and ITU-R P.452-18's clutter categories (TABLE 3)
	// are already defined in crc-covlib, however these mappings may be modified when desired. For example:
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_452_V18); // delete existing default mapping
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 10, ITU_R_P_452_V18, P452_URBAN_TREES_FOREST); // map 'Tree cover' (10) to 'urban/trees/forest' (4)
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 50, ITU_R_P_452_V18, P452_URBAN_TREES_FOREST); // map 'Built-up' (50) to 'urban/trees/forest' (4)
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 80, ITU_R_P_452_V18, P452_WATER_SEA); // map 'Permanent water bodies' (80) to 'water/sea' (1)
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_452_V18, P452_OPEN_RURAL); // map all other ESA WorldCover classes to 'open/rural' (2)
	// See https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf for a definition of ESA WorldCover classes.

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(200);
	sim->SetReceptionAreaNumVerticalPoints(200);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);

	std::cout << "Generating and exporting first coverage results ..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToBilFile("iturp452v18.bil");

	// Alternately, we can directly map ESA WorldCover's land cover classes to representative clutter heights
	// and avoid using ITU-R P.452-18's clutter categories.
	sim->SetITURP452LandCoverMappingType(P452_MAP_TO_REPR_CLUTTER_HEIGHT);
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_452_V18);
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 10, ITU_R_P_452_V18, 15); // map 'Tree cover' (10) to a representative clutter height of 15m
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 50, ITU_R_P_452_V18, 15); // map 'Built-up' (50) to a representative clutter height of 15m
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 80, ITU_R_P_452_V18, 0); // map 'Permanent water bodies' to a representative clutter height of 0m
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_452_V18, 0); // map all other ESA WorldCover classes to a representative clutter height of 0m

	std::cout << "Generating and exporting second coverage results ..." << std::endl;
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToBilFile("iturp452v18alt.bil");

	sim->Release();

	std::cout << "Simulations completed" << std::endl;

	return 0;
}