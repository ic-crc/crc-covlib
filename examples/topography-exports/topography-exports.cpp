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

	std::cout << std::endl << "crc-covlib - Topography Exports" << std::endl;

	sim = NewSimulation();

	// Set terrain elevation data parameters
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);

	// Export terrain elevation data covering the reception area (1000 x 1000 points).
	// The output is a .bil file that can be visualized with a GIS application like QGIS.
	// This allows to verify that crc-covlib was able to read the terrain elevation data and to
	// see whether any section of the terrain is missing.
	// Setting the last parameter to true will use a value of 0 (zero) when no data is available.
	// 0m elevation is the default value used by crc-covlib in terrain profiles when no data is avaiblable.
	sim->ExportReceptionAreaTerrainElevationToBilFile("cdem.bil", 1000, 1000, false);

	// Set land cover data parameters
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER);
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");

	// Export land cover data covering the reception area.
	// Notice the last parameter (mapValues) is set to false. This instructs crc-covlib to export land cover
	// classes as they appear from the source (here ESA WorldCover file(s)), before any mapping is applied.
	sim->ExportReceptionAreaLandCoverClassesToBilFile("worldcover-unmapped.bil", 1000, 1000, false);

	// Select the ITU-R P.1812 propagation model and define the mapping between ESA WorldCover's classes
	// and P.1812's clutter categories. 
	sim->SetPropagationModel(ITU_R_P_1812);
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812); // delete existing default mapping
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 80, ITU_R_P_1812, P1812_WATER_SEA); // map 'Permanent water bodies' (80) to 'water/sea' (1)
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812, P1812_OPEN_RURAL); // map all other ESA WorldCover classes to 'open/rural' (2)

	// The mapValues parameter is now set to true. This instructs crc-covlib to apply any defined mappings between
	// the land cover data and the currently selected propagation model before the export takes place.
	sim->ExportReceptionAreaLandCoverClassesToBilFile("worldcover-mapped-to-P1812-clutter.bil", 1000, 1000, true);

	// With P.1812, we can alternately map land cover classes the representative clutter heights instead of clutter
	// categories.
	sim->SetITURP1812LandCoverMappingType(P1812_MAP_TO_REPR_CLUTTER_HEIGHT);
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812); // delete existing mapping
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 10, ITU_R_P_1812, 15); // map 'Tree cover' (10) to a representative clutter height of 15m
	sim->SetLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, 50, ITU_R_P_1812, 15); // map 'Built-up' (50) to a representative clutter height of 15m
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812, 0); // map all other ESA WorldCover classes to a representative clutter height of 0m

	// The exported data will now contain the representative clutter heights.
	sim->ExportReceptionAreaLandCoverClassesToBilFile("worldcover-mapped-to-repr-heights.bil", 1000, 1000, true);

	// Set surface elevation data parameters
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_CDSM);
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_CDSM, "../../data/surface-elev-samples/NRCAN_CDSM");

	// Export surface elevation data covering the reception area (1000 x 1000 points).
	// Setting the last parameter to true will use a value of 0 (zero) when no data is available.
	// 0m surface elevation is the default value used by crc-covlib in terrain profiles when no data is avaiblable.
	sim->ExportReceptionAreaSurfaceElevationToBilFile("cdsm.bil", 1000, 1000, false);

	sim->Release();

	std::cout << "Topography exports completed" << std::endl;

	return 0;
}