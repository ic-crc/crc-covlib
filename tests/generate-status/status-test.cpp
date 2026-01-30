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
using namespace std;

void printStatus(int status)
{
	if( status == STATUS_OK )
		cout << "STATUS_OK" << endl;
	else
	{
		if( (status & STATUS_SOME_TERRAIN_ELEV_DATA_MISSING) != 0 )
			cout << "STATUS_SOME_TERRAIN_ELEV_DATA_MISSING" << endl;
		if( (status & STATUS_NO_TERRAIN_ELEV_DATA) != 0 )
			cout << "STATUS_NO_TERRAIN_ELEV_DATA" << endl;
		if( (status & STATUS_SOME_LAND_COVER_DATA_MISSING) != 0 )
			cout << "STATUS_SOME_LAND_COVER_DATA_MISSING" << endl;
		if( (status & STATUS_NO_LAND_COVER_DATA) != 0 )
			cout << "STATUS_NO_LAND_COVER_DATA" << endl;
		if( (status & STATUS_SOME_ITU_RCZ_DATA_MISSING) != 0 )
			cout << "STATUS_SOME_ITU_RCZ_DATA_MISSING" << endl;
		if( (status & STATUS_NO_ITU_RCZ_DATA) != 0 )
			cout << "STATUS_NO_ITU_RCZ_DATA" << endl;
		if( (status & STATUS_SOME_SURFACE_ELEV_DATA_MISSING) != 0 )
			cout << "STATUS_SOME_SURFACE_ELEV_DATA_MISSING" << endl;
		if( (status & STATUS_NO_SURFACE_ELEV_DATA) != 0 )
			cout << "STATUS_NO_SURFACE_ELEV_DATA" << endl;
	}
}

void centerTxOnRxArea(ISimulation* sim)
{
double lat0, lat1, lon0, lon1;

	lat0 = sim->GetReceptionAreaLowerLeftCornerLatitude();
	lon0 = sim->GetReceptionAreaLowerLeftCornerLongitude();
	lat1 = sim->GetReceptionAreaUpperRightCornerLatitude();
	lon1 = sim->GetReceptionAreaUpperRightCornerLongitude();
	sim->SetTransmitterLocation((lat0+lat1)/2.0, (lon0+lon1)/2.0);
}

void setToFullyCoveredByCdem(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.25, -75.5, 45.5, -75.25);
	centerTxOnRxArea(sim);
}

void setToPartiallyCoveredByCdem(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.75, -75.25, 46.25, -74.75);
	centerTxOnRxArea(sim);
}

void setToFullyUncoveredByCdem(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(46.25, -74.75, 46.5, -74.5);
	centerTxOnRxArea(sim);
}

void setToFullyCoveredByCdsm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.3, -75.9, 45.4, -75.6);
	centerTxOnRxArea(sim);
}

void setToPartiallyCoveredByCdsm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.3, -75.9, 45.55, -75.6);
	centerTxOnRxArea(sim);
}

void setToFullyUncoveredByCdsm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.55, -75.9, 45.7, -75.6);
	centerTxOnRxArea(sim);
}

void setToFullyCoveredByBothCdemAndCdsm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.3, -75.9, 45.4, -75.6);
	centerTxOnRxArea(sim);
}

void setToFullyCoveredByCdemAndPartiallyCoveredByCsdm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.3, -75.9, 45.55, -75.6);
	centerTxOnRxArea(sim);
}

void setToFullyCoveredByCdemAndFullyUncoveredByCsdm(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.55, -75.9, 45.7, -75.6);
	centerTxOnRxArea(sim);
}

void setToFullyCoveredByEsaWorldCover(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.25, -75.5, 45.5, -75.25);
	centerTxOnRxArea(sim);
}

void setToPartiallyCoveredByEsaWorldCover(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.75, -75.25, 46.25, -74.75);
	centerTxOnRxArea(sim);
}

void setToFullyUncoveredByEsaWorldCover(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(46.25, -74.75, 46.5, -74.5);
	centerTxOnRxArea(sim);
}

void setToPartiallyCoveredByBothCdemAndEsaWorldCover(ISimulation* sim)
{
	sim->SetReceptionAreaCorners(45.75, -75.25, 46.25, -74.75);
	centerTxOnRxArea(sim);
}

void printTestResult(int status, int expectedStatus, int& passedCount, int& failedCount)
{
	printStatus(status);
	if( status == expectedStatus )
	{
		cout << "PASSED" << endl;
		passedCount++;	
	}
	else
	{
		cout << "FAILED" << endl;
		failedCount++;
	}
}


int main(int argc, char* argv[])
{
int status;
ISimulation* sim = NewSimulation();
int numPassed = 0;
int numFailed = 0;

	SetITUProprietaryDataDirectory("../../data/itu-proprietary");

	sim->SetReceptionAreaNumHorizontalPoints(10);
	sim->SetReceptionAreaNumVerticalPoints(10);

	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_SRTM, "../../data/surface-elev-samples/SRTMGL30");
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_HRDEM_DTM, "../../data/terrain-elev-samples/NRCAN_HRDEM_DTM");
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_CDSM, "../../data/surface-elev-samples/NRCAN_CDSM");
	sim->SetSurfaceElevDataSourceDirectory(SURF_ELEV_NRCAN_HRDEM_DSM, "../../data/surface-elev-samples/NRCAN_HRDEM_DSM");
	sim->SetLandCoverDataSourceDirectory(LAND_COVER_ESA_WORLDCOVER, "../../data/land-cover-samples/ESA_Worldcover");


	cout << endl << "TEST: fully covered by terrain elev data" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetTertiaryTerrainElevDataSource(TERR_ELEV_NONE);
	setToFullyCoveredByCdem(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: partially covered by terrain elev data" << endl;
	setToPartiallyCoveredByCdem(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING, numPassed, numFailed);
	

	cout << endl << "TEST: not covered at all by terrain elev data" << endl;
	setToFullyUncoveredByCdem(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING + STATUS_NO_TERRAIN_ELEV_DATA, numPassed, numFailed);

	
	cout << endl << "TEST: not covered at all by terrain elev data BUT not using any terrain elev source" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NONE);
	setToFullyUncoveredByCdem(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: partially covered by primary terrain elev source BUT fully by secondary source" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_SRTM);
	setToPartiallyCoveredByCdem(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	sim->SetPropagationModel(ITU_R_P_1812);

	cout << endl << "TEST: fully covered by land cover data" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetTertiaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_ESA_WORLDCOVER);
	sim->SetSecondaryLandCoverDataSource(LAND_COVER_NONE);
	setToFullyCoveredByEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: partially covered by land cover data" << endl;
	setToPartiallyCoveredByEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_LAND_COVER_DATA_MISSING, numPassed, numFailed);


	cout << endl << "TEST: not covered at all by land cover data" << endl;
	setToFullyUncoveredByEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_LAND_COVER_DATA_MISSING + STATUS_NO_LAND_COVER_DATA, numPassed, numFailed);

	
	cout << endl << "TEST: fully covered by land cover data BUT no mappings" << endl;
	sim->ClearLandCoverClassMappings(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812);
	setToFullyCoveredByEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_LAND_COVER_DATA_MISSING + STATUS_NO_LAND_COVER_DATA, numPassed, numFailed);


	cout << endl << "TEST: partially covered by both terrain elev and land cover data" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetDefaultLandCoverClassMapping(LAND_COVER_ESA_WORLDCOVER, ITU_R_P_1812, P1812_OPEN_RURAL);
	setToPartiallyCoveredByBothCdemAndEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING + STATUS_SOME_LAND_COVER_DATA_MISSING, numPassed, numFailed); 


	sim->SetPropagationModel(LONGLEY_RICE);

	cout << endl << "TEST: partially covered by terrain elev and land cover data BUT propag model makes no use of land cover data" << endl;
	setToPartiallyCoveredByBothCdemAndEsaWorldCover(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING, numPassed, numFailed); // i.e. should not mention missing land cover data


	sim->SetPropagationModel(ITU_R_P_1812);
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetTertiaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetPrimaryLandCoverDataSource(LAND_COVER_NONE);
	sim->SetSecondaryLandCoverDataSource(LAND_COVER_NONE);

	cout << endl << "TEST: using ITU radio climatic zones file in covered region" << endl;
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");
	sim->SetTransmitterLocation(45, -75);
	sim->GenerateReceptionPointResult(45.001, -75.001);
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: using wrong path for ITU radio climatic zones file" << endl;
	sim->SetITURP1812RadioClimaticZonesFile("../../dataA/itu-radio-climatic-zones/rcz.tif");
	sim->GenerateReceptionPointResult(45.001, -75.001);
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_ITU_RCZ_DATA_MISSING + STATUS_NO_ITU_RCZ_DATA, numPassed, numFailed);


	cout << endl << "TEST: using ITU radio climatic zones file in uncovered region" << endl;
	sim->SetITURP1812RadioClimaticZonesFile("../../data/itu-radio-climatic-zones/rcz.tif");
	sim->SetTransmitterLocation(-65, -75);
	sim->GenerateReceptionPointResult(-65.001, -75.001);
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_ITU_RCZ_DATA_MISSING + STATUS_NO_ITU_RCZ_DATA, numPassed, numFailed);


	sim->SetPropagationModel(ITU_R_P_1812);
	sim->SetITURP1812SurfaceProfileMethod(P1812_USE_SURFACE_ELEV_DATA);

	cout << endl << "TEST: fully covered by surface elev data" << endl;
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_CDSM);
	sim->SetSecondarySurfaceElevDataSource(SURF_ELEV_NONE);
	sim->SetTertiarySurfaceElevDataSource(SURF_ELEV_NONE);
	sim->SetSurfaceAndTerrainDataSourcePairing(false); // no pairing
	setToFullyCoveredByCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: partially covered by surface elev data" << endl;
	setToPartiallyCoveredByCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_SURFACE_ELEV_DATA_MISSING, numPassed, numFailed);
	

	cout << endl << "TEST: not covered at all by surface elev data" << endl;
	setToFullyUncoveredByCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_SURFACE_ELEV_DATA_MISSING + STATUS_NO_SURFACE_ELEV_DATA, numPassed, numFailed);


	cout << endl << "TEST: not covered at all by surface elev data BUT not using any surface elev source" << endl;
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NONE);
	setToFullyUncoveredByCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	// pairing mode: paired terran and surface elev data need to be both present for any of them to be used and considered present
	sim->SetSurfaceAndTerrainDataSourcePairing(true);
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NRCAN_CDSM);
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);

	cout << endl << "TEST: fully covered by both terrain elev and surface elev data with pairing ON" << endl;
	setToFullyCoveredByBothCdemAndCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_OK, numPassed, numFailed);


	cout << endl << "TEST: fully covered by terrain elev data but partially covered by surface elev data with pairing ON" << endl;
	setToFullyCoveredByCdemAndPartiallyCoveredByCsdm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING + STATUS_SOME_SURFACE_ELEV_DATA_MISSING, numPassed, numFailed);


	cout << endl << "TEST: fully covered by terrain elev data but not covered at all by surface elev data with pairing ON" << endl;
	setToFullyCoveredByCdemAndFullyUncoveredByCsdm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING + STATUS_SOME_SURFACE_ELEV_DATA_MISSING + STATUS_NO_TERRAIN_ELEV_DATA + STATUS_NO_SURFACE_ELEV_DATA, numPassed, numFailed);


	cout << endl << "TEST: fully covered by both terrain elev (Primary source) and surface elev data (secondary source), with pairing ON (but no pair formed)" << endl;
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetSecondaryTerrainElevDataSource(TERR_ELEV_NONE);
	sim->SetPrimarySurfaceElevDataSource(SURF_ELEV_NONE);
	sim->SetSecondarySurfaceElevDataSource(SURF_ELEV_NRCAN_CDSM);
	setToFullyCoveredByBothCdemAndCdsm(sim);
	sim->GenerateReceptionAreaResults();
	status = sim->GetGenerateStatus();
	printTestResult(status, STATUS_SOME_TERRAIN_ELEV_DATA_MISSING + STATUS_SOME_SURFACE_ELEV_DATA_MISSING + STATUS_NO_TERRAIN_ELEV_DATA + STATUS_NO_SURFACE_ELEV_DATA, numPassed, numFailed);


	std::cout << std::endl;
	std::cout << "PASSED: " << numPassed << std:: endl;
	std::cout << "FAILED: " << numFailed << std:: endl;

	sim->Release();

	return 0;
}