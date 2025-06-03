/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <iostream>
#include <fstream>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;

void LoadRadioMobileV3File(ISimulation* sim, Terminal terminal, const char* pathname);

int main(int argc, char* argv[])
{
ISimulation* sim;

	std::cout << std::endl << "crc-covlib - antenna pattern usage" << std::endl;

	sim = NewSimulation();

	// Set transmitter parameters
	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterFrequency(2600);

	// Set receiver parameters
	sim->SetReceiverHeightAboveGround(1.0);

	// Set antenna parameters (at transmitter):
	// Load antenna pattern file
	LoadRadioMobileV3File(sim, TRANSMITTER, "generic_antenna.ant");
	// Make sure the antenna patterns are normalized (although in this particular case this does not
	// have any impact since the patterns in the generic_antenna.ant file are already normalized).
	sim->NormalizeAntennaHorizontalPattern(TRANSMITTER);
	sim->NormalizeAntennaVerticalPattern(TRANSMITTER);
	// Set max antenna gain to 16 dBi
	sim->SetAntennaMaximumGain(TRANSMITTER, 16);
	// Points antenna towards east (0=north, 90=east, 180=south, 270=west)
	sim->SetAntennaBearing(TRANSMITTER, TRUE_NORTH, 90);
	// Set antenna tilt (elec. or mech.) (-90=zenith, 0=horizon, +90=nadir)
	sim->SetAntennaElectricalTilt(TRANSMITTER, 0);
	sim->SetAntennaMechanicalTilt(TRANSMITTER, 0);
	// Select method to interpolate antenna gain from horizontal and vertical patterns
	sim->SetAntennaPatternApproximationMethod(TRANSMITTER, HYBRID);

	// Set antenna parameters (at receiver):
	LoadRadioMobileV3File(sim, RECEIVER, "generic_antenna.ant");
	sim->NormalizeAntennaHorizontalPattern(RECEIVER);
	sim->NormalizeAntennaVerticalPattern(RECEIVER);
	sim->SetAntennaMaximumGain(RECEIVER, 8);
	// For every reception point, have the receiver antenna points directly towards the transmitter
	// (0=towards other terminal, 180=away from other terminal).
	sim->SetAntennaBearing(RECEIVER, OTHER_TERMINAL, 0);

	// Select propagation model
	sim->SetPropagationModel(LONGLEY_RICE);

	// Use no terrain elevation to better see the impact of antennas
	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NONE);

	// Set reception/coverage area parameters
	sim->SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225);
	sim->SetReceptionAreaNumHorizontalPoints(200);
	sim->SetReceptionAreaNumVerticalPoints(200);
	
	// Select a result type that takes both transmitter and receiver antennas into account.
	sim->SetResultType(RECEIVED_POWER_DBM);

	// Set contour values and colors when exporting results to .mif or .kml files
	sim->ClearCoverageDisplayFills();
	sim->AddCoverageDisplayFill(-85, -75, 0x5555FF);
	sim->AddCoverageDisplayFill(-75, -65, 0x0000FF);
	sim->AddCoverageDisplayFill(-65, 0, 0x000088);

	std::cout << "Generating results (rx pointing at tx)..." << std::endl;

	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToKmlFile("antenna-pattern-rx-towards-tx.kml");

	std::cout << "Generating results (rx pointing away from tx)..." << std::endl;

	// Now have the receiver antenna point away from the transmitter. The resulting coverage
	// should be much smaller.
	sim->SetAntennaBearing(RECEIVER, OTHER_TERMINAL, 180);
	sim->GenerateReceptionAreaResults();
	sim->ExportReceptionAreaResultsToKmlFile("antenna-pattern-rx-away-from-tx.kml");

	sim->Release();

	std::cout << "Simulations completed" << std::endl;

	return 0;
}

void LoadRadioMobileV3File(ISimulation* sim, Terminal terminal, const char* pathname)
{
std::ifstream file;
double gain_db;

	sim->ClearAntennaPatterns(terminal, true, true);

	file.open(pathname, std::ios::in);
	if(file)
	{
		for(int azm=0 ; azm<360 ; azm++)
		{
			file >> gain_db;
			sim->AddAntennaHorizontalPatternEntry(terminal, azm, gain_db);
		}
		file >> gain_db;
		sim->AddAntennaVerticalPatternEntry(terminal, 0, -90, gain_db);
		sim->AddAntennaVerticalPatternEntry(terminal, 180, -90, gain_db);
		for(int elv=-89 ; elv<90 ; elv++)
		{
			file >> gain_db;
			sim->AddAntennaVerticalPatternEntry(terminal, 0, elv, gain_db);
		}
		file >> gain_db;
		sim->AddAntennaVerticalPatternEntry(terminal, 0, 90, gain_db);
		sim->AddAntennaVerticalPatternEntry(terminal, 180, 90, gain_db);
		for(int elv=89 ; elv>-90 ; elv--)
		{
			file >> gain_db;
			sim->AddAntennaVerticalPatternEntry(terminal, 180, elv, gain_db);
		}
	}
	file.close();
}