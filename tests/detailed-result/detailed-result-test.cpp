/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../src/CRC-COVLIB.h"

using namespace Crc::Covlib;


void SetAntennaPattern(ISimulation* sim, Crc::Covlib::Terminal terminal)
{
	sim->AddAntennaHorizontalPatternEntry(terminal, 0, 0); // azm (deg), gain (dB)
	sim->AddAntennaHorizontalPatternEntry(terminal, 60, -6);
	sim->AddAntennaHorizontalPatternEntry(terminal, 180, -30);
	sim->AddAntennaHorizontalPatternEntry(terminal, 300, -6);
	sim->AddAntennaVerticalPatternEntry(terminal, 0, 0, 0); // azm (deg), elev angle (Deg), gain (dB)
	sim->AddAntennaVerticalPatternEntry(terminal, 0, 30, -15);
	sim->AddAntennaVerticalPatternEntry(terminal, 0, -30, -15);
	sim->AddAntennaVerticalPatternEntry(terminal, 0, 90, -40);
	sim->AddAntennaVerticalPatternEntry(terminal, 0, -90, -40);
}

void PrintDetailedResult(ReceptionPointDetailedResult& dr)
{
	std::cout << "result: " << dr.result << std::endl;
	std::cout << "pathLoss_dB: " << dr.pathLoss_dB << std::endl;

	std::cout << "pathLength_km: " << dr.pathLength_km << std::endl;
	std::cout << "transmitterHeightAMSL_m: " << dr.transmitterHeightAMSL_m << std::endl;
	std::cout << "receiverHeightAMSL_m: " << dr.receiverHeightAMSL_m << std::endl;
	std::cout << "transmitterAntennaGain_dBi: " << dr.transmitterAntennaGain_dBi << std::endl;
	std::cout << "receiverAntennaGain_dBi: " << dr.receiverAntennaGain_dBi << std::endl;
	std::cout << "azimuthFromTransmitter_degrees: " << dr.azimuthFromTransmitter_degrees << std::endl;
	std::cout << "azimuthFromReceiver_degrees: " << dr.azimuthFromReceiver_degrees << std::endl;
	std::cout << "elevAngleFromTransmitter_degrees: " << dr.elevAngleFromTransmitter_degrees << std::endl;
	std::cout << "elevAngleFromReceiver_degrees: " << dr.elevAngleFromReceiver_degrees << std::endl;
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
ISimulation* sim;

	SetITUProprietaryDataDirectory("../../data/itu-proprietary");

	sim = NewSimulation();

	sim->SetTransmitterLocation(45.42531, -75.71573);
	sim->SetTransmitterHeight(30);
	sim->SetTransmitterPower(2, EIRP);
	sim->SetTransmitterLosses(2.5);
	sim->SetTransmitterFrequency(2600);

	SetAntennaPattern(sim, TRANSMITTER);
	sim->SetAntennaMaximumGain(TRANSMITTER, 12);
	sim->SetAntennaMechanicalTilt(TRANSMITTER, 3);
	sim->SetAntennaBearing(TRANSMITTER, TRUE_NORTH, 45);
	sim->SetAntennaPatternApproximationMethod(TRANSMITTER, HYBRID);

	double rxLat = 45.53;
	double rxLon = -75.71;
	sim->SetReceiverHeightAboveGround(1.5);
	sim->SetReceiverLosses(1);

	SetAntennaPattern(sim, RECEIVER);
	sim->SetAntennaMaximumGain(RECEIVER, 3);
	sim->SetAntennaBearing(RECEIVER, OTHER_TERMINAL, 0);
	sim->SetAntennaPatternApproximationMethod(TRANSMITTER, SUMMING);

	sim->SetPropagationModel(LONGLEY_RICE);

	sim->SetPrimaryTerrainElevDataSource(TERR_ELEV_NRCAN_CDEM);
	sim->SetTerrainElevDataSourceDirectory(TERR_ELEV_NRCAN_CDEM, "../../data/terrain-elev-samples/NRCAN_CDEM");
	sim->SetTerrainElevDataSamplingResolution(10);

	sim->SetResultType(PATH_LOSS_DB);
	ReceptionPointDetailedResult dr = sim->GenerateReceptionPointDetailedResult(rxLat, rxLon);
	PrintDetailedResult(dr);

	double Ptx_dBm = 10.0*log10(sim->GetTransmitterPower(TPO)) + 30.0;
	double Lp_dB = dr.pathLoss_dB;
	double Gtx_dBi = sim->GetAntennaGain(TRANSMITTER, dr.azimuthFromTransmitter_degrees, dr.elevAngleFromTransmitter_degrees, rxLat, rxLon);
	double Grx_dBi = sim->GetAntennaGain(RECEIVER, dr.azimuthFromReceiver_degrees, dr.elevAngleFromReceiver_degrees, rxLat, rxLon);
	double Ltx_dB = sim->GetTransmitterLosses();
	double Lrx_dB = sim->GetReceiverLosses();
	double f_GHz = sim->GetTransmitterFrequency()/1000.0;

	// TODO tx/rx gain diff ?

	double pathLossDiff = fabs(dr.result - sim->GenerateReceptionPointResult(rxLat, rxLon));

	double fieldStrength_dBuVm = Ptx_dBm - Lp_dB + Gtx_dBi - Ltx_dB + 137.21 + 20.0*log10(f_GHz);
	sim->SetResultType(FIELD_STRENGTH_DBUVM);
	double fieldStrengthDiff = fabs(fieldStrength_dBuVm - sim->GenerateReceptionPointResult(rxLat, rxLon));

	double transmissionLoss_dB = Lp_dB - Gtx_dBi + Ltx_dB - Grx_dBi + Lrx_dB;
	sim->SetResultType(TRANSMISSION_LOSS_DB);
	double transmissionLossDiff = fabs(transmissionLoss_dB - sim->GenerateReceptionPointResult(rxLat, rxLon));

	double receivedPower_dBm = Ptx_dBm - Lp_dB + Gtx_dBi - Ltx_dB + Grx_dBi - Lrx_dB;
	sim->SetResultType(RECEIVED_POWER_DBM);
	double receivedPowerDiff = fabs(receivedPower_dBm - sim->GenerateReceptionPointResult(rxLat, rxLon));

	std::cout << "path loss (db)          diff: " << std::setw(12) << std::left << pathLossDiff << " " << (pathLossDiff < 1E-7 ? "PASSED" : "FAILED") << std::endl;
	std::cout << "field strength (dBuV/m) diff: " << std::setw(12) << std::left << fieldStrengthDiff << " " << (fieldStrengthDiff < 1E-7 ? "PASSED" : "FAILED") << std::endl;
	std::cout << "transmission loss (dB)  diff: " << std::setw(12) << std::left << transmissionLossDiff << " " << (transmissionLossDiff < 1E-7 ? "PASSED" : "FAILED") << std::endl;
	std::cout << "received power (dBm)    diff: " << std::setw(12) << std::left << receivedPowerDiff << " " << (receivedPowerDiff < 1E-7 ? "PASSED" : "FAILED") << std::endl;

	return 0;
}