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


int main(int argc, char* argv[])
{
Crc::Covlib::ISimulation* sim;

	sim = Crc::Covlib::NewSimulation();

	std::cout << std::endl << "Hello crc-covlib (dynamic linking)" << std::endl;
	std::cout << "Default transmitter height (m): " << sim->GetTransmitterHeight() << std::endl;
	sim->SetTransmitterHeight(33);
	std::cout << "New transmitter height (m): " << sim->GetTransmitterHeight() << std::endl;

	sim->Release();

	return 0;
}