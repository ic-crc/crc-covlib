#include <stdio.h>
#include <iostream>
#include "../../src/CRC-COVLIB.h"


int main(int argc, char* argv[])
{
Crc::Covlib::ISimulation* sim;

	sim = Crc::Covlib::NewSimulation();

	std::cout << std::endl << "Hello crc-covlib (static linking)" << std::endl;
	std::cout << "Default transmitter height (m): " << sim->GetTransmitterHeight() << std::endl;
	sim->SetTransmitterHeight(33);
	std::cout << "New transmitter height (m): " << sim->GetTransmitterHeight() << std::endl;

	sim->Release();

	return 0;
}