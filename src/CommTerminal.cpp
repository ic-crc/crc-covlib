#include "CommTerminal.h"
#include <math.h>

using namespace Crc::Covlib;


// CommTerminal Class

CommTerminal::CommTerminal(void)
{
}

CommTerminal::~CommTerminal(void)
{
}


// Transmitter Class

Transmitter::Transmitter(void)
{
	lat = 45;
	lon = -75;
	rcagl = 50;
	freqMHz = 100;
	powerType = EIRP;
	power_watts = 1000;
	losses_dB = 0;
	pol = VERTICAL_POL;
	maxGain_dBi = 0;
	bearingDeg = 0;
	bearingRef = TRUE_NORTH;
	patternApproxMethod = HYBRID;
}

Transmitter::~Transmitter(void)
{
}

// see https://en.wikipedia.org/wiki/Effective_radiated_power
double Transmitter::tpo(PowerUnit unit) const
{
double tpo_dBW;

	if( powerType == TPO )
		tpo_dBW = 10*log10(power_watts);
	else if( powerType == ERP )
	{	
		double erp_dBW = 10*log10(power_watts);
		tpo_dBW = erp_dBW + losses_dB - maxGain_dBi + 2.15;
	}
	else if( powerType == EIRP )
	{
		double eirp_dBW = 10*log10(power_watts);
		tpo_dBW = eirp_dBW + losses_dB - maxGain_dBi;
	}
	else
		return 0;

	if( unit == PowerUnit::WATT )
		return pow(10.0, tpo_dBW/10.0);
	else if( unit == PowerUnit::DBW )
		return tpo_dBW;
	else if( unit == PowerUnit::DBM)
		return tpo_dBW + 30;
	else
		return 0;
}
	
double Transmitter::erp(PowerUnit unit) const
{
double erp_dBW;

	if( powerType == TPO )
	{
		double tpo_dBW = 10*log10(power_watts);
		erp_dBW = tpo_dBW - losses_dB + maxGain_dBi - 2.15;
	}
	else if( powerType == ERP )
		erp_dBW = 10*log10(power_watts);
	else if( powerType == EIRP )
	{
		double eirp_dBW = 10*log10(power_watts);
		erp_dBW = eirp_dBW - 2.15;
	}
	else
		return 0;

	if( unit == PowerUnit::WATT )
		return pow(10.0, erp_dBW/10.0);
	else if( unit == PowerUnit::DBW )
		return erp_dBW;
	else if( unit == PowerUnit::DBM)
		return erp_dBW + 30;
	else
		return 0;
}

double Transmitter::eirp(PowerUnit unit) const
{
double eirp_dBW;

	if( powerType == TPO )
	{
		double tpo_dBW = 10*log10(power_watts);
		eirp_dBW = tpo_dBW - losses_dB + maxGain_dBi;
	}
	else if( powerType == ERP )
	{
		double erp_dBW = 10*log10(power_watts);
		eirp_dBW = erp_dBW + 2.15;
	}
	else if( powerType == EIRP )
		eirp_dBW = 10*log10(power_watts);
	else
		return 0;

	if( unit == PowerUnit::WATT )
		return pow(10.0, eirp_dBW/10.0);
	else if( unit == PowerUnit::DBW )
		return eirp_dBW;
	else if( unit == PowerUnit::DBM)
		return eirp_dBW + 30;
	else
		return 0;
}


// Receiver Class

Receiver::Receiver(void)
{
	heightAGL = 1.5;
	losses_dB = 0;
	maxGain_dBi = 0;
	bearingDeg = 0;
	bearingRef = OTHER_TERMINAL;
	patternApproxMethod = HYBRID;
}

Receiver::~Receiver(void)
{
}