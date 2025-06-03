/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "AntennaPattern.h"
#include "CRC-COVLIB.h"

class CommTerminal
{
public:
	CommTerminal(void);
	virtual ~CommTerminal(void);

	AntennaPattern antPattern;
	double maxGain_dBi;
	double bearingDeg;
	Crc::Covlib::BearingReference bearingRef;
	Crc::Covlib::PatternApproximationMethod patternApproxMethod;
	double losses_dB;
};

class Transmitter : public CommTerminal
{
public:
	Transmitter(void);
	virtual ~Transmitter(void);

	enum class PowerUnit
	{
		WATT = 1,
		DBW  = 2,
		DBM  = 3
	};

	double tpo(PowerUnit unit) const;
	double erp(PowerUnit unit) const;
	double eirp(PowerUnit unit) const;

	double lat; // latitude (degrees)
	double lon; // longitude (degrees)
	double rcagl; // radiation center height above ground level (meters)
	double freqMHz; // frequency (MHz)
	Crc::Covlib::PowerType powerType;
	double power_watts; // may be TPO, ERP or EIRP depending on powerType (watts)
	Crc::Covlib::Polarization pol;
};

class Receiver : public CommTerminal
{
public:
	Receiver(void);
	virtual ~Receiver(void);

	double heightAGL; // height above ground level (m)
};