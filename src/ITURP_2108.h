/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once


class ITURP_2108
{
public:
	ITURP_2108();
	~ITURP_2108();

	static double StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent);

private:
	static double _StatisticalClutterLossForTerrestrialPath(double f_GHz, double distance_km, double location_percent);
};