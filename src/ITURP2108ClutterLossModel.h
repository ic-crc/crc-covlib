/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once

class ITURP2108ClutterLossModel
{
public:
	ITURP2108ClutterLossModel();
	~ITURP2108ClutterLossModel();

	void SetActiveState(bool active);
	bool IsActive() const;

	void SetLocationPercentage(double percent);
	double GetLocationPercentage() const;

	double CalcTerrestrialStatisticalLoss(double frequency_GHz, double distance_km) const;

private:
	bool pIsActive;
	double pLocationPercent;
};