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