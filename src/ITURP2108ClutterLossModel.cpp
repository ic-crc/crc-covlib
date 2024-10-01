#include "ITURP2108ClutterLossModel.h"
#include "ITURP_2108.h"


ITURP2108ClutterLossModel::ITURP2108ClutterLossModel()
{
	pIsActive = false;
	pLocationPercent = 50;
}

ITURP2108ClutterLossModel::~ITURP2108ClutterLossModel()
{

}

void ITURP2108ClutterLossModel::SetActiveState(bool active)
{
	pIsActive = active;
}

bool ITURP2108ClutterLossModel::IsActive() const
{
	return pIsActive;
}

void ITURP2108ClutterLossModel::SetLocationPercentage(double percent)
{
	if( percent < 0.000001 || percent > 99.999999 )
		return;
	pLocationPercent = percent;
}
	
double ITURP2108ClutterLossModel::GetLocationPercentage() const
{
	return pLocationPercent;
}

double ITURP2108ClutterLossModel::CalcTerrestrialStatisticalLoss(double frequency_GHz, double distance_km) const
{
	if( distance_km >= 0.25 && frequency_GHz >= 0.5 && frequency_GHz <= 67.0 )
		return ITURP_2108::StatisticalClutterLossForTerrestrialPath(frequency_GHz, distance_km, pLocationPercent);
	else
		return 0;
}
