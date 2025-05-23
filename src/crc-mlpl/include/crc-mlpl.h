#pragma once


struct IMLPLModel
{
	virtual void Release() = 0;
	virtual float ExcessLoss(float frequency_MHz, float distance_m, float obstructionDepth_m) = 0;
};

IMLPLModel* NewMLPLModel();
