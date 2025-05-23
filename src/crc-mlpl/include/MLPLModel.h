#pragma once
#include "crc-mlpl.h"
#include <fdeep/fdeep.hpp> // https://github.com/Dobiasd/frugally-deep


class MLPLModel : public IMLPLModel
{
public:
	MLPLModel(void);
	MLPLModel(const MLPLModel& original);
	virtual ~MLPLModel(void);

	const MLPLModel& operator=(const MLPLModel& original);

	virtual float ExcessLoss(float frequency_MHz, float distance_m, float obstructionDepth_m);
	virtual void Release();

private:
	fdeep::model fdeepModel_;

	static const char* MLPL_JSON;
};
