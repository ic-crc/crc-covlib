/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "crc-ml.h"
#include "MLPLModel.h"
#include "PathObscuraModel.h"


IMLPLModel* NewMLPLModel()
{
	return new MLPLModel;
}

IPathObscuraModel* NewPathObscuraModel()
{
	return new PathObscuraModel();
}
