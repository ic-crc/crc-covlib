/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "SurfaceElevSource.h"



SurfaceElevSource::SurfaceElevSource()
{
    pInterpolationType = BILINEAR;
}

SurfaceElevSource::~SurfaceElevSource()
{
    
}

void SurfaceElevSource::SetInterpolationType(Interpolation i)
{
	pInterpolationType = i;
}

SurfaceElevSource::Interpolation SurfaceElevSource::GetInterpolationType() const
{
    return pInterpolationType;
}