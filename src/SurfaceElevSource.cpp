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