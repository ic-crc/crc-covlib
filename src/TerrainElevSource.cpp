#include "TerrainElevSource.h"



TerrainElevSource::TerrainElevSource()
{
    pInterpolationType = BILINEAR;
}

TerrainElevSource::~TerrainElevSource()
{
    
}

void TerrainElevSource::SetInterpolationType(Interpolation i)
{
	pInterpolationType = i;
}

TerrainElevSource::Interpolation TerrainElevSource::GetInterpolationType() const
{
    return pInterpolationType;
}