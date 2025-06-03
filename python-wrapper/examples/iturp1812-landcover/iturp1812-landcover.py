# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R P. 1812 propagation model - using land cover data')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.536, -75.493)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)

    # Set ITU-R P.1812 propagation model parameters.
    # Note that specified values below correspond to default crc-covlib's values. Those methods don't need to
    # be called unless default values need to be changed. They are called here as a demonstration.
    sim.SetITURP1812TimePercentage(50)
    sim.SetITURP1812LocationPercentage(50)
    sim.SetITURP1812AverageRadioRefractivityLapseRate(covlib.AUTOMATIC) # use ITU digital map (DN50.TXT)
    sim.SetITURP1812SeaLevelSurfaceRefractivity(covlib.AUTOMATIC) # use ITU digital map (N050.TXT)
    sim.SetITURP1812PredictionResolution(100) # Width (in meters) of the square area over which the variability applies (see Annex 1, Section 4.7 of ITU-R P.1812 recommendation)
    sim.SetITURP1812LandCoverMappingType(covlib.P1812LandCoverMappingType.P1812_MAP_TO_CLUTTER_CATEGORY)
    CLUT = covlib.P1812ClutterCategory
    sim.SetITURP1812RepresentativeClutterHeight(CLUT.P1812_WATER_SEA, 0) # Values from TABLE 2, Section 3.2 of ITU-R P.1812 recommendation
    sim.SetITURP1812RepresentativeClutterHeight(CLUT.P1812_OPEN_RURAL, 0)
    sim.SetITURP1812RepresentativeClutterHeight(CLUT.P1812_SUBURBAN, 10)
    sim.SetITURP1812RepresentativeClutterHeight(CLUT.P1812_URBAN_TREES_FOREST, 15)
    sim.SetITURP1812RepresentativeClutterHeight(CLUT.P1812_DENSE_URBAN, 20)

    # Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    HRDEM_DTM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM_DTM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSamplingResolution(25)

    # Set land cover data parameters
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER # data from https://esa-worldcover.org/
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    # Define mapping of ESA WorldCover's land cover classes to ITU-R P.1812's clutter categories.
    # Mapping below corresponds to default crc-covlib mapping, so again those methods don't really
    # need to be called unless these default values need to be changed.
    # See https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf for a definition of ESA WorldCover classes.
    sim.ClearLandCoverClassMappings(WORLDCOVER, P1812) # delete existing default mapping
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P1812, CLUT.P1812_URBAN_TREES_FOREST) # map 'Tree cover' (10) to 'urban/trees/forest' (4)
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P1812, CLUT.P1812_URBAN_TREES_FOREST) # map 'Built-up' (50) to 'urban/trees/forest' (4)
    sim.SetLandCoverClassMapping(WORLDCOVER, 80, P1812, CLUT.P1812_WATER_SEA) # map 'Permanent water bodies' (80) to 'water/sea' (1)
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P1812, CLUT.P1812_OPEN_RURAL) # map all other ESA WorldCover classes to 'open/rural' (2)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.515, -75.512, 45.557, -75.474)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting first coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp1812-cluttercat.bil'))

    # Alternately, we can directly map ESA WorldCover's land cover classes to representative clutter heights
    # and avoid using ITU-R P.1812's clutter categories.
    sim.SetITURP1812LandCoverMappingType(covlib.P1812LandCoverMappingType.P1812_MAP_TO_REPR_CLUTTER_HEIGHT)
    sim.ClearLandCoverClassMappings(WORLDCOVER, P1812)
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P1812, 15) # map 'Tree cover' (10) to a representative clutter height of 15m
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P1812, 15) # map 'Built-up' (50) to a representative clutter height of 15m
    sim.SetLandCoverClassMapping(WORLDCOVER, 80, P1812, 0) # map 'Permanent water bodies' to a representative clutter height of 0m
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P1812, 0) # map all other ESA WorldCover classes to a representative clutter height of 0m

    print('Generating and exporting second coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp1812-reprheight.bil'))

    print('Simulations completed\n')
