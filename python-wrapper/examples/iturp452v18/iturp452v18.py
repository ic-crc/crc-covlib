# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R P. 452-18 propagation model')

    sim = covlib.Simulation()

    # Set transmitter parameters (i.e. interfering station's parameters)
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters (i.e. interfered-with station's parameters)
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    P452_18 = covlib.PropagationModel.ITU_R_P_452_V18
    sim.SetPropagationModel(P452_18)

    # Set ITU-R P.452-18 propagation model parameters
    # Note that specified values below correspond to default crc-covlib's values. Those methods don't need to
    # be called unless default values need to be changed. They are called here as a demonstration.
    sim.SetITURP452TimePercentage(50)
    sim.SetITURP452PredictionType(covlib.P452PredictionType.P452_AVERAGE_YEAR)
    sim.SetITURP452AverageRadioRefractivityLapseRate(covlib.AUTOMATIC) # use ITU digital map (DN50.TXT)
    sim.SetITURP452SeaLevelSurfaceRefractivity(covlib.AUTOMATIC) # use ITU digital map (N050.TXT)
    sim.SetITURP452AirTemperature(covlib.AUTOMATIC) # use annual mean surface temperature from ITU-R P.1510 (T_Annual.TXT)
    sim.SetITURP452AirPressure(covlib.AUTOMATIC) # use mean annual global reference atmosphere pressure from ITU-R P.835
    sim.SetITURP452LandCoverMappingType(covlib.P452LandCoverMappingType.P452_MAP_TO_CLUTTER_CATEGORY)
    CLUT = covlib.P452ClutterCategory
    sim.SetITURP452RepresentativeClutterHeight(CLUT.P452_WATER_SEA, 0) # Values from TABLE 3, Section 3.2.1 of ITU-R P.452-18 recommendation
    sim.SetITURP452RepresentativeClutterHeight(CLUT.P452_OPEN_RURAL, 0)
    sim.SetITURP452RepresentativeClutterHeight(CLUT.P452_SUBURBAN, 10)
    sim.SetITURP452RepresentativeClutterHeight(CLUT.P452_URBAN_TREES_FOREST, 15)
    sim.SetITURP452RepresentativeClutterHeight(CLUT.P452_DENSE_URBAN, 20)

    # Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
    sim.SetITURP452RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    sim.SetTerrainElevDataSamplingResolution(25) # Note: this sampling resolution also applies to landcover/clutter data.

    # Set land cover data parameters
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER # data from https://esa-worldcover.org/
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    # Some default mappings between ESA WorldCover's land cover classes and ITU-R P.452-18's clutter categories (TABLE 3)
    # are already defined in crc-covlib, however these mappings may be modified when desired. For example:
    sim.ClearLandCoverClassMappings(WORLDCOVER, P452_18) # delete existing default mapping
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P452_18, CLUT.P452_URBAN_TREES_FOREST) # map 'Tree cover' (10) to 'urban/trees/forest' (4)
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P452_18, CLUT.P452_URBAN_TREES_FOREST) # map 'Built-up' (50) to 'urban/trees/forest' (4)
    sim.SetLandCoverClassMapping(WORLDCOVER, 80, P452_18, CLUT.P452_WATER_SEA) # map 'Permanent water bodies' (80) to 'water/sea' (1)
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P452_18, CLUT.P452_OPEN_RURAL) # map all other ESA WorldCover classes to 'open/rural' (2)
    # See https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf for a definition of ESA WorldCover classes.

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting first coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp452v18.bil'))

    # Alternately, we can directly map ESA WorldCover's land cover classes to representative clutter heights
    # and avoid using ITU-R P.452-18's clutter categories.
    sim.SetITURP452LandCoverMappingType(covlib.P452LandCoverMappingType.P452_MAP_TO_REPR_CLUTTER_HEIGHT)
    sim.ClearLandCoverClassMappings(WORLDCOVER, P452_18)
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P452_18, 15) # map 'Tree cover' (10) to a representative clutter height of 15m
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P452_18, 15) # map 'Built-up' (50) to a representative clutter height of 15m
    sim.SetLandCoverClassMapping(WORLDCOVER, 80, P452_18, 0) # map 'Permanent water bodies' to a representative clutter height of 0m
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P452_18, 0) # map all other ESA WorldCover classes to a representative clutter height of 0m

    print('Generating and exporting second coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp452v18alt.bil'))

    print('Simulations completed\n')
