# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))
from crc_covlib import simulation as covlib


def printStatus(status):
    GEN = covlib.GenerateStatus
    if status == GEN.STATUS_OK:
        print('STATUS_OK')
    else:
        print('WARNING:')
        if (status & GEN.STATUS_SOME_TERRAIN_ELEV_DATA_MISSING) != 0:
            print('STATUS_SOME_TERRAIN_ELEV_DATA_MISSING')
        if (status & GEN.STATUS_NO_TERRAIN_ELEV_DATA) != 0:
            print('STATUS_NO_TERRAIN_ELEV_DATA')


if __name__ == '__main__':

    print('\ncrc-covlib - Geotiff terrain elevation files')

    sim = covlib.Simulation()

    #Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.0)

    # Set propagation model parameters
    sim.SetLongleyRiceTimePercentage(50)
    sim.SetLongleyRiceLocationPercentage(50)
    sim.SetLongleyRiceSituationPercentage(50)

    # crc-covlib can use geotiff files as data source, however not every coordinate reference
    # system (CRS) is supported. The GetGenerateStatus() method (see below) can be used to easily
    # verify whether terrain data could be properly read when generating the simulation's results.
    #
    # Supported CRS are: "WGS 84" (EPSG:4326), "NAD83" (EPSG:4269), "NAD83(CSRS)" (EPSG:4617),
    # "NAD83(CSRS98)" (EPSG:4140), "WGS 84 / UTM zone [num]", "NAD83 / UTM zone [num]",
    # "NAD83(CSRS) / UTM zone [num]" and "NAD83(CSRS) / Canada Atlas Lambert" (EPSG:3979).
    #
    # For geotiff files that do not use a supported CRS or for raster files not in the geotiff
    # format, one possible option is to use the LoadFileAsCustomTerrainElevData() function from the
    # crc_covlib.helper.topography module (see the topo-helper-terrain-elev.py example).
    # LoadFileAsCustomTerrainElevData() may be called repeatedly to load more than one file, as
    # long as there is sufficient RAM available.
    GEOTIFF = covlib.TerrainElevDataSource.TERR_ELEV_GEOTIFF
    sim.SetPrimaryTerrainElevDataSource(GEOTIFF)
    dir = os.path.join(script_dir, '../../../../data/terrain-elev-samples/NRCAN_CDEM')
    sim.SetTerrainElevDataSourceDirectory(GEOTIFF, dir, useIndexFile=False)
    sim.SetTerrainElevDataSamplingResolution(25)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.PATH_LOSS_DB)

    print('Generating and exporting coverage results...')

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-geotiff.txt'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-geotiff.bil'))

    printStatus(sim.GetGenerateStatus())

    print('Simulation completed\n')
