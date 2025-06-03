# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


def printStatus(status):
    GEN = covlib.GenerateStatus
    if status == GEN.STATUS_OK:
        print('STATUS_OK')
    else:
        if (status & GEN.STATUS_SOME_TERRAIN_ELEV_DATA_MISSING) != 0:
            print('STATUS_SOME_TERRAIN_ELEV_DATA_MISSING')
        if (status & GEN.STATUS_NO_TERRAIN_ELEV_DATA) != 0:
            print('STATUS_NO_TERRAIN_ELEV_DATA')
        if (status & GEN.STATUS_SOME_LAND_COVER_DATA_MISSING) != 0:
            print('STATUS_SOME_LAND_COVER_DATA_MISSING')
        if (status & GEN.STATUS_NO_LAND_COVER_DATA) != 0:
            print('STATUS_NO_LAND_COVER_DATA')


if __name__ == '__main__':

    print('\ncrc-covlib - Secondary Terrain Elevation Source')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.533030, -75.496224)
    sim.SetTransmitterHeight(15)
    sim.SetTransmitterPower(0.2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.0)

    # Set propagation model parameters
    sim.SetLongleyRiceTimePercentage(50)
    sim.SetLongleyRiceLocationPercentage(50)
    sim.SetLongleyRiceSituationPercentage(50)

    # Set terrain elevation data parameters
    HRDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSourceSamplingMethod(HRDEM, covlib.SamplingMethod.BILINEAR_INTERPOLATION) # alternately, could use NEAREST_NEIGHBOR
    # One terrain elevation value every 10m in the terrain profiles that will be provided to Longley-Rice
    sim.SetTerrainElevDataSamplingResolution(10)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.5140, -75.5110, 45.5653, -75.44194)
    sim.SetReceptionAreaNumHorizontalPoints(100)
    sim.SetReceptionAreaNumVerticalPoints(100)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(45, 60, 0x5555FF)
    sim.AddCoverageDisplayFill(60, 75, 0x0000FF)
    sim.AddCoverageDisplayFill(75, 300, 0x000088)

    print('Generating coverage results using HRDEM only...')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'hrdem-only-coverage.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'hrdem-only-coverage.bil'))
    sim.ExportReceptionAreaTerrainElevationToBilFile(os.path.join(script_dir, 'hrdem-only-terrain.bil'), 1000, 1000, False)
    printStatus(sim.GetGenerateStatus()) # use GetGenerateStatus() to check if any terrain elevation data is missing

    print('Adding CDEM as secondary terrain elevation source')
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetSecondaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))

    print('Generating coverage results using HRDEM as primary source and CDEM as secondary source...')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'hrdem-and-cdem-coverage.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'hrdem-and-cdem-coverage.bil'))
    sim.ExportReceptionAreaTerrainElevationToBilFile(os.path.join(script_dir, 'hrdem-and-cdem-terrain.bil'), 1000, 1000, False)
    printStatus(sim.GetGenerateStatus())

    # Optionally, a third terrain elevation source could be added using sim->SetTertiaryElevationDataSource()

    print('Simulations completed\n')
