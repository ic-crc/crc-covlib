import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - NRCAN HRDEM')

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
    dir = os.path.join(script_dir, '../../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM')
    sim.SetTerrainElevDataSourceDirectory(HRDEM, dir, useIndexFile=False)
    sim.SetTerrainElevDataSourceSamplingMethod(HRDEM, covlib.SamplingMethod.BILINEAR_INTERPOLATION) # alternately, could use NEAREST_NEIGHBOR
    # One terrain elevation value every 10m in the terrain profiles that will be provided to Longley-Rice
    sim.SetTerrainElevDataSamplingResolution(10)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.5140, -75.5110, 45.5477, -75.4619)
    sim.SetReceptionAreaNumHorizontalPoints(100)
    sim.SetReceptionAreaNumVerticalPoints(100)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(45, 60, covlib.RGBtoInt(85, 85, 255))
    sim.AddCoverageDisplayFill(60, 75, covlib.RGBtoInt(0, 0, 255))
    sim.AddCoverageDisplayFill(75, 300, covlib.RGBtoInt(0, 0, 136))

    print('Generating and exporting coverage results...')

    sim.GenerateReceptionAreaResults()

    # Export results in various formats
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-hrdem.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'terrain-elev-hrdem.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'terrain-elev-hrdem.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-hrdem.bil'))

    print('Simulation completed\n')

