import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - NRCAN CDEM')

    sim = covlib.Simulation()

    #Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.0)

    # Set propagation model parameters
    sim.SetLongleyRiceTimePercentage(50)
    sim.SetLongleyRiceLocationPercentage(50)
    sim.SetLongleyRiceSituationPercentage(50)

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    # One terrain elevation value every 25m in the terrain profiles that will be provided to Longley-Rice
    sim.SetTerrainElevDataSamplingResolution(25)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(45, 60, covlib.RGBtoInt(85, 85, 255))
    sim.AddCoverageDisplayFill(60, 75, covlib.RGBtoInt(0, 0, 255))
    sim.AddCoverageDisplayFill(75, 300, covlib.RGBtoInt(0, 0, 136))

    print('Generating and exporting coverage results (high resolution)...')

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-cdem-hires.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'terrain-elev-cdem-hires.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'terrain-elev-cdem-hires.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-cdem-hires.bil'))

    # Run same simulation again but at a lower resolution (much faster)
    print('Generating and exporting coverage results (low resolution)...')
    sim.SetTerrainElevDataSamplingResolution(200) # one value every 200m for terrain elevation profiles
    sim.SetReceptionAreaNumHorizontalPoints(60) # 3600 reception points (60x60)
    sim.SetReceptionAreaNumVerticalPoints(60)
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-cdem-lowres.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'terrain-elev-cdem-lowres.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'terrain-elev-cdem-lowres.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-cdem-lowres.bil'))

    print('Simulations completed\n')
