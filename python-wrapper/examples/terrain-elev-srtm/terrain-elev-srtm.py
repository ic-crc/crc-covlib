import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - SRTM')

    sim = covlib.Simulation()

    # Set transmitter parameters
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
    # NOTE: SRTM data samples contain surface elevation data (terrain elevation + clutter height with no 
    #       distinction between the two). Therefore the calculated field strength should be generally lesser
    #       than when using true terrain elevation (ground level) data.
    SRTM = covlib.TerrainElevDataSource.TERR_ELEV_SRTM
    sim.SetPrimaryTerrainElevDataSource(SRTM)
    sim.SetTerrainElevDataSourceDirectory(SRTM, os.path.join(script_dir, '../../../data/surface-elev-samples/SRTMGL1'))
	# One terrain elevation value every 50m in the terrain profiles that will be provided to Longley-Rice
    sim.SetTerrainElevDataSamplingResolution(50)

	# Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(140)
    sim.SetReceptionAreaNumVerticalPoints(140)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(45, 60, covlib.RGBtoInt(85, 85, 255))
    sim.AddCoverageDisplayFill(60, 75, covlib.RGBtoInt(0, 0, 255))
    sim.AddCoverageDisplayFill(75, 300, covlib.RGBtoInt(0, 0, 136))

    print('Generating and exporting coverage results...')

    sim.GenerateReceptionAreaResults()

    # Export results in various formats
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-srtm.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'terrain-elev-srtm.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'terrain-elev-srtm.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-srtm.bil'))

    print('Simulation completed\n')