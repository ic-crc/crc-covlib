import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))
from crc_covlib import simulation as covlib
from array import array


if __name__ == '__main__':

    print('\ncrc-covlib - Custom terrain elevation data file')

    sim = covlib.Simulation()

    # Preparing the custom terrain elevation data to provide to crc-covlib
    sizeX = 5448
    sizeY = 2733
    terrainElevData = array('f')
    with open(os.path.join(script_dir, '../../../../data/terrain-elev-samples/custom/cdem_ottawa_075asecs.float'), 'rb') as file:
        terrainElevData.fromfile(file, sizeX*sizeY)

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
    sim.SetPrimaryTerrainElevDataSource(covlib.TerrainElevDataSource.TERR_ELEV_CUSTOM)
    sim.AddCustomTerrainElevData(44.95875000, -76.37583333, 45.52791667, -75.24104167, sizeX, sizeY, terrainElevData)
    # terrainElevData can be discarded to free up memory, the sim object keeps a copy of it
    del terrainElevData[:]
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

    print('Generating and exporting coverage results...')

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'terrain-elev-custom.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'terrain-elev-custom.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'terrain-elev-custom.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'terrain-elev-custom.bil'))

    print('Simulation completed\n')
