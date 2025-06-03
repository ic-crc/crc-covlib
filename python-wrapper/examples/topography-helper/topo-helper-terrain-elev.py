# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from crc_covlib.helper import topography


if __name__ == '__main__':

    print('\ncrc-covlib - helper.topography.LoadFileAsCustomTerrainElevData()')

    sim = covlib.Simulation()

    rx_area = (45.37914, -75.81922, 45.47148, -75.61225)

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

    # Here we will use a CDEM data file as the source for terrain elevation data in our simulation.
    # The format and CRS (coordinate reference system) of this file are supported by crc-covlib and
    # it may be read using the TERR_ELEV_NRCAN_CDEM data source type. However, as a demonstration we will
    # use the TERR_ELEV_CUSTOM data source type here and read the file using the
    # crc_covlib.helper.topography.LoadFileAsCustomTerrainElevData() function. This function uses third-party
    # libraries rasterio (GDAL) and numpy in order to support a large variety of raster file formats
    # and CRS. The terrain elevation data is reprojected (if needed) and sent to the simulation object in 
    # crc-covlib's custom data format. The technique may be used to use elevation and/or land cover data
    # files that are not supported by crc-covlib's core functionalities (C++ implementation).
    #
    # The LoadFileAsCustomTerrainElevData() function takes an optional third parameter (bounds) to specify
    # what area from the file is to be used. Otherwise the whole uncompressed content of the file is
    # read and copied into the simulation object, and this may fail or take a great amount of time for
    # large files.
    sim.SetPrimaryTerrainElevDataSource(covlib.TerrainElevDataSource.TERR_ELEV_CUSTOM)
    sim.SetTerrainElevDataSamplingResolution(25)
    cdi: topography.CustomDataInfo
    cdi = topography.LoadFileAsCustomTerrainElevData(
        sim,
        os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM/cdem_dem_031G.tif'),
        bounds=topography.LatLonBox(*rx_area))
    if cdi is not None:
        print('Loaded {}x{} elevation data points'.format(cdi.numHorizSamples, cdi.numVertSamples))
    else:
        print('Warning: failed to load elevation data!')

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(*rx_area)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(45, 60, 0x5555FF)
    sim.AddCoverageDisplayFill(60, 75, 0x0000FF)
    sim.AddCoverageDisplayFill(75, 300, 0x000088)

    print('Generating and exporting coverage results...')

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'topo-helper-terrain-elev.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'topo-helper-terrain-elev.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'topo-helper-terrain-elev.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'topo-helper-terrain-elev.bil'))

    if( sim.GetGenerateStatus() != covlib.GenerateStatus.STATUS_OK ):
        print('Warning: some data missing!')

    print('Simulation completed\n')
