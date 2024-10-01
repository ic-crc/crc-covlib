import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from crc_covlib.helper import topography


if __name__ == '__main__':

    print('\ncrc-covlib - helper.topography.LoadFileAsCustomSurfaceElevData()')

    sim = covlib.Simulation()

    rx_area = (45.37914, -75.75000, 45.47148, -75.61225)

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.0)

    # Propagation model selection
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)

   # Set ITU-R P.1812 propagation model parameters
    sim.SetITURP1812TimePercentage(50)
    sim.SetITURP1812LocationPercentage(50)
    sim.SetITURP1812SurfaceProfileMethod(covlib.P1812SurfaceProfileMethod.P1812_USE_SURFACE_ELEV_DATA) # using surface elevation rather than clutter data
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    sim.SetTerrainElevDataSamplingResolution(25)

    # Here we will use a CDSM data file as the source for surface elevation data in our simulation.
    # The format and CRS (coordinate reference system) of this file are supported by crc-covlib and
    # it may be read using the SURF_ELEV_NRCAN_CDSM data source type. However, as a demonstration we will
    # use the SURF_ELEV_CUSTOM data source type here and read the file using the
    # crc_covlib.helper.topography.LoadFileAsCustomSurfaceElevData() function. This function uses third-party
    # libraries rasterio (GDAL) and numpy in order to support a large variety of raster file formats
    # and CRS. The surface elevation data is reprojected (if needed) and sent to the simulation object in 
    # crc-covlib's custom data format. The technique may be used to use elevation and/or land cover data
    # files that are not supported by crc-covlib's core functionalities (C++ implementation).
    #
    # The LoadFileAsCustomSurfaceElevData() function takes an optional third parameter (bounds) to specify
    # what area from the file is to be used. Otherwise the whole uncompressed content of the file is
    # read and copied into the simulation object, and this may fail or take a great amount of time for
    # large files.
    sim.SetPrimarySurfaceElevDataSource(covlib.SurfaceElevDataSource.SURF_ELEV_CUSTOM)
    cdi: topography.CustomDataInfo
    cdi = topography.LoadFileAsCustomSurfaceElevData(
        sim,
        os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_CDSM/031G05_cdsm_final_e.tif'),
        bounds=topography.LatLonBox(*rx_area))
    if cdi is not None:
        print('Loaded {}x{} surface elevation data points'.format(cdi.numHorizSamples, cdi.numVertSamples))
    else:
        print('Warning: failed to load surface elevation data!')

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
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'topo-helper-surface-elev.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'topo-helper-surface-elev.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'topo-helper-surface-elev.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'topo-helper-surface-elev.bil'))

    if( sim.GetGenerateStatus() != covlib.GenerateStatus.STATUS_OK ):
        print('Warning: some data missing!')

    print('Simulation completed\n')
