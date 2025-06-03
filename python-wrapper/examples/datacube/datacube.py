# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from crc_covlib.helper import datacube_canada


if __name__ == '__main__':

    print('\ncrc-covlib - Download of terrain/surface elevation data for ITU-R P.1812 simulation')

    # Reception area boundaries
    rx_min_lat = 45.37914
    rx_min_lon = -75.81922
    rx_max_lat = 45.47148
    rx_max_lon = -75.61225

    # - Download Medium Resolution Digital Elevation Model (MRDEM) data at 30-meter resolution.
    # - Functions for High Resolution Digital Elevation Model (HRDEM) at 1 or 2-meter resolution
    #   are also available from the datacube_canada module.
    # - Terrain and surface data files should be kept in separate directories for them to be
    #   properly handled by crc-covlib.
    bounds = datacube_canada.LatLonBox(rx_min_lat, rx_min_lon, rx_max_lat, rx_max_lon)
    terrain_elev_file = os.path.join(script_dir, 'terrain/mrdem_dtm.tif') # output file for MRDEM extract
    print('Downloading terrain elevation data...', end='')
    datacube_canada.DownloadMrdemDtm(bounds, terrain_elev_file)
    print('completed')
    surface_elev_file = os.path.join(script_dir, 'surface/mrdem_dsm.tif') # output file for MRDEM extract
    print('Downloading surface elevation data...', end='')
    datacube_canada.DownloadMrdemDsm(bounds, surface_elev_file)
    print('completed')


    # Proceed with the simulation

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterFrequency(2600)

    # Propagation model selection
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)

    # Using surface elevation rather than clutter data
    sim.SetITURP1812SurfaceProfileMethod(covlib.P1812SurfaceProfileMethod.P1812_USE_SURFACE_ELEV_DATA)

    # Set terrain elevation data parameters
    MRDEM_DTM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_MRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(MRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(MRDEM_DTM, os.path.join(script_dir, 'terrain'))
    sim.SetTerrainElevDataSamplingResolution(30)

    # Set surface elevation data parameters
    MRDEM_DSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_MRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(MRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(MRDEM_DSM, os.path.join(script_dir, 'surface'))

    # Set reception area parameters
    sim.SetReceptionAreaCorners(rx_min_lat, rx_min_lon, rx_max_lat, rx_max_lon)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.PATH_LOSS_DB)

    print('Generating and exporting path loss results...', end='')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'sim.bil'))
    print('completed\n')

    # Optionally delete the terrain and surface data files
    #os.remove(terrain_elev_file)
    #os.remove(surface_elev_file)
