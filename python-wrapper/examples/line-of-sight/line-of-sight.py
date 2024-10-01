import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from crc_covlib.helper import itur_p2001
from crc_covlib.helper import topography


if __name__ == '__main__':

    print('\ncrc-covlib - Calculating whether paths are line-of-sight or non-line-of-sight')

    sim = covlib.Simulation()

    # Set terrain elevation data source
    HRDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))

    # Set surface elevation data source
    HRDEM_DSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_HRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(HRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(HRDEM_DSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_HRDEM_DSM'))

    tx_lat = 45.53012
    tx_lon = -75.486773
    tx_height = 15 # transmitter's radiation center height above ground level (meters)
    rx_height = 1.5 # receiver's radiation center height above ground level (meters)
    AREA_NUM_PTS_X = 140
    AREA_NUM_PTS_Y = 100
    sampling_res = 1.0 # terrain/surface elevation data sampling resolution (meters) for profiles

    sim.SetReceptionAreaCorners(45.529165, -75.488674, 45.531045, -75.485219)
    sim.SetReceptionAreaNumHorizontalPoints(AREA_NUM_PTS_X)
    sim.SetReceptionAreaNumVerticalPoints(AREA_NUM_PTS_Y)

    # We will use the sim object to read the terrain and surface elevation data files but we will
    # fill the reception area's results matrix ourselves by using the IsLOS function from the 
    # itur_p2001 module.
    for x in range(0, AREA_NUM_PTS_X):
        for y in range(0, AREA_NUM_PTS_Y):
            rx_lat = sim.GetReceptionAreaResultLatitude(x, y)
            rx_lon = sim.GetReceptionAreaResultLongitude(x, y)

            tx_terr_elev = sim.GetTerrainElevation(tx_lat, tx_lon)
            rx_terr_elev = sim.GetTerrainElevation(rx_lat, rx_lon)
            
            lat_lon_profile = topography.GetLatLonProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res)
            dist_profile = topography.GetDistanceProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res)

            # Alternately, we could use a terrain elevation profile (instead of surface) to only take
            # ground level into account.
            surf_elev_profile, status = topography.GetSurfaceElevProfile(sim, lat_lon_profile)

            is_los = itur_p2001.IsLOS(tx_lat, tx_lon, tx_height+tx_terr_elev,
                                      rx_lat, rx_lon, rx_height+rx_terr_elev,
                                      dist_profile, surf_elev_profile)
            if is_los == True:
                sim.SetReceptionAreaResultValue(x, y, 1)
            else:
                sim.SetReceptionAreaResultValue(x, y, 0)

    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'los.bil'))

    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(0.5, 1.5, 0x00FFF0) # fill contour for points that are in line-of-sight
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'los.kml'))

    print('Simulation completed\n')