import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from pandas import read_csv
import matplotlib.pyplot as plt


if __name__ == '__main__':

    print('\ncrc-covlib - Exporting profiles to csv file, plotting profiles, using custom profiles')

    sim = covlib.Simulation()

    # set simulation parameters
    rx_lat = 45.51353
    rx_lon = -75.45785
    sim.SetPropagationModel(covlib.PropagationModel.ITU_R_P_1812)
    sim.SetITURP1812SurfaceProfileMethod(covlib.P1812SurfaceProfileMethod.P1812_USE_SURFACE_ELEV_DATA)
    sim.SetResultType(covlib.ResultType.PATH_LOSS_DB)
    sim.SetTransmitterLocation(45.557573,-75.506452)
    HRDEM_DTM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM_DTM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    HRDEM_DSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_HRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(HRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(HRDEM_DSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_HRDEM_DSM'))
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))
    sim.SetTerrainElevDataSamplingResolution(10)

    # calculate path loss and export profiles to .csv file
    csv_pathname = os.path.join(script_dir, 'profiles-p1812-surface.csv')
    sim.ExportProfilesToCsvFile(csv_pathname, rx_lat, rx_lon)

    # read some of the exported profiles into list objects,
    csv_data = read_csv(csv_pathname)
    path_length_km_profile = csv_data['path length (km)'].tolist()
    terr_profile = csv_data['terrain elevation (m)'].tolist()
    surf_profile = csv_data['surface elevation (m)'].tolist()
    itu_radio_clim_zone_profile = csv_data['P1812 radio climatic zone'].tolist()
    path_loss_profile = csv_data['path loss (dB)'].tolist()

    # plot some profiles
    fig, ax1 = plt.subplots()
    fig.set_size_inches(16,9)
    ax2 = ax1.twinx()
    # [2:] is to discard the first two profile points where pathloss is 0 (P1812 requires at least 3 profile points)
    ax1.plot(path_length_km_profile[2:], terr_profile[2:], color='#966414', label='terrain elevation (m)')
    ax1.plot(path_length_km_profile[2:], surf_profile[2:], 'g-', label='surface elevation (m)')
    ax2.plot(path_length_km_profile[2:], path_loss_profile[2:], 'b-')
    ax1.set_xlabel('path length (km)')
    ax1.set_ylabel('height (m)')
    ax2.set_ylabel('path loss (dB)', color='b')
    ax1.legend()
    plt.show()

    # As a quick test, compute path loss using GenerateProfileReceptionPointResult() with terrain profiles from the .csv file,
    # and then see if it matches the computed path loss values also from the .csv file (values may still be sligthly different
    # because of rounded decimals in the .csv file).
    path_loss = sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(terr_profile), terr_profile, None, surf_profile, itu_radio_clim_zone_profile)
    print('path loss from .csv file: {:.2f} dB'.format(path_loss_profile[-1]))
    print('path loss from GenerateProfileReceptionPointResult() using custom terrain elevation and surface elevation profiles: {:.2f} dB'.format(path_loss))

    print('Simulation completed\n')
