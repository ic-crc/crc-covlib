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
    sim.SetPropagationModel(covlib.PropagationModel.FREE_SPACE)
    sim.SetResultType(covlib.ResultType.PATH_LOSS_DB)
    sim.SetTransmitterLocation(45.557573,-75.506452)
    HRDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSamplingResolution(25)

    # calculate path loss and export profiles to .csv file
    csv_pathname = os.path.join(script_dir, 'profiles-FreeSpace.csv')
    sim.ExportProfilesToCsvFile(csv_pathname, rx_lat, rx_lon)
    
    # read some of the exported profiles into list objects
    csv_data = read_csv(csv_pathname)
    path_length_km_profile = csv_data['path length (km)'].tolist()
    terrain_elev_profile = csv_data['terrain elevation (m)'].tolist()
    path_loss_profile = csv_data['path loss (dB)'].tolist()

    # remove first items (at distance=0) before plotting
    path_length_km_profile.pop(0)
    terr_elev_0 = terrain_elev_profile.pop(0)
    path_loss_profile.pop(0)

    # plot some profiles
    fig, ax1 = plt.subplots()
    fig.set_size_inches(16,9)
    ax2 = ax1.twinx()
    ax1.plot(path_length_km_profile, terrain_elev_profile, color='#966414', label='terrain elevation (m)')
    ax2.plot(path_length_km_profile, path_loss_profile, 'b-')
    ax1.set_xlabel('path length (km)')
    ax1.set_ylabel('height (m)')
    ax2.set_ylabel('path loss (dB)', color='b')
    ax1.legend()
    plt.show()

    # As a quick test, compute path loss using GenerateProfileReceptionPointResult() with terrain profiles from the .csv file,
    # and then see if it matches the computed path loss values also from the .csv file (values may still be sligthly different
    # because of rounded decimals in the .csv file).
    terrain_elev_profile.insert(0, terr_elev_0) # restore to original profile
    path_loss = sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(terrain_elev_profile), terrain_elev_profile)
    print('path loss from .csv file: {:.2f} dB'.format(path_loss_profile[-1]))
    print('path loss from GenerateProfileReceptionPointResult() using custom terrain elevation profile: {:.2f} dB'.format(path_loss))

    print('Simulation completed\n')
