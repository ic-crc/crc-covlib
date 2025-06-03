# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

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
    sim.SetPropagationModel(covlib.PropagationModel.ITU_R_P_452_V18)
    sim.SetResultType(covlib.ResultType.PATH_LOSS_DB)
    sim.SetTransmitterLocation(45.557573,-75.506452)
    HRDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))
    sim.SetITURP452RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))
    sim.SetTerrainElevDataSamplingResolution(25)

    # calculate path loss and export profiles to .csv file
    csv_pathname = os.path.join(script_dir, 'profiles-p452v18.csv')
    sim.ExportProfilesToCsvFile(csv_pathname, rx_lat, rx_lon)

    # read some of the exported profiles into list objects,
    csv_data = read_csv(csv_pathname)
    path_length_km_profile = csv_data['path length (km)'].tolist()
    terr_elev_profile = csv_data['terrain elevation (m)'].tolist()
    clutter_category_profile = csv_data['P452-18 clutter category'].tolist()
    repr_clutter_height_profile = csv_data['P452-18 representative clutter height (m)'].tolist()
    itu_radio_clim_zone_profile = csv_data['P452-18 radio climatic zone'].tolist()
    path_loss_profile = csv_data['path loss (dB)'].tolist()

    # get a 'terrain elevation + representative clutter height' profile
    terr_plus_repr_height_profile = []
    for i in range(len(terr_elev_profile)):
        terr_plus_repr_height_profile.append(terr_elev_profile[i] + repr_clutter_height_profile[i])

    # plot some profiles
    fig, ax1 = plt.subplots()
    fig.set_size_inches(16,9)
    ax2 = ax1.twinx()
    # [2:] is to discard the first two profile points where pathloss is 0 (P452 requires at least 3 profile points)
    ax1.plot(path_length_km_profile[2:], terr_elev_profile[2:], color='#966414', label='terrain elevation (m)')
    ax1.plot(path_length_km_profile[2:], terr_plus_repr_height_profile[2:], 'g-', label='terrain elevation + representative clutter height (m)')
    ax2.plot(path_length_km_profile[2:], path_loss_profile[2:], 'b-')
    ax1.set_xlabel('path length (km)')
    ax1.set_ylabel('height (m)')
    ax2.set_ylabel('path loss (dB)', color='b')
    ax1.legend()
    plt.show()

    # As a quick test, compute path loss using GenerateProfileReceptionPointResult() with terrain profiles from the .csv file,
    # and then see if it matches the computed path loss values also from the .csv file (values may still be sligthly different
    # because of rounded decimals in the .csv file).
    path_loss = sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(terr_elev_profile), terr_elev_profile, clutter_category_profile, itu_radio_clim_zone_profile)
    print('path loss from .csv file: {:.2f} dB'.format(path_loss_profile[-1]))
    print('path loss from GenerateProfileReceptionPointResult() using custom terrain elevation and clutter category profiles: {:.2f} dB'.format(path_loss))
    # alternately, could use the representative clutter height profile...
    sim.SetITURP452LandCoverMappingType(covlib.P452LandCoverMappingType.P452_MAP_TO_REPR_CLUTTER_HEIGHT)
    path_loss = sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(terr_elev_profile), terr_elev_profile, [int(i) for i in repr_clutter_height_profile], itu_radio_clim_zone_profile)
    print('path loss from GenerateProfileReceptionPointResult() using custom terrain elevation and representative clutter height profiles: {:.2f} dB'.format(path_loss))

    print('Simulation completed\n')
