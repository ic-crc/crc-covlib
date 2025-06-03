# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))

from crc_covlib.helper import topography
from crc_covlib.helper import itur_p1411
from crc_covlib.helper import itur_p530
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Calculate path loss according to Section 4.1.2 of ITU-R P.1411-12.

    print('\ncrc-covlib - ITU-R P.1411-12, site specific models for propagation within street canyons (LoS)')

    tx_lat = 45.416904
    tx_lon = -75.705472
    tx_height_m = 4.0
    rx_lat = 45.421069
    rx_lon = -75.695849
    rx_height_m = 1.5
    frequencies_GHz = [1.5, 9.0, 28.0]
    path_losses_dB = [[],[],[]]  # one list of path losses over distance for each frequency
    rain_att_dB = [] # rain attenuation over distance

    sampling_res_m = 1.0 # sampling resolution, in meters
    dist_profile_m = topography.GetDistanceProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res_m) * 1000.0
    dist_profile_m = dist_profile_m[1:] # remove first element (i.e. distance of 0m) 

    for dist_m in dist_profile_m:
        for fi, f_GHz in enumerate(frequencies_GHz):
            if f_GHz <= 3.0:
                L, _, _ = itur_p1411.SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz=f_GHz, d_m=dist_m, h1_m=tx_height_m, h2_m=rx_height_m)
            elif f_GHz <= 10.0:
                hs_m = 0.7 # effective road height (m), depends on the traffic on the road (see ITU-R P.1411-12, TABLE 5 and 6)
                L, _, _ = itur_p1411.SiteSpecificWithinStreetCanyonsSHFLoS(f_GHz=f_GHz, d_m=dist_m, h1_m=tx_height_m, h2_m=rx_height_m, hs_m=hs_m)
            else:
                n = 2.1 # basic transmission loss exponent (see ITU-R P.1411-12, TABLE 7)
                L = itur_p1411.SiteSpecificWithinStreetCanyonsEHFLoS(f_GHz=f_GHz, d_m=dist_m, n=n) # includes atmospheric gases attenuation but 
                                                                                                   # not rain attenuation

                # Caluculate rain attenuation exceeded for p% of the time.
                p = 1.0 # time percentage (%) (yearly stats), valid values from 0.001 to 1
                Ap = itur_p530.RainAttenuationLongTermStatistics(
                    p=p,
                    d_km=dist_m,
                    f_GHz=f_GHz,
                    pathElevAngle_deg=0, # may be approximated to 0 for most terrestrial paths
                    polTiltAngle_deg=90, # polarization tilt angle relative to the horizontal (90=vertical pol)
                    lat=tx_lat, lon=tx_lon) # location for getting rainfall rate stats, may use either tx or rx location for short paths
                rain_att_dB.append(Ap)

            path_losses_dB[fi].append(L)

    
    # Plot results
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    max_dist_m = int(dist_profile_m[-1])
    ax1.set_xlim([0, max_dist_m])
    ax1.set_xticks([*range(0, max_dist_m+1, 50)])
    ax1.set_ylim([140, 40])
    ax1.set_yticks([*range(40, 140+1, 20)])
    ax1.plot(dist_profile_m, path_losses_dB[0], color='#008DD2', label='{:.1f} GHz'.format(frequencies_GHz[0]))
    ax1.plot(dist_profile_m, path_losses_dB[1], color='#E31E24', label='{:.1f} GHz'.format(frequencies_GHz[1]))
    ax1.plot(dist_profile_m, path_losses_dB[2], color='#1D8545', label='{:.1f} GHz'.format(frequencies_GHz[2]))
    ax1.plot(dist_profile_m, [sum(x) for x in zip(path_losses_dB[2], rain_att_dB)],
             color='#B73B7B', label='{:.1f} GHz with rain attenuation\nexceeded for {}% of the time'.format(frequencies_GHz[2], p))
    ax1.set_title('ITU-R P.1411-12, site specific models for propagation within street canyons (LoS)')
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Path loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()
