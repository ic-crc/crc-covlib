# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))

from crc_covlib.helper import streets
from crc_covlib.helper import itur_p1411
from crc_covlib.helper import itur_p530
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # This script calculates path loss according to equations from Section 4.1.3.2 of ITU-R
    # P.1411-12. Distances to and from the street corner are computed using OpenStreetMap data.

    print('\ncrc-covlib - ITU-R P.1411-12, site specific model for propagation within ' \
        'street canyons (NLoS) in the 2 to 38 GHz band')

    freq_GHz = 26.0
    tx_lat = 45.41853319
    tx_lon = -75.69908430
    tx_height_m = 5.0
    tx_street_width_m = 19.0
    rx_lat = 45.4212524
    rx_lon = -75.6972646
    rx_height_m = 1.5

    # Download OpenStreetMap data and get a street graph for the area surrounding the transmitter and
    # receiver locations.
    street_graph = streets.GetStreetGraphFromPoints(latLonPoints=[(tx_lat, tx_lon), (rx_lat, rx_lon)])

    # If desired, the street graph may be displayed.
    streets.DisplayStreetGraph(street_graph)

    # GetStreetCanyonsRadioPaths() computes one or more paths following street lines based on a
    # shortest travel distance algorithm. The shortest path is often but not always the path having
    # the lowest number of turns at street crossings, so we will ask for a few different paths to be
    # returned and then select one with the lowest amount of turns at street crossings (usually
    # preferable propagation-wise). 
    radio_paths = streets.GetStreetCanyonsRadioPaths(graph=street_graph, txLat=tx_lat, txLon=tx_lon,
                                                     rxLat=rx_lat, rxLon=rx_lon, numPaths=4)
    
    # Sort paths in ascending order of turns at street crossings and select the first one. We can
    # ask for the paths to be simplified before the sort takes place. This is ususally helpful as
    # it removes very small street sections that unnecessarily complexifies the path propagation-wise.
    selected_path = streets.SortRadioPathsByStreetCornerCount(radioPaths=radio_paths,
                                                              simplifyPaths=True)[0]
    
    # If desired, a path may be exported to a geojson file and visualized using a GIS application.
    streets.ExportRadioPathToGeojsonFile(radioPath=selected_path,
                                         pathname=os.path.join(script_dir, 'radio_path.geojson'))
    
    x_m_list = streets.DistancesToStreetCorners(radioPath=selected_path)
    alpha_deg_list = streets.StreetCornerAngles(radioPath=selected_path)

    if len(alpha_deg_list) != 1: # Check that the selected path only has one turn at street crossings.
        raise RuntimeError('Could not find path with one street corner turn.')

    total_travel_dist_m = sum(x_m_list)
    travel_distances_m = [min(i, total_travel_dist_m) for i in range(1, int(total_travel_dist_m+2))]
    path_losses_dB = []
    rain_attenuations_dB = []

    for travel_dist_m in travel_distances_m:
        if travel_dist_m < x_m_list[0]:
            x1_m = travel_dist_m
            x2_m = 0
        else:
            x1_m = x_m_list[0]
            x2_m = travel_dist_m - x1_m

        # Calculate path loss L including atmospheric gases attenuation but not rain attenuation.
        L = itur_p1411.SiteSpecificWithinStreetCanyonsSHFNonLoS(
            f_GHz=freq_GHz,
            x1_m=x1_m, x2_m=x2_m, w1_m=tx_street_width_m,
            h1_m=tx_height_m, h2_m=rx_height_m,
            hs_m=0.7, # Effective road height (m), depends on the traffic on the road (see ITU-R
                      # P.1411-12, TABLE 5 and 6). May not be used depending on frequency.
            n=2.1, # Basic transmission loss exponent (see ITU-R P.1411-12, TABLE 7).
            env=itur_p1411.EnvironmentC.URBAN,
            bldgLayout=itur_p1411.BuildingsLayout.WEDGE_SHAPED)
        path_losses_dB.append(L)

        # Caluculate rain attenuation exceeded for p% of the time.
        p = 1.0 # Time percentage (%) (yearly stats), valid values from 0.001 to 1.
        Ap = itur_p530.RainAttenuationLongTermStatistics(
            p=p,
            d_km=travel_dist_m,
            f_GHz=freq_GHz,
            pathElevAngle_deg=0, # May be approximated to 0 for most terrestrial paths.
            polTiltAngle_deg=90, # Polarization tilt angle relative to the horizontal (90=vertical pol).
            lat=tx_lat, lon=tx_lon) # Location for getting rainfall rate stats, may use either tx or rx location for short paths.
        rain_attenuations_dB.append(Ap)

    # Plot results
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    max_dist_m = int(travel_distances_m[-1])
    ax1.set_xlim([0, max_dist_m])
    ax1.set_xticks([*range(0, max_dist_m+1, 50)])
    ax1.set_ylim([160, 60])
    ax1.set_yticks([*range(60, 160+1, 20)])
    ax1.plot(travel_distances_m, path_losses_dB, color='#008DD2', label='{:.1f} GHz'.format(freq_GHz))
    ax1.plot(travel_distances_m, [sum(x) for x in zip(path_losses_dB, rain_attenuations_dB)],
             color='#E31E24', label='{:.1f} GHz with rain attenuation\nexceeded for {}% of the time'.format(freq_GHz, p))
    ax1.set_title('ITU-R P.1411-12, site specific model for propagation within street canyons (NLoS)')
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Path loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()
