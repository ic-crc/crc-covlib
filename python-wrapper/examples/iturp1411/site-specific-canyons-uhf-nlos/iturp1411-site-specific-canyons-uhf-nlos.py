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
from math import radians


if __name__ == '__main__':
    # This script calculates path loss for a single reception point according to equations from
    # Section 4.1.3.1 of ITU-R P.1411-12 (see also FIGURE 3 from Section 3.1.2). Distances to and
    # from the street corner and the street corner angle are computed using OpenStreetMap data.

    print('\ncrc-covlib - ITU-R P.1411-12, site specific model for propagation within ' \
          'street canyons (NLoS) in the 0.8 to 2.0 GHz band')
    
    freq_GHz = 1.5
    tx_lat = 43.659072
    tx_lon = -79.382108 
    rx_lat = 43.6613634
    rx_lon = -79.3818735
    tx_street_width_m = 27.0
    rx_street_width_m = 27.0

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

    distances_m = streets.DistancesToStreetCorners(radioPath=selected_path)
    angles_deg = streets.StreetCornerAngles(radioPath=selected_path)

    if len(angles_deg) == 1: # Check that the selected path only has one turn at street crossings.

        # Compute path loss
        L = itur_p1411.SiteSpecificWithinStreetCanyonsUHFNonLoS(f_GHz=freq_GHz,
                                                                x1_m=distances_m[0], x2_m=distances_m[1],
                                                                w1_m=tx_street_width_m, w2_m=rx_street_width_m,
                                                                alpha_rad=radians(angles_deg[0]))
 
        print('Distance from station 1 to street crossing, x1 (m): {:.1f}'.format(distances_m[0]))
        print('Distance from street crossing to station 2, x2 (m): {:.1f}'.format(distances_m[1]))
        print('Total travel distance (m): {:.1f}'.format(sum(distances_m)))
        print('Street corner angle, alpha (rad): {:.2f}  ({:.2f} deg)'.format(radians(angles_deg[0]), angles_deg[0]))
        print('Path loss (dB): {:.1f}\n'.format(L))
