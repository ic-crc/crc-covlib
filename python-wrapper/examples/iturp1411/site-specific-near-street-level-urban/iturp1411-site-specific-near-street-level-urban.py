import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))

from crc_covlib.helper import streets
from crc_covlib.helper import itur_p1411
from math import log10


def main():
    # Calculate path loss according to Section 4.3.2 of ITU-R P.1411-12.

    print('\ncrc-covlib - ITU-R P.1411-12, site specific model for propagation between terminals ' \
          'located from below roof-top height to near street level in an urban environment\n')

    freq_GHz = 3.5
    tx_height_m = 4.0
    rx_height_m = 1.5
    tx_lat = 45.4189209
    tx_lon = -75.6994100
    hs_m = 0.75 # Effective road height (m), see ITU-R P.1411-12 TABLE 5 and TABLE 6
    rx_locations = [(45.4202813,-75.7005318), (45.4213236,-75.6971346), (45.4205041,-75.6980416)]

    # Download OpenStreetMap data and get a street graph for the area surrounding the transmitter and
    # receiver locations.
    street_graph = streets.GetStreetGraphFromPoints(rx_locations + [(tx_lat, tx_lon)])

    # If desired, the street graph may be displayed.
    streets.DisplayStreetGraph(street_graph)

    for rx_location in rx_locations:
        rx_lat = rx_location[0]
        rx_lon = rx_location[1]

        # GetStreetCanyonsRadioPaths() computes one or more paths following street lines based on a
        # shortest travel distance algorithm. The shortest path is often but not always the path
        # having the lowest number of turns at street crossings, so we will ask for a few different
        # paths (up to 4) to be returned and then select the one(s) with the lowest amount of turns
        # at street crossings (usually preferable propagation-wise). 
        radio_paths = streets.GetStreetCanyonsRadioPaths(graph=street_graph, txLat=tx_lat, txLon=tx_lon,
                                                         rxLat=rx_lat, rxLon=rx_lon, numPaths=4)
        
        # Sort paths in ascending order of turns at street crossings. We can ask for the paths to be
        # simplified before the sort takes place. This is ususally helpful as it removes very small
        # street sections that unnecessarily complexifies the path propagation-wise.
        radio_paths = streets.SortRadioPathsByStreetCornerCount(radioPaths=radio_paths,
                                                                simplifyPaths=True)
        
        # Get the recommendation's x1, x2 (if applicable) and x3 (if applicable) values using the first
        # path in the sorted list of paths.
        x_list = streets.DistancesToStreetCorners(radioPath=radio_paths[0])

        if len(x_list) == 1:
            print('LoS situation')
            L_dB = itur_p1411.SiteSpecificNearStreetLevelUrban(f_GHz=freq_GHz, x1_m=x_list[0], x2_m=0, x3_m=0,
                                                               h1_m=tx_height_m, h2_m=rx_height_m, hs_m=hs_m)
            filename = os.path.join(script_dir, 'LoS_radio_path.geojson')
            streets.ExportRadioPathToGeojsonFile(radio_paths[0], filename)
            print('  path loss = {:.1f} dB\n'.format(L_dB))

        elif len(x_list) == 2:
            print('1-Turn NLoS situation')
            L_dB = itur_p1411.SiteSpecificNearStreetLevelUrban(f_GHz=freq_GHz, x1_m=x_list[0], x2_m=x_list[1],
                                                               x3_m=0, h1_m=tx_height_m, h2_m=rx_height_m, hs_m=hs_m)
            filename = os.path.join(script_dir, '1-turn_NLoS_radio_path.geojson')
            streets.ExportRadioPathToGeojsonFile(radio_paths[0], filename)
            print('  path loss = {:.1f} dB\n'.format(L_dB))

        elif len(x_list) == 3:
            print('2-Turn NLoS situation')

            # Compute the path loss for different route paths and get the overall loss using
            # equation (68) from ITU-R P.1411-12.
            i = 0
            summing = 0
            while len(x_list) == 3:
                Li_dB = itur_p1411.SiteSpecificNearStreetLevelUrban(f_GHz=freq_GHz, x1_m=x_list[0], x2_m=x_list[1],
                                                                    x3_m=x_list[2], h1_m=tx_height_m, h2_m=rx_height_m,
                                                                    hs_m=hs_m)
                summing += 1.0/pow(10.0, Li_dB/10.0)
                filename = os.path.join(script_dir, '2-turn_NLoS_radio_path_{}.geojson'.format(i))
                streets.ExportRadioPathToGeojsonFile(radio_paths[i], filename)
                print('  radio path [{}]: travel distance = {:.1f} m, path loss = {:.1f} dB'.format(i, sum(x_list), Li_dB))
                i += 1
                if len(radio_paths) > i:
                    x_list = streets.DistancesToStreetCorners(radioPath=radio_paths[i])
                else:
                    break
            L_dB = -10.0*log10(summing)
            print('  overall path loss = {:.1f} dB\n'.format(L_dB))

        else:
            print('No path with 2 turns or less found.\n')


if __name__ == '__main__':
    main()
