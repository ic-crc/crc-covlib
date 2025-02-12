import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))

from crc_covlib.helper import buildings
from crc_covlib.helper import topography
from crc_covlib.helper import itur_p1411
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')
warnings.filterwarnings("ignore", category=FutureWarning, module='crc_covlib.helper.buildings')


def main():
    # Calculate path loss according to Section 4.2.2.1 of ITU-R P.1411-12.

    print('\ncrc-covlib - ITU-R P.1411-12, site specific model for propagation over rooftops in an urban area')

    freq_GHz = 10.0
    tx_height_m = 55.0
    rx_height_m = 2.0
    tx_lat = 45.4314310
    tx_lon = -75.6635237
    rx_lat = 45.4288756
    rx_lon = -75.6627794

    # Download building footprints from OpenStreetMap. Building heights are estimated from the
    # number of floors ('building:levels' tag) when present. Otherwise buildings are assigned
    # the specified default height in meters. This building data will be used to calculate input
    # parameters for the propagation model.
    bounds = buildings.LatLonBox(minLat=min(tx_lat, rx_lat), minLon=min(tx_lon, rx_lon), 
                                 maxLat=max(tx_lat, rx_lat), maxLon=max(tx_lon, rx_lon))
    bldg_list = buildings.GetBuildingsFromOpenStreetMap(bounds, defaultHeight_m=3.0, avgFloorHeight_m=3.0)
    
    # Alternately, building footprints and heights may be obtained from a shapefile (.shp).
    # For example:
    # bldg_list = buildings.GetBuildingsFromShapefile(pathname='C:/exemple.shp',
    #                                                 heightAttribute='to specify',
    #                                                 defaultHeight_m=3.0)

    # Display the building heights profile on the direct path from the transmitter to the receiver.
    sampling_res_m = 1.0
    lat_lon_profile = topography.GetLatLonProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res_m)
    dist_profile_m = topography.GetDistanceProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res_m)*1000.0
    bldg_heights_profile_m, bldg_list = buildings.GetBuildingHeightsProfile(bldg_list, lat_lon_profile)
    DisplayBuildingHeightsProfile(dist_profile_m, bldg_heights_profile_m, 'Building heights profile')

    # In order to get correct metrics for the multi-screen diffraction model, we remove the first and
    # third buildings (i.e. the buildings that are not part of the multi-screen).
    bldg_list.pop(2)
    bldg_list.pop(0)

    # Display the updated building heights profile.
    bldg_heights_profile_m, _ = buildings.GetBuildingHeightsProfile(bldg_list, lat_lon_profile)
    DisplayBuildingHeightsProfile(dist_profile_m, bldg_heights_profile_m, 'Updated building heights profile (multi-screen)')

    # Calculate path loss along the transmitter to receiver path.
    path_loss_profile_dB = [float('nan')] * len(lat_lon_profile)
    for i in range(1, len(lat_lon_profile)):

        if bldg_heights_profile_m[i] > 0:
            continue # skip indoor locations

        # Extract building metrics to use as inputs for the propagation model (see ITU-R P.1411-12,
        # FIGURE 2 for more details):
        #   d_m: Path length (great-circle distance) from the transmitter to the receiver (meters).
        #   hr_m: Average height of buildings along the path (meters).
        #   b_m: Average building separation distance along the path (meters).
        #   l_m: Length of the path covered by buildings (meters).
        #   w_m (float): Street width at the receiver location (meters), which is the distance between
        #                the two buildings encompassing the receiver. -1 if can't be calculated.
        d_m, hr_m, b_m, l_m, w_m = buildings.GetP1411UrbanMetrics(bldg_list, tx_lat, tx_lon,
                                       rxLat=lat_lon_profile[i][0], rxLon=lat_lon_profile[i][1])
        
        if l_m > 0 and w_m < 0:
            continue # skip points where the street width cannot be obtained

        Lb = itur_p1411.SiteSpecificOverRoofTopsUrban(f_GHz=freq_GHz, d_m=d_m, h1_m=tx_height_m,
                 h2_m=rx_height_m, hr_m=hr_m, l_m=l_m, b_m=b_m, w2_m=w_m,
                 phi_deg=90.0, # Street orientation with respect to the direct path (deg).
                               # When streets separate buildings, ReceiverStreetOrientation() from the
                               # helper.streets module can be used to calculate phi.
                 env=itur_p1411.EnvironmentD.METROPOLITAN_CENTRE) # env is only used when f_GHz <= 2

        path_loss_profile_dB[i] = Lb

    # Plot profiles
    dist_profile_m = dist_profile_m
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax2 = ax1.twinx()
    ax1.plot(dist_profile_m, path_loss_profile_dB, 'g-', label='path loss (dB)')
    ax2.plot(dist_profile_m, bldg_heights_profile_m, color='#B0B0B0')
    ax1.invert_yaxis()
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Path loss (dB)')
    ax2.set_ylabel('Building height (m)')
    ax1.set_title('ITU-R P.1411-12, site specific model for propagation over rooftops in an urban area, {:.1f} GHz'.format(freq_GHz))
    ax1.legend()
    plt.show()


def DisplayBuildingHeightsProfile(dist_profile_m: list[float], bldg_heights_profile_m: list[float], title: str):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.plot(dist_profile_m, bldg_heights_profile_m, color='#B0B0B0')
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Building height (m)')
    ax1.set_title(title)
    plt.show()


if __name__ == '__main__':
    main()
