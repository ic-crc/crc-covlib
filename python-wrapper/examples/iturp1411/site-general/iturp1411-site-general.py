import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../../'))

from crc_covlib.helper import buildings
from crc_covlib.helper import topography
from crc_covlib.helper import itur_p2001
from crc_covlib.helper import itur_p1411
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')
warnings.filterwarnings("ignore", category=FutureWarning, module='crc_covlib.helper.buildings')


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R P.1411-12, site general model for propagation within street canyons')

    freq_GHz = 28.0
    tx_lat = 45.419626
    tx_lon = -75.6999886
    rx_lat = 45.4193700
    rx_lon = -75.6970786
    radio_env = itur_p1411.EnvironmentA.URBAN_HIGH_RISE
    bldg_bounds = buildings.LatLonBox(minLat=min(tx_lat, rx_lat), minLon=min(tx_lon, rx_lon), 
                                      maxLat=max(tx_lat, rx_lat), maxLon=max(tx_lon, rx_lon))

    # Download building footprints from OpenStreetMap. Building heights are estimated from the
    # number of floors ('building:levels' tag) when present. Otherwise buildings are assigned
    # the specified default height in meters. This building data will be used to determine whether
    # a path is in line-of-sight (LoS) or non-line-of-sight (NLoS).
    print('Getting building footprint data...')
    bldg_list = buildings.GetBuildingsFromOpenStreetMap(bounds=bldg_bounds, defaultHeight_m=3.0)
    
    # Alternately, building footprints and heights may be obtained from a shapefile (.shp):
    # bldg_list = buildings.GetBuildingsFromShapefile(pathname='C:/exemple.shp',
    #                                                 heightAttribute='to specify',
    #                                                 defaultHeight_m=3.0)

    # Optionally, obtained footprints may be exported to geojson and visualized using a
    # GIS application.                
    buildings.ExportBuildingsToGeojsonFile(buildings=bldg_list,
                                           pathname=os.path.join(script_dir, 'osm_footprints.geojson'))
    
    sampling_res_m = 1.0 # sampling resolution, in meters
    lat_lon_profile = topography.GetLatLonProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res_m)
    dist_profile_km = topography.GetDistanceProfile(tx_lat, tx_lon, rx_lat, rx_lon, sampling_res_m)
    bldg_heights_profile_m, _ = buildings.GetBuildingHeightsProfile(buildings=bldg_list, latLonProfile=lat_lon_profile)
    path_loss_profile_dB = [float('nan')] * len(dist_profile_km)
    path_loss_profile_dB_out_of_range = [float('nan')] * len(dist_profile_km)

    print('Computing path loss...')
    for i in range(1, len(dist_profile_km)):

        if bldg_heights_profile_m[i] > 0:
            continue # skip indoor locations

        # Site-general models from ITU-R P.1411 do not use antenna height parameters. Here antenna
        # height values of 1.5m are used only to help determine whether there is a LoS or not.
        is_los = itur_p2001.IsLOS(latt=tx_lat, lont=tx_lon, ht_mamsl=1.5,
                                  latr=lat_lon_profile[i][0], lonr=lat_lon_profile[i][1], hr_mamsl=1.5,
                                  dProfile_km=dist_profile_km[:i], hProfile_mamsl=bldg_heights_profile_m[:i])
        
        if is_los == True:
            path_type = itur_p1411.PathType.LOS
        else:
            path_type = itur_p1411.PathType.NLOS

        # Note: Two other functions for site general models are available, namely SiteGeneralOverRoofTops() and
        # SiteGeneralNearStreetLevel().
        Lb, warn_flag = itur_p1411.SiteGeneralWithinStreetCanyons(f_GHz=freq_GHz, d_m=dist_profile_km[i]*1000,
                                                                  env=radio_env, path=path_type,
                                                                  addGaussianRandomVar=False)
        
        if warn_flag == 0: # warning flag indicating whether the frequency or distance is out of range for the model
            path_loss_profile_dB[i] = Lb
        else:
            path_loss_profile_dB_out_of_range[i] = Lb

    # Plot profiles
    dist_profile_m = dist_profile_km * 1000
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax2 = ax1.twinx()
    ax1.plot(dist_profile_m, path_loss_profile_dB, 'g-', label='path loss (dB)')
    ax1.plot(dist_profile_m, path_loss_profile_dB_out_of_range, 'xr', label="path loss (dB) - out of model's valid range")
    ax2.plot(dist_profile_m, bldg_heights_profile_m, color='#B0B0B0')
    ax1.invert_yaxis()
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Path loss (dB)')
    ax2.set_ylabel('Building height (m)')
    ax1.set_title('ITU-R P.1411-12, site general model for propagation within street canyons, {:.1f} GHz'.format(freq_GHz))
    ax1.legend()
    plt.show()
