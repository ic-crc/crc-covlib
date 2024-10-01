import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
from crc_covlib.helper import topography


if __name__ == '__main__':

    print('\ncrc-covlib - helper.topography.LoadFileAsCustomLandCoverData()')

    sim = covlib.Simulation()

    rx_area = (45.37914, -75.81922, 45.47148, -75.61225)

    # Here we will use a ESA Worldcover land cover data file as the source for land cover data.
    # The format and CRS (coordinate reference system) of this file are supported by crc-covlib and
    # it may be read using the LAND_COVER_ESA_WORLDCOVER data source type. However, as a demonstration
    # we will use the LAND_COVER_CUSTOM data source type here and read the file using the
    # crc_covlib.helper.topography.LoadFileAsCustomLandCoverData() function. This function uses third-party
    # libraries rasterio (GDAL) and numpy in order to support a large variety of raster file formats
    # and CRS. The land cover data is reprojected (if needed) and sent to the simulation object in 
    # crc-covlib's custom data format. The technique may be used to use elevation and/or land cover data
    # files that are not supported by crc-covlib's core functionalities (C++ implementation).
    #
    # The LoadFileAsCustomLandCoverData() function takes an optional third parameter (bounds) to specify
    # what area from the file is to be used. Otherwise the whole uncompressed content of the file is
    # read and copied into the simulation object, and this may fail  or take a great amount of time for
    # large files.
    LC_CUSTOM = covlib.LandCoverDataSource.LAND_COVER_CUSTOM
    sim.SetPrimaryLandCoverDataSource(LC_CUSTOM)
    cdi: topography.CustomDataInfo
    cdi = topography.LoadFileAsCustomLandCoverData(
        sim,
        os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover/ESA_WorldCover_10m_2021_v200_N45W078_Map.tif'),
        bounds=topography.LatLonBox(*rx_area))
    if cdi is not None:
        print('Loaded {}x{} land cover data points'.format(cdi.numHorizSamples, cdi.numVertSamples))
    else:
        print('Warning: failed to load land cover data!')

    sim.SetReceptionAreaCorners(*rx_area)

    # Export the land cover data that the simulation object is now "seeing" over the simulation area.
    # The exported file is in the BIL format, which can be displayed using most GIS software (QGIS for
    # example). This is an easy way to assess that the land cover data file we used was read properly
    # and covers the reception area.
    sim.ExportReceptionAreaLandCoverClassesToBilFile(
        os.path.join(script_dir, 'topo-helper-land-cover.bil'), 
        cdi.numHorizSamples,
        cdi.numVertSamples,
        mapValues=False)

    # If we wanted to use this land cover data in a simulation, we would need to map the land cover classes
    # from the land cover data file to other classes or values the propagation model can understand.
    # For example, here we map the land cover class 80 (representing "water" in ESA WorldCover products) to 
    # the P.1812 clutter type 1 (interpreted as "water" by the ITU-R P.1812 propagation model).
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    CLUT = covlib.P1812ClutterCategory
    sim.SetLandCoverClassMapping(LC_CUSTOM, 80, P1812, CLUT.P1812_WATER_SEA) # map 'Permanent water bodies' (80) to 'water/sea' (1)
    sim.SetDefaultLandCoverClassMapping(LC_CUSTOM, P1812, CLUT.P1812_OPEN_RURAL) # map all other ESA WorldCover classes to 'open/rural' (2)

    # Select ITU-R P.1812 as the current propagation model
    sim.SetPropagationModel(P1812)

    # Export the land cover classes again but this time the 4th parameter (mapValues) is set to True.
    # This means the export process will use the defined mappings between the currently selected land
    # cover data source and the currently selected propagation model. Since we only mapped to "water/sea"
    # and "open/rural", we should now only see two different types of clutter when displaying the 
    # exported bil file.
    sim.ExportReceptionAreaLandCoverClassesToBilFile(
        os.path.join(script_dir, 'topo-helper-mapped-land-cover.bil'), 
        cdi.numHorizSamples,
        cdi.numVertSamples,
        mapValues=True)

    print('Program execution completed\n')