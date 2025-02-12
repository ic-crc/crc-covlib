"""Additional building related functionalities in support of crc-covlib.

This module allows to get building footprint and height data from either a shapefile or from
OpenStreetMap web services (see GetBuildingsFromShapefile() and GetBuildingsFromOpenStreetMap()).
From this data, additional building releted metrics can be computed along a direct transmission
path (see GetP1411UrbanMetrics(), GetP1411ResidentialMetrics()).
"""
import os
import shapely
import shapefile
import osmnx
import pyproj
import numpy
import numpy.typing as npt
import collections
from ..helper import topography
from ..helper import itur_p2001
from typing import Union
from math import isnan


__all__ = ['LatLonBox', # class
           'Building', # class
           'GetBuildingsFromShapefile',
           'GetBuildingsFromOpenStreetMap',
           'ExportBuildingsToGeojsonFile',
           'GetBuildingHeightsProfile',
           'GetP1411UrbanMetrics',
           'GetP1411ResidentialMetrics'
          ]


class LatLonBox:
    """
    Bounding box in geographical coordinates (WGS 84 or EPSG 4326 assumed).

    Args:
        minLat (float): minimum latitude of bounding box (degrees).
        minLon (float): minimum longitude of bounding box (degrees).
        maxLat (float): maximum latitude of bounding box (degrees).
        maxLon (float): maximum longutide of bounding box (degrees).

    Attributes:
        minLat (float): minimum latitude of bounding box (degrees).
        minLon (float): minimum longitude of bounding box (degrees).
        maxLat (float): maximum latitude of bounding box (degrees).
        maxLon (float): maximum longutide of bounding box (degrees).
    """
    def __init__(self, minLat:float, minLon:float, maxLat:float, maxLon:float):
        self.minLat:float = min(minLat, maxLat)
        self.minLon:float = min(minLon, maxLon)
        self.maxLat:float = max(minLat, maxLat)
        self.maxLon:float = max(minLon, maxLon)


class Building:
    """
    Data class for a building.

    Attributes:
        height_m (float): Height of the building (meters).
        footprint (shapely.Polygon): Coordinates of the building's footprint.
        osmid (int|None): The building's OpenStreetMap ID when applicable, None otherwise.
        footprintCRS (pyproj.crs.CRS): The coordinate reference system of the building's footprint.
    """
    def __init__(self):
        self.height_m: float
        self.footprint: shapely.Polygon
        self.osmid: Union[int,None]
        self.footprintCRS: pyproj.crs.CRS


def GetBuildingsFromShapefile(pathname: str, heightAttribute: str,
                              bounds: Union[LatLonBox,None]=None,
                              toCRS: Union[pyproj.crs.CRS,None]=pyproj.CRS.from_epsg(4326),
                              defaultHeight_m: float=3.0
                              ) -> list[Building]:
    """
    Extracts building height and footprint data from the specified shapefile. The footprints are
    assumed to be stored using the Polygon shape type (5) within the shapefile.

    Args:
        pathname (str): Absolute or relative path to a shapefile (.shp) containing building
            heights and footprints.
        heightAttribute (str): Name of the shapefile attribute to get the building heights from.
        bounds (crc_covlib.helper.buildings.LatLonBox|None): When specified, only the area covered
            by the specified bounds is read. Otherwise the whole content of the file is used.
        toCRS (pyproj.crs.CRS|None): A coordinate reference system (CRS). When specified, the
            extracted footprints are converted to this CRS. Otherwise the footprints are returned
            in their original CRS from the shapefile.
        defaultHeight_m (float): Default height value (meters) to be applied to any building whose
            height cannot be obtained from the heightAttribute value.

    Returns:
        (list[crc_covlib.helper.buildings.Building]): A list of Building objects.
    """
    bldg_list = []

    prj_pathname = os.path.splitext(pathname)[0] + '.prj'
    with open(prj_pathname) as f:
        prj_content = f.read()
        prj_file_crs = pyproj.crs.CRS(prj_content)

    if bounds is not None:
        crs_4326 = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(crs_from=crs_4326, crs_to=prj_file_crs, always_xy=True)
        bbox = transformer.transform_bounds(left=bounds.minLon, bottom=bounds.minLat,
                                            right=bounds.maxLon, top=bounds.maxLat)
    else:
        bbox = None

    footprint_transformer = None
    if toCRS is not None:
        if prj_file_crs.equals(toCRS) == False:
            footprint_transformer = pyproj.Transformer.from_crs(crs_from=prj_file_crs, crs_to=toCRS, always_xy=True)

    sf = shapefile.Reader(pathname)
    #print(sf.fields)
    for shape_rec in sf.iterShapeRecords(bbox=bbox):
        if shape_rec.shape.shapeType == 5: # 5=POLYGON
            if footprint_transformer is not None:
                x_coords, y_coords = list(zip(*(shape_rec.shape.points)))
                new_x_coords, new_y_coords = footprint_transformer.transform(x_coords, y_coords)
                shape_rec.shape.points = list(zip(list(new_x_coords), list(new_y_coords)))
            bldg = Building()
            if len(shape_rec.shape.parts) == 1:
                bldg.footprint = shapely.Polygon(shell=shape_rec.shape.points)
            elif len(shape_rec.shape.parts) > 1:
                shell = shape_rec.shape.points[0 : shape_rec.shape.parts[1]]
                holes = []
                for i in range(1, len(shape_rec.shape.parts)-1):
                    holes.append(shape_rec.shape.points[shape_rec.shape.parts[i] : shape_rec.shape.parts[i+1]])
                holes.append(shape_rec.shape.points[shape_rec.shape.parts[-1] : ])
                bldg.footprint = shapely.Polygon(shell=shell, holes=holes)
            try:
                bldg.height_m = float(shape_rec.record[heightAttribute])
            except:
                bldg.height_m = defaultHeight_m
            try:
                bldg.osmid = int(shape_rec.record['osm_id'])
            except:
                bldg.osmid = None
            if toCRS is not None:
                bldg.footprintCRS = toCRS
            else:
                bldg.footprintCRS = prj_file_crs
            bldg_list.append(bldg)
    return bldg_list


def GetBuildingsFromOpenStreetMap(bounds: LatLonBox,
                                  toCRS: Union[pyproj.crs.CRS,None]=pyproj.CRS.from_epsg(4326),
                                  defaultHeight_m: float=3.0, avgFloorHeight_m: float=3.0
                                  ) -> list[Building]:
    """
    Downloads building footprint data from OpenStreetMap. Building heights are estimated from the
    number of above-ground floors ('building:levels' tag) when present.

    Args:
        bounds (crc_covlib.helper.buildings.LatLonBox): A bounding box from which to get the
            building data.
        toCRS (pyproj.crs.CRS|None): A coordinate reference system (CRS). When specified, the
            downloaded footprints are converted to this CRS. Otherwise the footprints are returned
            in their original CRS from OpenStreetMap.
        defaultHeight_m (float): Default height value (meters) to be applied to any building whose
            height cannot be estimated (i.e. when the 'building:levels' tag is missing).
        avgFloorHeight_m (float): Average floor height (meters) for estimating building heights.

    Returns:
        (list[crc_covlib.helper.buildings.Building]): A list of Building objects.
    """
    bldg_list = []

    osmnx_major_ver = int(osmnx.__version__.split('.')[0])
    if osmnx_major_ver >= 2:
        # bbox in (left, bottom, right, top) format
        bbox = (bounds.minLon, bounds.minLat, bounds.maxLon, bounds.maxLat)
    else:
        # bbox in (north, south, east, west) format
        bbox = (bounds.maxLat, bounds.minLat, bounds.maxLon, bounds.minLon)

    gdf = osmnx.features.features_from_bbox(bbox=bbox, tags={'building': True}) # returns geopandas.GeoDataFrame
    footprints = gdf['geometry'].values.tolist()
    num_levels = gdf['building:levels'].values.tolist() # string or nan
    indexes = gdf.index.values.tolist()
    _, osmids = list(zip(*indexes))

    footprint_transformer = None
    crs_4326 = pyproj.CRS.from_epsg(4326)
    if toCRS is not None:
        if toCRS.equals(crs_4326) == False:
            footprint_transformer = pyproj.Transformer.from_crs(crs_from=crs_4326, crs_to=toCRS, always_xy=True)

    for footprint, floor_count, osmid in zip(footprints, num_levels, osmids):
        if footprint_transformer is not None:
            footprint = _ConvertPolygonCoordsTr(footprint, footprint_transformer)
        bldg = Building()
        bldg.footprint = footprint
        floor_count = float(floor_count)
        if isnan(floor_count):
            bldg.height_m = defaultHeight_m
        else:
            bldg.height_m = float(floor_count)*avgFloorHeight_m
        bldg.osmid = osmid
        if toCRS is not None:
            bldg.footprintCRS = toCRS
        else:
            bldg.footprintCRS = crs_4326
        bldg_list.append(bldg)

    return bldg_list


def ExportBuildingsToGeojsonFile(buildings: list[Building], pathname: str) -> None:
    """
    Exports a list of Building objects to a geojson file.

    Args:
        buildings (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained
            from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
        pathname (str): Absolute or relative path to the geojson file to create or overwrite.
    """
    features = []
    crs_4326 = pyproj.CRS.from_epsg(4326)
    for building in buildings:
        if building.footprintCRS.equals(crs_4326) == False:
            footprint = _ConvertPolygonCoordsCrs(building.footprint, building.footprintCRS, crs_4326)
        else:
            footprint = building.footprint
        str = '{{"type":"Feature","geometry":{},"properties":{{"height (m)":"{:.1f}","osmid":"{}"}}}}'.format(
            shapely.to_geojson(footprint), building.height_m, building.osmid)
        features.append(str)
    str = '{{"type":"FeatureCollection","features":[{}]}}'.format(','.join(features))
    with open(pathname, 'w') as f:
        f.write(str)


def GetBuildingHeightsProfile(buildings: list[Building], latLonProfile: npt.ArrayLike
                             ) -> tuple[npt.NDArray[numpy.float64], list[Building]]:
    """
    Gets a building heights profile (meters) from the specified building data.

    Args:
        buildings (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained
            from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees, EPSG:4326)
            in the form of a 2D list or array. The latitude of the first point should be at
            latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained
            from the topography.GetLatLonProfile() function.

    Return:
        bldgHeightsProfile_m (numpy.typing.NDArray[numpy.float64]): The building heights profile
            (meters) for the points specified in latLonProfile. A value of 0 is used at locations
            where there is no building.
        encounteredBuildings (list[Building]): List of Building objects from buildings that were
            encountered along latLonProfile. Buildings are listed in the order they were
            encountered iterating over latLonProfile.
    """
    epsg4326_bldgs = _GetBuildingsInEPSG4326(buildings)

    # Get buildings that intersect with the lat-lon profile
    intersect_bldgs = []
    lon_lat_profile = numpy.flip(latLonProfile, 1)
    path_linestring = shapely.LineString(lon_lat_profile)
    for bldg in epsg4326_bldgs:
        if path_linestring.intersects(bldg.footprint) == True:
            intersect_bldgs.append(bldg)

    num_points = len(latLonProfile)
    bldg_height_profile = numpy.zeros(num_points, dtype=numpy.float64)
    encountered_bldgs = []
    for i in range(0, num_points):
        pt = shapely.Point(latLonProfile[i][1], latLonProfile[i][0])
        for bldg_index, bldg in enumerate(intersect_bldgs):
            if bldg.footprint.contains(pt) == True:
                bldg_height_profile[i] = intersect_bldgs[bldg_index].height_m
                if intersect_bldgs[bldg_index] not in encountered_bldgs:
                    encountered_bldgs.append(intersect_bldgs[bldg_index])
                break

    return bldg_height_profile, encountered_bldgs


def GetP1411UrbanMetrics(buildings: list[Building], txLat: float, txLon: float,
                         rxLat: float, rxLon: float, res_m: float=1.0, pathExt_m: float=100.0
                         ) -> tuple[float, float, float, float, float]:
    """
    Gets building related metrics for a direct transmitter to receiver path. The metrics mainly
    consists of input parameter values for site specific, over roof-tops urban/suburban propagation
    models to be found in the ITU-R P.1411 recommendation. See FIGURE 2 of ITU-R P.1411-12 for more
    details.

    Args:
        buildings (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained
            from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
        txLat (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
        txLon (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
        rxLat (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
        rxLon (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
        res_m (float): Resolution (meters). Presence of buildings along the path will be evaluated
            about every res_m meters.
        pathExt_m (float): Path extension length (meters). Extra distance passed the receiver to
            look for buildings in order to calculate the street width and building separation
            distance values.
            
    Returns:
        d_m (float): Path length (great-circle distance) from the transmitter to the receiver
            (meters).
        hr_m (float): Average height of buildings along the path (meters). Set to -1 when no
            building is found along the path.
        b_m (float): Average building separation distance along the path (meters). Set to -1 when
            less than 2 buildings are found along the path.
        l_m (float): Length of the path covered by buildings (meters). Set to zero when no building
            is found along the path.
        w_m (float): Street width at the receiver location (meters), which is the distance between
            the two buildings encompassing the receiver. Set to -1 when two such buildings cannot
            be found along the path.
    """
    d_m = itur_p2001.PathLength(lat0=txLat, lon0=txLon, lat1=rxLat, lon1=rxLon)*1000.0
    lat_ext, lon_ext = itur_p2001.IntermediatePathPoint(lat0=txLat, lon0=txLon, lat1=rxLat, lon1=rxLon,
                                                        dist_km=(d_m+pathExt_m)/1000.0)

    bldg_dists_ext_list: list[_BldgDistsToTx] = _GetBuildingsDistancesToTx(buildings=buildings, txLat=txLat,
                                                    txLon=txLon, pathEndLat=lat_ext, pathEndLon=lon_ext, res_m=res_m)
    
    # min/max distances (m) to tx for each building along the tx-rx path (i.e. wihtout the extension)
    bldg_dists_list: list[_BldgDistsToTx] = [bd for bd in bldg_dists_ext_list if bd.maxDist_m < d_m]

    # Calculates hr_m, the average height of buildings (tx-rx path)
    hr_m = -1.0
    if len(bldg_dists_list) > 0:
        bldg_height_sum = sum(bldg.height_m for bldg in bldg_dists_list)
        hr_m = bldg_height_sum / len(bldg_dists_list)

    # Calculates l_m, the length of the path covered by buildings (tx-rx path)
    l_m = 0.0
    if len(bldg_dists_list) > 0:
        l_m = bldg_dists_list[-1].maxDist_m - bldg_dists_list[0].minDist_m

    rx_prev_bldg_dists = None # first building "before" the rx
    if len(bldg_dists_list) > 0:
        rx_prev_bldg_dists = bldg_dists_list[-1]

    rx_next_bldg_dists = None # first building "after" the tx
    for bd in bldg_dists_ext_list:
        if bd.minDist_m > d_m:
            rx_next_bldg_dists = bd
            break

    # Add first building "after" the rx, if any
    if rx_next_bldg_dists is not None:
        bldg_dists_list.append(rx_next_bldg_dists)

    # Calculates b_m, the average building separation distance (tx-rx path + 1 bldg after the rx)
    b_m = -1.0
    sep_dist_list = []
    for i in range(1, len(bldg_dists_list)):
        sep_dist = ((bldg_dists_list[i].minDist_m + bldg_dists_list[i].maxDist_m) / 2.0) -  \
                   ((bldg_dists_list[i-1].minDist_m + bldg_dists_list[i-1].maxDist_m) / 2.0)
        sep_dist_list.append(sep_dist)
    if len(sep_dist_list) > 0:
        b_m = sum(sep_dist_list) / len(sep_dist_list)

    # Calculates w_m, the street width (i.e. distance between previous and next buildings from receiver)
    w_m = -1.0
    if rx_prev_bldg_dists is not None and rx_next_bldg_dists is not None:
        w_m = rx_next_bldg_dists.minDist_m - rx_prev_bldg_dists.maxDist_m

    return (d_m, hr_m, b_m, l_m, w_m)


def GetP1411ResidentialMetrics(buildings: list[Building], txLat: float, txLon: float,
                               rxLat: float, rxLon: float, res_m: float=1.0
                               ) -> tuple[float, float, float, float, float, float]:
    """
    Gets building related metrics for a direct transmitter to receiver path. The metrics mainly
    consists of input parameter values for the site specific, below roof-top to near street level
    residential propagation model to be found in the ITU-R P.1411 recommendation. See FIGURE 12 of
    ITU-R P.1411-12 for more details.

    Args:
        buildings (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained
            from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
        txLat (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
        txLon (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
        rxLat (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
        rxLon (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
        res_m (float): Resolution (meters). Presence of buildings along the path will be evaluated
            about every res_m meters.
            
    Returns:
        d_m (float): Path length (great-circle distance) from the transmitter to the receiver
            (meters).
        hbTx_m (float): Height of nearest building from transmitter in receiver direction (meters).
            Set to -1 when no building is found along the path.
        hbRx_m (float): Feight of nearest building from receiver in transmitter direction (meters).
            Set to -1 when no building is found along the path.
        a_m (float): Distance between transmitter and nearest building from transmitter (meters).
            Set to -1 when no building is found along the path.
        b_m (float): Distance between nearest buildings from transmitter and receiver (meters). Set
            to -1 when no building is found along the path.
        c_m (float): Distance between receiver and nearest building from receiver (meters). Set to
            -1 when no building is found along the path.
    """
    d_m = itur_p2001.PathLength(lat0=txLat, lon0=txLon, lat1=rxLat, lon1=rxLon)*1000.0
    bldg_dists_list: list[_BldgDistsToTx] = _GetBuildingsDistancesToTx(buildings=buildings, txLat=txLat,
                                                txLon=txLon, pathEndLat=rxLat, pathEndLon=rxLon, res_m=res_m)
    
    hbTx_m = -1.0
    hbRx_m = -1.0
    a_m = -1.0
    b_m = -1.0
    c_m = -1.0
    if len(bldg_dists_list) > 0:
        hbTx_m = bldg_dists_list[0].height_m
        hbRx_m = bldg_dists_list[-1].height_m
        a_m = bldg_dists_list[0].minDist_m
        c_m = d_m - bldg_dists_list[-1].maxDist_m
        if len(bldg_dists_list) == 1:
            b_m = 0.0
        else:
            b_m = d_m - a_m - c_m

    return (d_m, hbTx_m, hbRx_m, a_m, b_m, c_m)


class _BldgDistsToTx:
    def __init__(self, height_m:float, minDist_m:float, maxDist_m:float):
        self.height_m:float = height_m
        self.minDist_m:float = minDist_m # min distance to tx along path (m)
        self.maxDist_m:float = maxDist_m # max distance to tx along path (m)


def _GetBuildingsDistancesToTx(buildings: list[Building], txLat: float, txLon: float,
                               pathEndLat: float, pathEndLon: float, res_m: float
                               ) -> list[_BldgDistsToTx]:
    lat_lon_profile = topography.GetLatLonProfile(lat0=txLat, lon0=txLon, lat1=pathEndLat,
                                                  lon1=pathEndLon, res_m=res_m, minNumPoints=3)
    d_m = itur_p2001.PathLength(lat0=txLat, lon0=txLon, lat1=pathEndLat, lon1=pathEndLon)*1000.0
    delta_dist_m = d_m / (len(lat_lon_profile)-1) # precise value as res_m is approximate
    epsg4326_bldgs = _GetBuildingsInEPSG4326(buildings)

    # Get buildings that intersect with the path
    intersect_bldgs = []
    lon_lat_profile = numpy.flip(lat_lon_profile, 1)
    path_linestring = shapely.LineString(lon_lat_profile)
    for bldg in epsg4326_bldgs:
        if path_linestring.intersects(bldg.footprint) == True:
            intersect_bldgs.append(bldg)

    # Build profile of encountered buildings along the path (i.e. index of building in 
    # intersect_bldgs or -1 where there is no building)
    bldg_index_profile = [-1]*len(lat_lon_profile)
    for profile_index, lat_lon in enumerate(lat_lon_profile):
        pt = shapely.Point(lat_lon[1], lat_lon[0])
        for bldg_index, bldg in enumerate(intersect_bldgs):
            if bldg.footprint.contains(pt) == True:
                bldg_index_profile[profile_index] = bldg_index
                break

    # Get min/max distances (m) to tx for each building along the path
    bldg_dists_dict = collections.OrderedDict()
    for i in range(0, len(bldg_index_profile)):
        bldg_index = bldg_index_profile[i]
        if bldg_index != -1:
            di_m = i*delta_dist_m
            bldg_dists_dict.setdefault(bldg_index, _BldgDistsToTx(
                                                       height_m=intersect_bldgs[bldg_index].height_m,
                                                       minDist_m=max(0, di_m-(delta_dist_m/2.0)),
                                                       maxDist_m=min(d_m, di_m+(delta_dist_m/2.0))))
            bldg_dists_dict[bldg_index].maxDist_m = min(d_m, di_m+(delta_dist_m/2.0))
    bldg_dists_list = list(bldg_dists_dict.values()) # in the order that buildings are encountered along the path

    return bldg_dists_list


def _ConvertPolygonCoordsTr(polygon: shapely.Polygon, transformer: pyproj.Transformer) -> shapely.Polygon:
    x_coords, y_coords = list(zip(*(polygon.exterior.coords)))
    new_x_coords, new_y_coords = transformer.transform(x_coords, y_coords)
    shell = list(zip(list(new_x_coords), list(new_y_coords)))

    holes = []
    for linear_ring in polygon.interiors:
        x_coords, y_coords = list(zip(*(linear_ring.coords)))
        new_x_coords, new_y_coords = transformer.transform(x_coords, y_coords)
        hole = list(zip(list(new_x_coords), list(new_y_coords)))
        holes.append(hole)

    new_polygon = shapely.Polygon(shell=shell, holes=holes)
    return new_polygon


def _ConvertPolygonCoordsCrs(polygon: shapely.Polygon, fromCRS: pyproj.crs.CRS, toCRS: pyproj.crs.CRS) -> shapely.Polygon:
    transformer = pyproj.Transformer.from_crs(crs_from=fromCRS, crs_to=toCRS, always_xy=True)
    return _ConvertPolygonCoordsTr(polygon, transformer)


def _GetBuildingsInEPSG4326(buildings: list[Building]) -> list[Building]:
    new_bldg_list = []
    crs_4326 = pyproj.CRS.from_epsg(4326)
    for building in buildings:
        if building.footprintCRS.equals(crs_4326) == False:
            new_bldg = Building()
            new_bldg.footprint = _ConvertPolygonCoordsCrs(building.footprint, building.footprintCRS, crs_4326)
            new_bldg.height_m = building.height_m
            new_bldg.osmid = building.osmid
            new_bldg.footprintCRS = crs_4326
            new_bldg_list.append(new_bldg)
        else:
            new_bldg_list.append(building)
    return new_bldg_list
