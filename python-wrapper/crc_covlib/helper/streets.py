# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Additional street related functionalities using OpenStreetMap data in support of crc-covlib.

This module aims at determining potential propagation paths for scenarios where antennas are below
rooftops and the radio path is understood to occur mainly through street canyons. Typical usage is
the obtention of input data for propagation models such as those found in ITU-R P.1411.

Typical workflow:
  1.GetStreetGraphFromBox() or GetStreetGraphFromPoints() to download OpenStreetMap data for the
    area of interest. This data may be visualized using DisplayStreetGraph().
  2.Use GetStreetCanyonsRadioPaths() to compute the shortest path(s) along street lines from the
    transmitter to the receiver stations. GetStreetCanyonsRadioPaths() returns one or more
    StreetCanyonsRadioPath objects. ExportRadioPathToGeojsonFile() may be used to visualize those
    path objects from a GIS software.
  3.Optionally, computed StreetCanyonsRadioPath objects may be further simplified (i.e. that is, to
    reduce the number of vertices in the path) using SimplifyRadioPath() or SortRadioPathsByStreet-
    CornerCount().
  4.The DistancesToStreetCorners(), StreetCornerAngles() and ReceiverStreetOrientation() functions,
    as well as attributes from the StreetCanyonsRadioPath objects (distances_m, turnAngles_deg,
    etc.) may be used for input parameters to propagation models (ITU-R P.1411 for instance).
"""
import networkx as nx
import osmnx as ox
import shapely as shp
import geopandas as gpd
from typing import Union
from math import degrees, atan2


__all__ = ['StreetGraph', # class
           'StreetCanyonsRadioPath', # class
           'GetStreetGraphFromBox',
           'GetStreetGraphFromPoints',
           'DisplayStreetGraph',
           'GetStreetCanyonsRadioPaths',
           'ExportRadioPathToGeojsonFile',
           'SimplifyRadioPath',
           'DistancesToStreetCorners',
           'StreetCornerAngles',
           'SortRadioPathsByStreetCornerCount',
           'ReceiverStreetOrientation'
          ]


class StreetGraph:
    """
    Street graph generated from OpenStreetMap (OSM) data. An instance may be obtained from
    GetStreetGraphFromBox() or GetStreetGraphFromPoints() and is typically used as input for
    GetStreetCanyonsRadioPaths().
    """
    def __init__(self):
        self._OSMGraph: nx.MultiDiGraph
        self._edges: gpd.GeoDataFrame
        self._nodes: gpd.GeoDataFrame


class StreetCanyonsRadioPath:
    """
    Data class for storing details about a propagation path occuring through street canyons
    (i.e. propagation along street lines). One ore more instance(s) may be obtained from
    GetStreetCanyonsRadioPaths().

    Attributes:
        latLonPath (list[tuple[float, float]]): List of n (latitude, longitude) tuples constituting
            the vertices for the propagation path along street segments, from transmitter to
            receiver (degrees, EPSG:4326). 
        turnAngles_deg (list[float]): List of n-2 turn angles between street segments in latLonPath
            (degrees). The first angle is at latLonPath[1] and the last angle is at latLonPath[-2].
            Angle values are from 0 (going straight forward) to 180 (u-turn). Note that turn angles
            are always positive so there is no differentiation between right-hand and left-hand
            turns at the same angle.
        isIntersection (list[bool]): List of n-2 boolean values indicating whether points between
            street segments in latLonPath are street intersections. The first boolean value is at
            latLonPath[1] and the last one is at latLonPath[-2].
        distances_m (list[float]): List of n-1 street segment distances (meters) in latLonPath.
            The first value is the distance between latLonPath[0] and latLonPath[1], the last value
            is the distance between latLonPath[-2] and latLonPath[-1].
        txLatLon (tuple[float, float]): Transmitter's location as a (latitude, longitude) tuple
            (degrees, EPSG:4326). latLonPath[0] is the closest point from txLatLon that is located
            on a street segment.
        rxLatLon (tuple[float, float]): Receiver's location as a (latitude, longitude) tuple
            (degrees, EPSG:4326). latLonPath[-1] is the closest point from rxLatLon that is located
            on a street segment.
    """
    def __init__(self):
        self.latLonPath: list[tuple[float, float]] = [] # n items
        self.turnAngles_deg: list[float] = [] # n-2 items
        self.isIntersection: list[bool] = [] # n-2 items
        self.distances_m: list[float] = [] # n-1 items
        self.txLatLon: tuple[float, float] = (0, 0)
        self.rxLatLon: tuple[float, float] = (0, 0)
        self._projectedCoordsPath: list[tuple[float, float]] = [] # n items
        self._projectedCRS: str = ''


def GetStreetGraphFromBox(minLat: float, minLon: float, maxLat: float, maxLon: float
                          ) -> StreetGraph:
    """
    Downloads OpenStreetMap (OSM) data for the specified boundaries and creates a graph using OSMnx.
    The returned graph object may be used as input for the GetStreetCanyonsRadioPaths() function.

    Args:
        minLat (float): Minimum latitude boundary for the OSM data download (degrees, EPSG:4326),
            with -90 <= minLat <= 90.
        minLon (float): Minimum longitude boundary for the OSM data download (degrees, EPSG:4326),
            with -180 <= minLon <= 180.
        maxLat (float): Maximum latitude boundary for the OSM data download (degrees, EPSG:4326),
            with -90 <= maxLat <= 90.
        maxLon (float): Maximum longitude boundary for the OSM data download (degrees, EPSG:4326),
            with -180 <= maxLon <= 180.

    Returns:
        (crc_covlib.helper.streets.StreetGraph): OSM data street graph.
    """
    return _GetStreetsGraph(minLat=minLat, minLon=minLon, maxLat=maxLat, maxLon=maxLon)


def GetStreetGraphFromPoints(latLonPoints: list[tuple[float,float]], bufferDist_m: float=250
                            ) -> StreetGraph:
    """
    Downloads OpenStreetMap (OSM) data for a boundary box containing the specified buffered points
    and creates a graph using OSMnx. The returned graph object may be used as input for the
    GetStreetCanyonsRadioPaths() function.

    Args:
        latLonPoints (list[tuple[float,float]]): List of (latitude, longitude) tuples for
            determining the OSM data download boundary box (degrees, EPSG:4326).
        bufferDist_m (float): Buffer distance around the specified points (meters).

    Returns:
        (crc_covlib.helper.streets.StreetGraph): OSM data street graph.
    """
    latitudes, longitudes = zip(*latLonPoints)
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    bbox1 = ox.utils_geo.bbox_from_point(point=(min_lat, min_lon), dist=bufferDist_m,
                                         project_utm=False, return_crs=False)
    bbox2 = ox.utils_geo.bbox_from_point(point=(max_lat, max_lon), dist=bufferDist_m,
                                         project_utm=False, return_crs=False)
    if _osmnxMajorVer >= 2:
        # bbox in (left, bottom, right, top) format
        min_lat, min_lon, max_lat, max_lon = min(bbox1[1], bbox2[1]), min(bbox1[0], bbox2[0]), \
                                             max(bbox1[3], bbox2[3]), max(bbox1[2], bbox2[2])
    else:
        # bbox in (north, south, east, west) format
        min_lat, min_lon, max_lat, max_lon = min(bbox1[1], bbox2[1]), min(bbox1[3], bbox2[3]), \
                                             max(bbox1[0], bbox2[0]), max(bbox1[2], bbox2[2])
    return _GetStreetsGraph(minLat=min_lat, minLon=min_lon, maxLat=max_lat, maxLon=max_lon)


def _GetStreetsGraph(minLat: float, minLon: float, maxLat: float, maxLon: float) -> StreetGraph:
    """
    Downloads OpenStreetMap (OSM) data for the specified boundaries and creates a graph using OSMnx.
    The returned graph object may be used as input for the GetStreetCanyonsRadioPaths() function.

    Args:
        minLat (float): Minimum latitude boundary for the OSM data download (degrees, EPSG:4326),
            with -90 <= minLat <= 90.
        minLon (float): Minimum longitude boundary for the OSM data download (degrees, EPSG:4326),
            with -180 <= minLon <= 180.
        maxLat (float): Maximum latitude boundary for the OSM data download (degrees, EPSG:4326),
            with -90 <= maxLat <= 90.
        maxLon (float): Maximum longitude boundary for the OSM data download (degrees, EPSG:4326),
            with -180 <= maxLon <= 180.

    Returns:
        (crc_covlib.helper.streets.StreetGraph): OSM data street graph.
    """
    if _osmnxMajorVer >= 2:
        # bbox in (left, bottom, right, top) format
        bbox = (min(minLon, maxLon), min(minLat, maxLat), max(minLon, maxLon), max(minLat, maxLat))
    else:
        # bbox in (north, south, east, west) format
        bbox = (max(minLat, maxLat), min(minLat, maxLat), min(minLon, maxLon), max(minLon, maxLon))
    G = ox.graph_from_bbox(bbox=bbox, network_type='drive', simplify=True, retain_all=True,
                           truncate_by_edge=True, custom_filter=None)

    centre_lat = (minLat+maxLat)/2
    centre_lon = (minLon+maxLon)/2
    _, crs_utm = ox.projection.project_geometry(shp.Point(centre_lon, centre_lat),
                                                crs='epsg:4326', to_crs=None, to_latlong=False)
    G_utm = ox.projection.project_graph(G=G, to_crs=crs_utm, to_latlong=False)
    G_utm = ox.convert.to_undirected(G=G_utm) # radiowaves do not follow oneway signs

    gdf_edges = ox.convert.graph_to_gdfs(G=G_utm, nodes=False, edges=True, node_geometry=False,
                                    fill_edge_geometry=True) [['from', 'to', 'length', 'geometry']]
    gdf_nodes = ox.convert.graph_to_gdfs(G=G_utm, nodes=True, edges=False, node_geometry=False,
                                         fill_edge_geometry=False) [['x', 'y']]

    streets_graph = StreetGraph()
    streets_graph._OSMGraph = G_utm
    streets_graph._edges = gdf_edges
    streets_graph._nodes = gdf_nodes
    return streets_graph


def DisplayStreetGraph(graph: StreetGraph) -> None:
    """
    Displays a StreetGraph object.

    Args:
        graph (crc_covlib.helper.streets.StreetGraph): The StreetGraph object to be displayed.
    """
    fig, ax = ox.plot_graph(graph._OSMGraph)  


def GetStreetCanyonsRadioPaths(graph: Union[StreetGraph,None], txLat: float, txLon: float,
                               rxLat: float, rxLon: float, numPaths: int
                              ) -> list[StreetCanyonsRadioPath]:
    """
    Computes one or more radio paths along street lines from the transmitter location to the
    receiver location. Returned paths are those to be found having the smallest travel distances.
    The information contained in the returned object(s) may be used as input for some propagation
    models (ITU-R P.1411 for instance) where antennas are below rooftops and the radio path is
    understood to occur mainly through street canyons.

    Args:
        graph (crc_covlib.helper.streets.StreetGraph|None): A StreetGraph object obtained from
            GetStreetGraphFromBox() or GetStreetGraphFromPoints(). When set to None, a new graph
            object is produced internally based on the transmitter and receiver locations.
        txLat (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
        txLon (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
        rxLat (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
        rxLon (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
        numPaths (int): Maximum number of paths to be computed and returned, with 1 <= numPaths.

    Returns:
        (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): List of StreetCanyonsRadioPath
            objects. The number of items in the returned list will usually be equal to numPaths,
            but it can be less if less than numPaths paths could be computed.
    """
    if graph is None:
        graph = GetStreetGraphFromPoints(latLonPoints=[(txLat, txLon),(rxLat, rxLon)])

    G_utm: nx.MultiDiGraph = graph._OSMGraph
    crs_utm = graph._OSMGraph.graph['crs']
    tx_utm: shp.Point = ox.projection.project_geometry(shp.Point(txLon, txLat), crs='epsg:4326',
                                                       to_crs=crs_utm, to_latlong=False)[0]
    rx_utm: shp.Point = ox.projection.project_geometry(shp.Point(rxLon, rxLat), crs='epsg:4326',
                                                       to_crs=crs_utm, to_latlong=False)[0]

    tx_edge: tuple[float, float, float] # (nodeID, nodeID, 0)
    rx_edge: tuple[float, float, float]
    tx_edge = ox.distance.nearest_edges(G=G_utm, X=tx_utm.x, Y=tx_utm.y, return_dist=False)
    rx_edge = ox.distance.nearest_edges(G=G_utm, X=rx_utm.x, Y=rx_utm.y, return_dist=False)

    # place tx and rx "on the road"
    snapped_tx_utm: shp.Point = _NearestEdgePoint(graph=graph, point=tx_utm, edge=tx_edge)
    snapped_rx_utm: shp.Point = _NearestEdgePoint(graph=graph, point=rx_utm, edge=rx_edge)

    # shortest_paths uses nodes, not random points on an edge (i.e. street), so compute paths
    # using all possible nodeID combinations. Here a path is a list of nodes.
    gen_list = [
        ox.k_shortest_paths(G=G_utm, orig=tx_edge[0], dest=rx_edge[0], k=numPaths, weight='length'),
        ox.k_shortest_paths(G=G_utm, orig=tx_edge[1], dest=rx_edge[0], k=numPaths, weight='length'),
        ox.k_shortest_paths(G=G_utm, orig=tx_edge[0], dest=rx_edge[1], k=numPaths, weight='length'),
        ox.k_shortest_paths(G=G_utm, orig=tx_edge[1], dest=rx_edge[1], k=numPaths, weight='length')]
    temp_path_list = []
    for generator in gen_list:
        for path in generator:
            temp_path_list.append(path)

    path_list =[]
    two_nodes_path_found = False
    for path in temp_path_list:
        path = _EnsureEdgeIsFirstEdgeOfPath(path=path, edge=tx_edge)
        path = _EnsureEdgeIsLastEdgeOfPath(path=path, edge=rx_edge)
        if len(path) == 2:
            if two_nodes_path_found == True:
                continue # when tx and rx are on same edge, avoid having both path [nodeA, nodeB] and [nodeB, nodeA]
            else:
                two_nodes_path_found = True
        if path not in path_list:
            path_list.append(path)

    rp_list: list[StreetCanyonsRadioPath] = []
    for path in path_list:
        rp = StreetCanyonsRadioPath()
        ls = shp.LineString()

        if len(path)==2:
            # Case where tx and rx are on same street with no intersection (i.e no node) between
            # the two.
            temp_ls = _EdgeLineString(graph=graph, edge=(path[0], path[1], 0))
            ls_a, ls_b = _SplitLineString(temp_ls, snapped_tx_utm)
            try:
                ls_c, ls_d = _SplitLineString(ls_a, snapped_rx_utm)
            except:
                ls_c, ls_d = _SplitLineString(ls_b, snapped_rx_utm)
            ls_c_first = shp.Point(ls_c.coords[0])
            ls_c_last = shp.Point(ls_c.coords[-1])
            if snapped_tx_utm == ls_c_first or snapped_tx_utm == ls_c_last:
                ls = ls_c
            else:
                ls = ls_d
            if snapped_tx_utm != shp.Point(ls.coords[0]):
                ls = ls.reverse()
            rp.isIntersection = [False for _ in range(1, len(ls.coords)-1)]
        else:
            first_ls = _EdgeLineString(graph=graph, edge=(path[0], path[1], 0))
            _, first_ls = _SplitLineString(first_ls, snapped_tx_utm)
            ls = first_ls
            rp.isIntersection = [False for _ in range(1, len(ls.coords)-1)]
            for i in range(1, len(path)-2):
                temp_ls = _EdgeLineString(graph=graph, edge=(path[i], path[i+1], 0))
                ls = _MergeLineStrings(ls, temp_ls)
                if len(ls.coords)-2 > len(rp.isIntersection):
                    rp.isIntersection.append(True)
                while len(ls.coords)-2 > len(rp.isIntersection):
                    rp.isIntersection.append(False)
            last_ls = _EdgeLineString(graph=graph, edge=(path[-2], path[-1], 0))
            last_ls, _ = _SplitLineString(last_ls, snapped_rx_utm)
            if last_ls is not None:
                ls = _MergeLineStrings(ls, last_ls)
                if len(ls.coords)-2 > len(rp.isIntersection):
                    rp.isIntersection.append(True)
                while len(ls.coords)-2 > len(rp.isIntersection):
                    rp.isIntersection.append(False)

        lonLatLinestring, _ = ox.projection.project_geometry(ls, crs=crs_utm, to_crs=None,
                                                             to_latlong=True)
        rp.latLonPath = _LineStringToCoordTuples(lonLatLinestring, True)
        rp.distances_m = _LineStringDistances(ls)
        rp.turnAngles_deg = _LineStringTurnAnglesDeg(ls)
        rp.txLatLon = (txLat, txLon)
        rp.rxLatLon = (rxLat, rxLon)
        rp._projectedCoordsPath = _LineStringToCoordTuples(ls, False)
        rp._projectedCRS = crs_utm.srs
        rp_list.append(rp)

    rp_list = sorted(rp_list, key=lambda rp: (sum(rp.distances_m)))
    return rp_list[:numPaths]


def ExportRadioPathToGeojsonFile(radioPath: StreetCanyonsRadioPath, pathname: str) -> None:
    """
    Exports a street canyons radio path to a geojson file.

    Args:
        radioPath (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath
            object obtained from the GetStreetCanyonsRadioPaths() function.
        pathname (str): Absolute or relative path to the geojson file to create or overwrite.
    """
    features = []

    str = '{{"type":"Feature","geometry":{},"properties":{{"name":"transmitter"}}}}'.format(
        shp.to_geojson(shp.Point(radioPath.txLatLon[1], radioPath.txLatLon[0])))
    features.append(str)

    str = '{{"type":"Feature","geometry":{},"properties":{{"name":"receiver"}}}}'.format(
        shp.to_geojson(shp.Point(radioPath.rxLatLon[1], radioPath.rxLatLon[0])))
    features.append(str)

    reversed_coords = [tuple([coord[1], coord[0]]) for coord in radioPath.latLonPath]
    str = '{{"type":"Feature","geometry":{}}}'.format(
        shp.to_geojson(shp.LineString(reversed_coords)))
    features.append(str)

    for i, is_intersection in enumerate(radioPath.isIntersection):
        if is_intersection == True:
            str = '{{"type":"Feature","geometry":{},"properties":{{"name":"intersection","turn angle (deg)":{:.2f}}}}}'.format(
                shp.to_geojson(shp.Point(radioPath.latLonPath[i+1][1], radioPath.latLonPath[i+1][0])),
                radioPath.turnAngles_deg[i])
            features.append(str)

    str = '{{"type":"FeatureCollection","features":[{}]}}'.format(','.join(features))

    with open(pathname, 'w') as f:
        f.write(str)


def SimplifyRadioPath(radioPath: StreetCanyonsRadioPath, tolerance_m: float=12) -> StreetCanyonsRadioPath:
    """
    Simplifies a street canyons radio path (i.e. reduces the number of vertices in the path).

    Args:
        radioPath (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath
            object obtained from the GetStreetCanyonsRadioPaths() function.
        tolerance_m (float): Maximum allowed path geometry displacement (meters).

    Returns:
        (crc_covlib.helper.streets.StreetCanyonsRadioPath): A new street canyons radio path that is
            a simplified version of the original.
    """
    new_rp = StreetCanyonsRadioPath()
    original_ls_m = shp.LineString(radioPath._projectedCoordsPath)
    simplified_ls_m = shp.simplify(geometry=original_ls_m, tolerance=tolerance_m, preserve_topology=True)
    simplified_ls_dd, _ = ox.projection.project_geometry(simplified_ls_m, crs=radioPath._projectedCRS,
                                                         to_crs=None, to_latlong=True)
    new_rp.latLonPath = _LineStringToCoordTuples(simplified_ls_dd, True)
    new_rp._projectedCoordsPath = _LineStringToCoordTuples(simplified_ls_m, False)
    new_rp.distances_m = _LineStringDistances(simplified_ls_m)
    new_rp.turnAngles_deg = _LineStringTurnAnglesDeg(simplified_ls_m)
    for i in range(1, len(new_rp._projectedCoordsPath)-1):
        new_rp.isIntersection.append(_IsIntersection(radioPath, new_rp._projectedCoordsPath[i][0],
                                                     new_rp._projectedCoordsPath[i][1]))
    new_rp.txLatLon = radioPath.txLatLon
    new_rp.rxLatLon = radioPath.rxLatLon
    new_rp._projectedCRS = radioPath._projectedCRS
    return new_rp


def DistancesToStreetCorners(radioPath: StreetCanyonsRadioPath,
                             turnAngleThreshold_deg: float=20) -> list[float]:
    """
    Gets a list of travel distances, in meters, to the next street corner (i.e. the next turn point
    at a turn angle of at least turnAngleThreshold_deg) or to the receiver, along the specified
    street canyons radio path.

    Examples of return values:
        - [50, 100, 30]:
            Travel distance from tx station to first street corner is 50m.
            Travel distance from first street corner to second street corner is 100m.
            Travel distance from second street corner to rx station is 30m.
        - [75]:
            Travel distance from tx station to rx station is 75m (no street corner encountered).

    Args:
        radioPath (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath
            object obtained from the GetStreetCanyonsRadioPaths() function.
        turnAngleThreshold_deg (float): Turn angle threshold used to identify street corners
            (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to
            zero means the path is going straight through the intersection. A turn angle value
            around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle
            will vary depending on the angle at which the streets are crossing).

    Returns:
        (list[float]): List of travel distances, in meters, to the next street corner or to the
            receiver.
    """
    if len(radioPath.distances_m) == 0:
        return [0.0]
    x_m = [radioPath.distances_m[0]]
    for i in range(0, len(radioPath.turnAngles_deg)):
        if radioPath.turnAngles_deg[i] >= turnAngleThreshold_deg:
            x_m.append(radioPath.distances_m[i+1])
        else:
            x_m[-1] += radioPath.distances_m[i+1]
    return x_m


def StreetCornerAngles(radioPath: StreetCanyonsRadioPath,
                       turnAngleThreshold_deg: float=20) -> list[float]:
    """
    Gets the list of street corner angles from a street canyons radio path, that is, the list
    of street corner angles along the path where the turn angle is equal or above the specified
    threshold. Note that a street corner angle corresponds to 180 degrees minus what is referred to
    as the turn angle in this module. See FIGURE 3 of ITU-R P.1411-12 for a representation of the
    street corner angle.

    Args:
        radioPath (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath
            object obtained from the GetStreetCanyonsRadioPaths() function.
        turnAngleThreshold_deg (float): Turn angle threshold used to identify street corners
            (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to
            zero means the path is going straight through the intersection. A turn angle value
            around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle
            will vary depending on the angle at which the streets are crossing).
            
    Returns:
        (list[float]): List of street corner angles, in degrees, found along the specified street
            canyons radio path.
    """
    if len(radioPath.distances_m) == 0:
        return []
    street_corner_angles_deg = []
    for i in range(0, len(radioPath.turnAngles_deg)):
        if radioPath.turnAngles_deg[i] >= turnAngleThreshold_deg:
            street_corner_angles_deg.append(180.0-radioPath.turnAngles_deg[i])
    return street_corner_angles_deg


def SortRadioPathsByStreetCornerCount(radioPaths: list[StreetCanyonsRadioPath], simplifyPaths: bool,
                                      turnAngleThreshold_deg: float=20, tolerance_m: float=12
                                      ) -> list[StreetCanyonsRadioPath]:
    """
    Sorts the specified street canyons radio paths ascendingly based on their number of turns at
    street corners. Radio paths having the same amount of turns at street corners are secondarily
    sorted ascendingly in order of travel distance.

    Args:
        radioPaths (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): List of
            StreetCanyonsRadioPath objects obtained from the GetStreetCanyonsRadioPaths() function.
        simplifyPaths (bool): Indicates whether to simplify the radio paths before sorting and
            returning them. This is ususally desirable as it removes very small street sections
            that unnecessarily complexifies the path propagation-wise.
        turnAngleThreshold_deg (float): Turn angle threshold used to identify street corners
            (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to
            zero means the path is going straight through the intersection. A turn angle value
            around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle
            will vary depending on the angle at which the streets are crossing).
        tolerance_m (float): Maximum allowed path geometry displacement (meters) when simplifying
            paths.
            
    Returns:
        (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): Sorted list of possibly simplified
            street canyons radio paths.

    """
    simplified_radio_paths = []
    if simplifyPaths == True:
        for radio_path in radioPaths:
            simplified_radio_paths.append(SimplifyRadioPath(radioPath=radio_path, tolerance_m=tolerance_m))
        radioPaths = simplified_radio_paths

    sorted_radio_paths = sorted(radioPaths, key=lambda radio_path: (len(DistancesToStreetCorners(radio_path, turnAngleThreshold_deg)),
                                                                    sum(DistancesToStreetCorners(radio_path, turnAngleThreshold_deg))))
    return sorted_radio_paths


def ReceiverStreetOrientation(graph: Union[StreetGraph,None], txLat: float, txLon: float,
                              rxLat: float, rxLon: float) -> float:
    """
    Gets the receiver' street orientation with respect to the direct path between transmitter and
    receiver (degrees), from 0 to 90 degrees inclusively.

    Args:
        graph (crc_covlib.helper.streets.StreetGraph|None): A StreetGraph object obtained from
            GetStreetGraphFromBox() or GetStreetGraphFromPoints(). When set to None, a new graph
            object is produced internally based on the receiver location.
        txLat (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
        txLon (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
        rxLat (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
        rxLon (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.

    Returns:
        (float): The receiver' street orientation with respect to the direct path between
            transmitter and receiver (degrees), from 0 to 90 degrees inclusively.
    """
    if graph is None:
        graph = GetStreetGraphFromPoints(latLonPoints=[(rxLat, rxLon)])

    G_utm: nx.MultiDiGraph = graph._OSMGraph
    crs_utm = graph._OSMGraph.graph['crs']
    tx_utm: shp.Point = ox.projection.project_geometry(shp.Point(txLon, txLat), crs='epsg:4326',
                                                       to_crs=crs_utm, to_latlong=False)[0]
    rx_utm: shp.Point = ox.projection.project_geometry(shp.Point(rxLon, rxLat), crs='epsg:4326',
                                                       to_crs=crs_utm, to_latlong=False)[0]

    rx_edge: tuple[float, float, float] # (nodeID, nodeID, 0)
    rx_edge = ox.distance.nearest_edges(G=G_utm, X=rx_utm.x, Y=rx_utm.y, return_dist=False)
    snapped_rx_utm: shp.Point = _NearestEdgePoint(graph=graph, point=rx_utm, edge=rx_edge)

    ls: shp.LineString = _EdgeLineString(graph=graph, edge=rx_edge)
    ls1, ls2 = _SplitLineString(linestring=ls, splitterPoint=snapped_rx_utm)

    if ls1 is not None:
        ls1_coords = list(ls1.coords)
        street_point = ls1_coords[-2]
        dist_m = ox.distance.euclidean(snapped_rx_utm.y, snapped_rx_utm.x, street_point[1], street_point[0])
        if dist_m == 0:
            street_point = None
    if street_point is None and ls2 is not None:
        ls2_coords = list(ls2.coords)
        street_point = ls2_coords[1]
        dist_m = ox.distance.euclidean(snapped_rx_utm.y, snapped_rx_utm.x, street_point[1], street_point[0])
        if dist_m == 0:
            raise RuntimeError('Failed to calculate rx street orientation.')

    rx_street_orientation_deg = _TurnAngleDeg(x1=tx_utm.x, y1=tx_utm.y,
                                              x2=snapped_rx_utm.x, y2=snapped_rx_utm.y,
                                              x3=street_point[0], y3=street_point[1])
    if rx_street_orientation_deg > 90:
        rx_street_orientation_deg = 180 - rx_street_orientation_deg

    return rx_street_orientation_deg


def _EnsureEdgeIsFirstEdgeOfPath(path: list[int], edge: tuple[float, float, float]) -> list[int]:
    if len(path)==1:
        node = path[0]
        if node==edge[0]:
            path.insert(0, edge[1])
        elif node==edge[1]:
            path.insert(0, edge[0])
        else:
            raise RuntimeError('Encountered unexpected node id.')
    elif len(path) >= 2:
        first_node = path[0]
        second_node = path[1]
        if (first_node==edge[0] and second_node==edge[1]) or (first_node==edge[1] and second_node==edge[0]):
            pass # edge is already first edge of path
        else:
            if first_node==edge[0]:
                path.insert(0, edge[1])
            elif first_node==edge[1]:
                path.insert(0, edge[0])
            else:
                raise RuntimeError('Encountered unexpected node id.')
    else:
        raise RuntimeError('Encountered unexpected path length.')
    return path


def _EnsureEdgeIsLastEdgeOfPath(path: list[int], edge: tuple[float, float, float]) -> list[int]:
    if len(path)==1:
        node = path[0]
        if node==edge[0]:
            path.append(edge[1])
        elif node==edge[1]:
            path.append(edge[0])
        else:
            raise RuntimeError('Encountered unexpected node id.')
    elif len(path) >= 2:
        last_node = path[-1]
        second_last_node = path[-2]
        if (last_node==edge[0] and second_last_node==edge[1]) or (last_node==edge[1] and second_last_node==edge[0]):
            pass # edge is already last edge of path
        else:
            if last_node==edge[0]:
                path.append(edge[1])
            elif last_node==edge[1]:
                path.append(edge[0])
            else:
                raise RuntimeError('Encountered unexpected node id.')
    else:
        raise RuntimeError('Encountered unexpected path length.')
    return path


def _NearestEdgeNode(graph: StreetGraph, point: shp.Point, edge: tuple[float, float, float]) -> int:
    pt_node1 = _NodePoint(graph=graph, node=edge[0])
    pt_node2 = _NodePoint(graph=graph, node=edge[1])
    dist1_m = ox.distance.euclidean(point.y, point.x, pt_node1.y, pt_node1.x)
    dist2_m = ox.distance.euclidean(point.y, point.x, pt_node2.y, pt_node2.x)
    if dist1_m < dist2_m:
        return edge[0]
    else:
        return edge[1]


def _NearestEdgePoint(graph: StreetGraph, point: shp.Point, edge: tuple[float, float, float]) -> shp.Point:
    ls: shp.LineString = _EdgeLineString(graph=graph, edge=edge)
    pt = shp.ops.nearest_points(ls, point)[0]
    return pt


def _NodePoint(graph: StreetGraph, node: int) -> shp.Point:
    query_result = graph._nodes.query('osmid=={}'.format(node))
    coords = query_result.iloc[0, [graph._nodes.columns.get_loc('x'), graph._nodes.columns.get_loc('y')]]
    pt = shp.Point(coords.x, coords.y)
    return pt


def _EdgeLineString(graph: StreetGraph, edge: tuple[float, float, float]) -> shp.LineString:
    query_result = graph._edges.query('(u=={} & v=={}) | (u=={} & v=={})'.format(edge[0], edge[1],
                                     edge[1], edge[0]))
    ls = query_result.iloc[0, graph._edges.columns.get_loc('geometry')]
    from_node = query_result.iloc[0, graph._edges.columns.get_loc('from')]
    if from_node != edge[0]:
        ls = ls.reverse()
    return ls


def _LineStringDistances(linestring: shp.LineString) -> list[float]:
    distances = []
    for i in range(1, len(linestring.coords)):
        prev_coord = linestring.coords[i-1]
        coord = linestring.coords[i]
        dist = ox.distance.euclidean(y1=prev_coord[1], x1=prev_coord[0], y2=coord[1], x2=coord[0])
        distances.append(dist)
    return distances


def _LineStringTurnAnglesDeg(linestring: shp.LineString) -> list[float]:
    angles_deg = []
    if len(linestring.coords) < 3:
        return angles_deg
    for i in range(1, len(linestring.coords)-1):
        prev_coord = linestring.coords[i-1]
        coord = linestring.coords[i]
        next_coord = linestring.coords[i+1]
        angle = _TurnAngleDeg(x1=prev_coord[0], y1=prev_coord[1], x2=coord[0], y2=coord[1],
                              x3=next_coord[0], y3=next_coord[1])
        angles_deg.append(angle)
    return angles_deg


def _TurnAngleDeg(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    # return value from 0 (going straight forward) to 180 (making a U-turn)
    bearing1_deg = degrees(atan2(x2-x1, y1-y2))
    bearing2_deg = degrees(atan2(x3-x2, y2-y3))
    angle_deg = abs(bearing1_deg-bearing2_deg)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg


def _MergeLineStrings(ls1: shp.LineString, ls2: shp.LineString) -> shp.LineString:
    multi_line = shp.MultiLineString([ls1, ls2])
    # if end points do not corresponds, line_merge returns a MultiLineString instead
    # of a LineString otherwise
    merged_line = shp.line_merge(line=multi_line, directed=True)
    if shp.get_type_id(merged_line) != 1: # 1 is LineString
        raise RuntimeError('Failed to merge LineStrings.')
    return merged_line


def _SplitLineString(linestring: shp.LineString, splitterPoint: shp.Point) -> tuple[shp.LineString, shp.LineString]:
    linestring = shp.ops.snap(g1=linestring, g2=splitterPoint, tolerance=0.001)
    geometry_collection = shp.ops.split(geom=linestring, splitter=splitterPoint)
    # "If the splitter does not split the geometry, a collection with a single geometry equal to
    # the input geometry is returned."
    if len(geometry_collection.geoms) == 2:
        return (geometry_collection.geoms[0], geometry_collection.geoms[1])
    elif len(geometry_collection.geoms) == 1:
        if shp.Point(linestring.coords[0]) == splitterPoint:
            return (None, geometry_collection.geoms[0])
        elif shp.Point(linestring.coords[-1]) == splitterPoint:
            return (geometry_collection.geoms[0], None)
    RuntimeError('Failed to split LineString.')


def _LineStringToCoordTuples(linestring: shp.LineString, invertxy: bool) -> list[tuple[float,float]]:
    tuple_list = []
    if invertxy == False:
        for coord in list(linestring.coords):
            tuple_list.append((coord[0], coord[1]))
    else:
        for coord in list(linestring.coords):
            tuple_list.append((coord[1], coord[0]))
    return tuple_list


def _IsIntersection(radioPath: StreetCanyonsRadioPath, xProjectedCoord: float, yProjectedCoord) -> bool:
    for i in range(0, len(radioPath._projectedCoordsPath)):
        x = radioPath._projectedCoordsPath[i][0]
        y = radioPath._projectedCoordsPath[i][1]
        if x == xProjectedCoord and y == yProjectedCoord:
            return radioPath.isIntersection[i-1]
    raise RuntimeError('Failed to find point.')


_osmnxMajorVer = int(ox.__version__.split('.')[0])
