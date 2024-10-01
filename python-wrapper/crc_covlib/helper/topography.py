"""Additional topography related functionalities in support of crc-covlib.
"""

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio import warp
import ctypes
import math
from typing import Union
from ..simulation import Simulation, TerrainElevDataSource, LandCoverDataSource, SurfaceElevDataSource
from . import itur_p2001
from numba import jit


__all__ = ['CustomDataInfo',
           'LatLonBox',
           'LoadFileAsCustomTerrainElevData',
           'LoadFileAsCustomLandCoverData',
           'LoadFileAsCustomSurfaceElevData',
           'GetDistanceProfile',
           'GetLatLonProfile',
           'GetTerrainElevProfile',
           'GetSurfaceElevProfile',
           'GetLandCoverProfile',
           'GetMappedLandCoverProfile',
           'GetRasterProfile']


class CustomDataInfo:
    """
    Holds metadata for crc-covlib's custom terrain elevation data, land cover data or surface
    elevation data. Geographical coordinates in WGS 84 (EPSG 4326).

    Attributes:
        lowerLeftCornerLat_degrees (float): lower left corner latitude (i.e. minimum latitude)
            of the custom data (degrees).
        lowerLeftCornerLon_degrees (float): lower left corner longitude (i.e. minimum longitude)
            of the custom data (degrees).
        upperRightCornerLat_degrees (float): upper right corner latitude (i.e. maximum latitude)
            of the custom data (degrees).
        upperRightCornerLon_degrees (float): upper right corner longitude (i.e. maximum longitude)
            of the custom data (degrees).
        numHorizSamples (int): number of horizontal samples, or width, of the custom data.
        numVertSamples (int): number of vertical samples, or height, of the custom data.
        isNoDataValueDefined (bool): indicates whether a specific value within the custom data
            should be interpreted as "no data".
        noDataValue (int|float): value to interpret as "no data", irrelevant when 
            isNoDataValueDefined is set to False.
    """
    def __init__(self):
        self.lowerLeftCornerLat_degrees:float = 0
        self.lowerLeftCornerLon_degrees:float = 0
        self.upperRightCornerLat_degrees:float = 0
        self.upperRightCornerLon_degrees:float = 0
        self.numHorizSamples:int = 0
        self.numVertSamples:int = 0
        self.isNoDataValueDefined:bool = False
        self.noDataValue:Union[int,float] = 0


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


def LoadFileAsCustomTerrainElevData(sim: Simulation, pathname: str, bounds: Union[LatLonBox,None]=None,
                                    setAsPrimary: bool=True, band: int=1) -> CustomDataInfo:
    """
    Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326)
    geographic coordinate system if required, and passes the content as custom terrain elevation
    data to the crc-covlib Simulation object (calling AddCustomTerrainElevData).

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        pathname (str): Absolute or relative path to a georeferenced raster file containing
            terrain elevation data in meters.
        bounds (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered
            by the specified bounds is read. Otherwise the whole content of the file is used.
        setAsPrimary (bool): If set to True, the sim object's primary terrain elevation data source
            is set to TERR_ELEV_CUSTOM.
        band (int): Band to use for the raster file, indexed from 1.

    Returns:
        crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object
            containing the argument values used when calling AddCustomTerrainElevData().
            Returns None on failure.
    """
    if bounds is None:
        cd = _GetEpsg4326Data(pathname, band, 'single')
    else:
        intersect_bounds = _IntersectingBounds(pathname, bounds)
        if intersect_bounds is not None:
            cd = _GetEpsg4326SubData(pathname, intersect_bounds, band, 'single', -32768)
        else:
            return None
    if cd.data is None or cd.info.numHorizSamples < 2 or cd.info.numVertSamples < 2:
        return None
    cd.data = np.require(cd.data, dtype=np.single, requirements=['C'])
    success = sim._lib.AddCustomTerrainElevData(
        sim._sim_ptr,
        cd.info.lowerLeftCornerLat_degrees, cd.info.lowerLeftCornerLon_degrees,
        cd.info.upperRightCornerLat_degrees, cd.info.upperRightCornerLon_degrees,
        cd.info.numHorizSamples, cd.info.numVertSamples,
        cd.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cd.info.isNoDataValueDefined, cd.info.noDataValue)
    if setAsPrimary == True:
        sim.SetPrimaryTerrainElevDataSource(TerrainElevDataSource.TERR_ELEV_CUSTOM)
    if success == False:
        return None
    return cd.info


def LoadFileAsCustomLandCoverData(sim: Simulation, pathname: str, bounds: Union[LatLonBox,None]=None,
                                  setAsPrimary: bool=True, band: int=1) -> CustomDataInfo:
    """
    Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326)
    geographic coordinate system if required, and passes the content as custom land cover data
    to the crc-covlib Simulation object (calling AddCustomLandCoverData).

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        pathname (str): Absolute or relative path to a georeferenced raster file containing
            land cover (clutter) data.
        bounds (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered
            by the specified bounds is read. Otherwise the whole content of the file is used.
        setAsPrimary (bool): If set to True, the sim object's primary land cover data source
            is set to LAND_COVER_CUSTOM.
        band (int): Band to use for the raster file, indexed from 1.

    Returns:
        crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object
            containing the argument values used when calling AddCustomLandCoverData().
            Returns None on failure.
    """
    if bounds is None:
        cd = _GetEpsg4326Data(pathname, band, 'short')
    else:
        intersect_bounds = _IntersectingBounds(pathname, bounds)
        if intersect_bounds is not None:
            cd = _GetEpsg4326SubData(pathname, intersect_bounds, band, 'short', -32768)
        else:
            return None
    if cd.data is None or cd.info.numHorizSamples < 2 or cd.info.numVertSamples < 2:
        return None
    cd.data = np.require(cd.data, dtype=np.short, requirements=['C'])
    success = sim._lib.AddCustomLandCoverData(
        sim._sim_ptr,
        cd.info.lowerLeftCornerLat_degrees, cd.info.lowerLeftCornerLon_degrees,
        cd.info.upperRightCornerLat_degrees, cd.info.upperRightCornerLon_degrees,
        cd.info.numHorizSamples, cd.info.numVertSamples,
        cd.data.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
        cd.info.isNoDataValueDefined, int(cd.info.noDataValue))
    if setAsPrimary == True:
        sim.SetPrimaryLandCoverDataSource(LandCoverDataSource.LAND_COVER_CUSTOM)
    if success == False:
        return None
    return cd.info


def LoadFileAsCustomSurfaceElevData(sim: Simulation, pathname: str, bounds: Union[LatLonBox,None]=None,
                                    setAsPrimary: bool=True, band: int=1) -> CustomDataInfo:
    """
    Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326)
    geographic coordinate system if required, and passes the content as custom surface elevation
    data to the crc-covlib Simulation object (calling AddCustomSurfaceElevData).

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        pathname (str): Absolute or relative path to a georeferenced raster file containing
            surface elevation data in meters.
        bounds (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered
            by the specified bounds is read. Otherwise the whole content of the file is used.
        setAsPrimary (bool): If set to True, the sim object's primary surface elevation data source
            is set to SURF_ELEV_CUSTOM.
        band (int): Band to use for the raster file, indexed from 1.

    Returns:
        crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object
            containing the argument values used when calling AddCustomSurfaceElevData().
            Returns None on failure.
    """
    if bounds is None:
        cd = _GetEpsg4326Data(pathname, band, 'single')
    else:
        intersect_bounds = _IntersectingBounds(pathname, bounds)
        if intersect_bounds is not None:
            cd = _GetEpsg4326SubData(pathname, intersect_bounds, band, 'single', -32768)
        else:
            return None
    if cd.data is None or cd.info.numHorizSamples < 2 or cd.info.numVertSamples < 2:
        return None
    cd.data = np.require(cd.data, dtype=np.single, requirements=['C'])
    success = sim._lib.AddCustomSurfaceElevData(
        sim._sim_ptr,
        cd.info.lowerLeftCornerLat_degrees, cd.info.lowerLeftCornerLon_degrees,
        cd.info.upperRightCornerLat_degrees, cd.info.upperRightCornerLon_degrees,
        cd.info.numHorizSamples, cd.info.numVertSamples,
        cd.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cd.info.isNoDataValueDefined, cd.info.noDataValue)
    if setAsPrimary == True:
        sim.SetPrimarySurfaceElevDataSource(SurfaceElevDataSource.SURF_ELEV_CUSTOM)
    if success == False:
        return None
    return cd.info


class _CustomData:
    def __init__(self):
        self.info = CustomDataInfo()
        self.data:np.ndarray = None


def _GetEpsg4326Data(pathname: str, band: int=1, outDataType: str=None) -> _CustomData:
    """
    Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326)
    geographic coordinate system if required, and returns the content in a numpy array.

    Args:
        pathname (str): Absolute or relative path to a georeferenced raster file.
        band (int): Band to use from the raster file, indexed from 1.
        outDataType (str): Numpy data type to use for the output numpy array, uses the file's
            data type if none specified.

    Returns:
        crc_covlib.helper.topography._CustomData: object containing the numpy array and other
            related information.
    """
    custom_data = _CustomData()

    with rio.open(pathname) as src:
        auth = src.crs.to_authority()
        if auth == ('EPSG', '4326') or auth == ('OGC', 'CRS84'):
            custom_data.data = np.flipud(src.read(indexes=band, out_dtype=outDataType)) # note: use np.fliplr() instead if calling read() without the band no
            dst = src
        else:
            dst_crs = 'EPSG:4326'
            dst_transform, dst_width, dst_height = rio.warp.calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})
            with rio.io.MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst: # dst is of class rasterio.io.DatasetReader
                    rio.warp.reproject(source=rio.band(src, band),
                                       destination=rio.band(dst, 1),
                                       src_transform=src.transform,
                                       src_crs=src.crs,
                                       dst_transform=dst_transform,
                                       dst_crs=dst_crs,
                                       resampling=rio.warp.Resampling.nearest)
                    custom_data.data = np.flipud(dst.read(indexes=1, out_dtype=outDataType)) # note: use np.fliplr() instead if calling read() without the band no

        pixel_width = dst.res[0]
        pixel_height =  dst.res[1]
        custom_data.info.lowerLeftCornerLat_degrees = dst.bounds.bottom + (pixel_height/2)
        custom_data.info.lowerLeftCornerLon_degrees = dst.bounds.left + (pixel_width/2)
        custom_data.info.upperRightCornerLat_degrees = dst.bounds.top - (pixel_height/2)
        custom_data.info.upperRightCornerLon_degrees = dst.bounds.right - (pixel_width/2)
        custom_data.info.numHorizSamples = dst.width
        custom_data.info.numVertSamples = dst.height
        if dst.nodatavals[0] is None:
            custom_data.info.isNoDataValueDefined = False
        else:
            custom_data.info.isNoDataValueDefined = True
            custom_data.info.noDataValue = dst.nodatavals[0]

    return custom_data


def _GetEpsg4326SubData(pathname: str, bounds: LatLonBox, band: int=1, outDataType: str=None, noDataValue: Union[int,float,None]=None) -> _CustomData:
    """
    Reads part of the specified raster data file, reprojects it (in memory) to the WGS84
    (EPSG 4326) geographic coordinate system if required, and returns the content in a numpy
    array.

    Args:
        pathname (str): Absolute or relative path to a georeferenced raster file.
        bounds (crc_covlib.helper.topography.LatLonBox): Only the area covered by the specified
            bounds is read.
        band (int): Band to use from the raster file, indexed from 1.
        outDataType (str): Numpy data type to use for the output numpy array, uses the file's
            data type if none specified.
        noDataValue (int|float|None): By default the file's "no data" value is also used for
            representing "no data" in the output numpy array. However, if the file does not
            contain any "no data" value, noDataValue will be used if specified.

    Returns:
        crc_covlib.helper.topography._CustomData: object containing the numpy array and other
            related information.
    """
    custom_data = _CustomData()

    with rio.open(pathname) as src:
        nodata = src.nodatavals[band-1]
        if nodata is None:
            nodata = noDataValue
        auth = src.crs.to_authority()
        if auth == ('EPSG', '4326') or auth == ('OGC', 'CRS84'):
            # get row/colum offsets and width/height of the data within the file that cover the specified boundaries
            win = rio.windows.from_bounds(left=bounds.minLon, bottom=bounds.minLat, right=bounds.maxLon,
                                          top=bounds.maxLat, transform=src.transform)
            # add a little bit extra to those values to make sure we won't miss any data at the edge of the specified boundaries
            win = rio.windows.Window(col_off=math.floor(win.col_off)-1, row_off=math.floor(win.row_off)-1,
                                     height=math.ceil(win.height)+2, width=math.ceil(win.width)+2)
            # get the new boundaries (in ESPG 4326) corresponding to those new file offets and width/height
            dst_left, dst_bottom, dst_right, dst_top = rio.windows.bounds(win, src.transform)
            custom_data.data = np.flipud(src.read(indexes=band, window=win, boundless=True, out_dtype=outDataType, fill_value=nodata))
            pixel_width = src.res[0]
            pixel_height = src.res[1]
        else:
            dst_crs = 'EPSG:4326'
            # convert bounds from EPSG 4326 to the source file's CRS
            left, bottom, right, top = rio.warp.transform_bounds(src_crs=dst_crs, dst_crs=src.crs, 
                                                                 bottom=bounds.minLat, left=bounds.minLon,
                                                                 top=bounds.maxLat, right=bounds.maxLon)
            # get row/colum offsets and width/height of the data within the file that cover those boundaries
            win = rio.windows.from_bounds(left=left, bottom=bottom, right=right, top=top, transform=src.transform)
            # add a little bit extra to those values to make sure we won't miss any data at the edge of the specified boundaries
            src_win = rio.windows.Window(col_off=math.floor(win.col_off)-1, row_off=math.floor(win.row_off)-1,
                             height=math.ceil(win.height)+2, width=math.ceil(win.width)+2)
            # get the new boundaries (in the source file's CRS) corresponding to those new file offets and width/height
            src_left, src_bottom, src_right, src_top = rio.windows.bounds(src_win, src.transform)

            src_win_transform = rio.windows.transform(window=src_win, transform=src.transform)
            src_data = src.read(indexes=band, window=src_win, boundless=True, out_dtype=outDataType, fill_value=nodata)
            dst_transform, dst_width, dst_height = rio.warp.calculate_default_transform(src_crs=src.crs, dst_crs=dst_crs, 
                                                       width=src_data.shape[1], height=src_data.shape[0],
                                                       left=src_left, bottom=src_bottom, right=src_right, top=src_top)

            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height,
                           'dtype': src_data.dtype, 'nodata': nodata})
            with rio.io.MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    rio.warp.reproject(source=src_data,
                                       destination=rio.band(dst, 1),
                                       src_transform=src_win_transform,
                                       src_crs=src.crs,
                                       dst_transform=dst_transform,
                                       dst_crs=dst_crs,
                                       resampling=rio.warp.Resampling.nearest,
                                       src_nodata=nodata,
                                       dst_nodata=nodata)
                    #custom_data.data = np.flipud(dst.read(indexes=1, boundless=True))

                    # At this point, the reprojected data covers the specified area (bounds in EPSG 4326) but also
                    # potentially additional data depending on the source file's projection, since its projection 
                    # may not perfectly align with latitude/longitude lines.
                    # To minimize the amount of data we may continue by selecting data from the requested area only.

                with memfile.open() as reproj_dst:
                    win = rio.windows.from_bounds(left=bounds.minLon, bottom=bounds.minLat, right=bounds.maxLon,
                                                  top=bounds.maxLat, transform=reproj_dst.transform)
                    win = rio.windows.Window(col_off=math.floor(win.col_off)-1, row_off=math.floor(win.row_off)-1,
                                             height=math.ceil(win.height)+2, width=math.ceil(win.width)+2)
                    dst_left, dst_bottom, dst_right, dst_top = rio.windows.bounds(win, reproj_dst.transform)
                    custom_data.data = np.flipud(reproj_dst.read(indexes=1, window=win, boundless=True))
                    pixel_width = reproj_dst.res[0]
                    pixel_height = reproj_dst.res[1]

        custom_data.info.numHorizSamples = custom_data.data.shape[1] # or win.width
        custom_data.info.numVertSamples = custom_data.data.shape[0]  # or win.height
        custom_data.info.lowerLeftCornerLat_degrees = dst_bottom + (pixel_height/2)
        custom_data.info.lowerLeftCornerLon_degrees = dst_left + (pixel_width/2)
        custom_data.info.upperRightCornerLat_degrees = dst_top - (pixel_height/2)
        custom_data.info.upperRightCornerLon_degrees = dst_right - (pixel_width/2)

        if nodata is None:
            custom_data.info.isNoDataValueDefined = False
        else:
            custom_data.info.isNoDataValueDefined = True
            custom_data.info.noDataValue = nodata

    return custom_data


def _IntersectingBounds(pathname: str, bounds: LatLonBox) -> Union[LatLonBox,None]:
    """
    Gets the intersection between pathname's bounds and the specified bounds.

    Args:
        pathname (str): Absolute or relative path to a georeferenced raster file.
        bounds (crc_covlib.helper.topography.LatLonBox): A lat/lon box for the intersection
            calculation.

    Returns:
        crc_covlib.helper.topography.LatLonBox | None: The intersection between pathname's
            bounding lat/lon box and the specified lat/lon box. Returns None if there is no
            intersecting area between the two.
    """
    with rio.open(pathname) as src:
        auth = src.crs.to_authority()
        if auth == ('EPSG', '4326') or auth == ('OGC', 'CRS84'):
            min_lat = src.bounds.bottom
            max_lat = src.bounds.top
            min_lon = src.bounds.left
            max_lon = src.bounds.right
        else:
            min_lon, min_lat, max_lon, max_lat = rio.warp.transform_bounds(src_crs=src.crs, dst_crs='EPSG:4326', 
                                                                 bottom=src.bounds.bottom, left=src.bounds.left,
                                                                 top=src.bounds.top, right=src.bounds.right)
        if min_lat >= bounds.maxLat: return None
        if max_lat <= bounds.minLat: return None
        if min_lon >= bounds.maxLon: return None
        if max_lon <= bounds.minLon: return None
        new_bounds = LatLonBox(minLat=max(bounds.minLat, min_lat),
                                minLon=max(bounds.minLon, min_lon),
                                maxLat=min(bounds.maxLat, max_lat),
                                maxLon=min(bounds.maxLon, max_lon))
        return new_bounds


# Profile functions

@jit(nopython=True)
def GetDistanceProfile(lat0: float, lon0: float, lat1: float, lon1: float, res_m: float,
                       minNumPoints: int=3, maxNumPoints: Union[int,None]=None
                       ) -> npt.NDArray[np.float64]:
    """
    Gets a great-circle distance profile (km) of evenly spaced points between the two specified
    points on the surface of the Earth.

    Note: Calling GetDistanceProfile() and GetLatLonProfile() with the same input parameters yields
    profiles for the same corresponding points.

    Args:
        lat0 (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
        lon0 (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
        lat1 (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
        lon1 (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.
        res_m (float): Resolution (meters). Indicates that the returned profile points must be
            approximately spaced by res_m meters.
        minNumPoints (float): Minimum number of points (i.e. distances) that the returned profile
            must contain.
        maxNumPoints (int|None): Maximum number of points (i.e. distances) allowed in the returned
            profile. No limit when set to None.

    Returns:
        distProfile (numpy.typing.NDArray[numpy.float64]): A great-circle distance profile (km) of
            evenly spaced points between the two specified points on the surface of the Earth.
    """
    path_lengh_km = itur_p2001.PathLength(lat0, lon0, lat1, lon1)
    num_points = 1 + int((path_lengh_km*1000/res_m) + 1) # same rounding as in the C++ version
    num_points = max(minNumPoints, num_points)
    if maxNumPoints is not None:
        num_points = min(maxNumPoints, num_points)
    dist_km_profile = np.linspace(0.0, path_lengh_km, num_points)
    return dist_km_profile


@jit(nopython=True)
def GetLatLonProfile(lat0: float, lon0: float, lat1: float, lon1: float, res_m: float,
                     minNumPoints: int=3, maxNumPoints: Union[int,None]=None) -> npt.NDArray[np.float64]:
    """
    Gets a latitude/longitude profile (degrees) of evenly spaced points (using a great-circle
    distance algorithm) between the two specified points on the surface of the Earth.

    Note: Calling GetDistanceProfile() and GetLatLonProfile() with the same input parameters yields
    profiles for the same corresponding points.

    Args:
        lat0 (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
        lon0 (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
        lat1 (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
        lon1 (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.
        res_m (float): Resolution (meters). Indicates that the returned profile points must be
            approximately spaced by res_m meters.
        minNumPoints (float): Minimum number of points (i.e. distances) that the returned profile
            must contain.
        maxNumPoints (int|None): Maximum number of points (i.e. distances) allowed in the returned
            profile. No limit when set to None.

    Returns:
        latLonProfile (numpy.typing.NDArray[numpy.float64]): A latitude/longitude profile (degrees)
            of evenly spaced points (using a great-circle distance algorithm) between the two
            specified points on the surface of the Earth. The returned profile is a 2D array of
            shape (numPoints, 2). The latitude of the first point is at latLonProfile[0][0] and its
            longitude is at latLonProfile[0][1].
    """
    path_lengh_km = itur_p2001.PathLength(lat0, lon0, lat1, lon1)
    num_points = 1 + int((path_lengh_km*1000/res_m) + 1) # same rounding as in the C++ version
    num_points = max(minNumPoints, num_points)
    if maxNumPoints is not None:
        num_points = min(maxNumPoints, num_points)
    delta_dist_km = path_lengh_km/(num_points-1)
    lat_lon_profile = np.zeros((num_points, 2))
    for i in range(1, num_points-1):
        lat, lon = itur_p2001.IntermediatePathPoint(lat0, lon0, lat1, lon1, i*delta_dist_km)
        lat_lon_profile[i][0] = lat
        lat_lon_profile[i][1] = lon

    lat_lon_profile[0][0] = lat0
    lat_lon_profile[0][1] = lon0
    lat_lon_profile[-1][0] = lat1
    lat_lon_profile[-1][1] = lon1
    
    return lat_lon_profile


def GetTerrainElevProfile(sim: Simulation, latLonProfile: npt.ArrayLike,
                          noDataValue: float=0) -> tuple[npt.NDArray[np.float64], bool]:
    """
    Gets a terrain elevation profile (meters) using the terrain elevation source(s) from the
    specified Simulation object.

    Args:
        sim (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (float): Terrain elevation (m) value to be used in the returned profile when no
            terrain elevation data can be retrieved at a specific location.

    Returns:
        terrainElevProfile (numpy.typing.NDArray[numpy.float64]): The terrain elevation profile
            (meters) for the points specified in latLonProfile.
        status (bool): True when all terrain elevation data could be successfully retrieved. False
            when the terrain elevation data could not be retrieved for at least one of the locations
            in latLonProfile.
    """
    no_missing_data = True
    nan = float('nan')
    num_points = len(latLonProfile)
    terr_elev_profile = np.zeros(num_points, dtype=np.float64)
    for i in range(0, num_points):
        terr_elev = sim.GetTerrainElevation(latLonProfile[i][0], latLonProfile[i][1], nan)
        if math.isnan(terr_elev):
            no_missing_data = False
            terr_elev_profile[i] = noDataValue
        else:
            terr_elev_profile[i] = terr_elev
    return (terr_elev_profile, no_missing_data)


def GetSurfaceElevProfile(sim: Simulation, latLonProfile: npt.ArrayLike,
                          noDataValue: float=0) -> tuple[npt.NDArray[np.float64], bool]:
    """
    Gets a surface elevation profile (meters) using the surface elevation source(s) from the
    specified Simulation object.

    Args:
        sim (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (float): Surface elevation (m) value to be used in the returned profile when no
            surface elevation data can be retrieved at a specific location.

    Returns:
        surfaceElevProfile (numpy.typing.NDArray[numpy.float64]): The surface elevation profile
            (meters) for the points specified in latLonProfile.
        status (bool): True when all surface elevation data could be successfully retrieved. False
            when the surface elevation data could not be retrieved for at least one of the locations
            in latLonProfile.
    """
    no_missing_data = True
    nan = float('nan')
    num_points = len(latLonProfile)
    surf_elev_profile = np.zeros(num_points, dtype=np.float64)
    for i in range(0, num_points):
        surf_elev = sim.GetSurfaceElevation(latLonProfile[i][0], latLonProfile[i][1], nan)
        if math.isnan(surf_elev):
            no_missing_data = False
            surf_elev_profile[i] = noDataValue
        else:
            surf_elev_profile[i] = surf_elev
    return (surf_elev_profile, no_missing_data)


def GetLandCoverProfile(sim: Simulation, latLonProfile: npt.ArrayLike,
                        noDataValue: int=-1) -> tuple[npt.NDArray[np.intc], bool]:
    """
    Gets a land cover profile using the land cover source(s) from the specified Simulation object.

    Args:
        sim (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (int): Value to be used in the returned profile when no land cover data can
            be retrieved at a specific location.

    Returns:
        landCoverProfile (numpy.typing.NDArray[numpy.intc]): The land cover profile for the points
            specified in latLonProfile.
        status (bool): True when all land cover data could be successfully retrieved. False when
            the land cover data could not be retrieved for at least one of the locations in
            latLonProfile.
    """
    no_missing_data = True
    num_points = len(latLonProfile)
    land_cover_profile = np.zeros(num_points, dtype=np.intc)
    for i in range(0, num_points):
        land_cover_class = sim.GetLandCoverClass(latLonProfile[i][0], latLonProfile[i][1])
        if land_cover_class == -1:
            no_missing_data = False
            land_cover_profile[i] = noDataValue
        else:
            land_cover_profile[i] = land_cover_class
    return (land_cover_profile, no_missing_data)


def GetMappedLandCoverProfile(sim: Simulation, latLonProfile: npt.ArrayLike,
                              noDataValue: int=-1) -> tuple[npt.NDArray[np.intc], bool]:
    """
    Gets a mapped land cover profile using the land cover source(s) and mappings from the specified
    Simulation object. A mapped land cover value is a land cover value that has been converted to a
    recognized and usable value by the currently selected propagation model in the Simulation
    object.

    Args:
        sim (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (int): Value to be used in the returned profile when no land cover data or
            mapping can be retrieved.

    Returns:
        mappedLandCoverProfile (numpy.typing.NDArray[numpy.intc]): The mapped land cover profile
            for the points specified in latLonProfile.
        status (bool): True when all land cover data could be successfully retrieved and mapped.
            False when the land cover data could not be retrieved or mapped for at least one of the
            locations in latLonProfile.
    """
    no_missing_data = True
    propag_model_id = sim.GetPropagationModel()
    num_points = len(latLonProfile)
    mapped_land_cover_profile = np.zeros(num_points, dtype=np.intc)
    for i in range(0, num_points):
        mapped_land_cover_class = sim.GetLandCoverClassMappedValue(latLonProfile[i][0],
                                      latLonProfile[i][1], propag_model_id)
        if mapped_land_cover_class == -1:
            no_missing_data = False
            mapped_land_cover_profile[i] = noDataValue
        else:
            mapped_land_cover_profile[i] = mapped_land_cover_class
    return (mapped_land_cover_profile, no_missing_data)


def GetRasterProfile(pathnames: Union[str, list[str]], latLonProfile: npt.ArrayLike,
                     noDataValue: float=0, band:int=1) -> tuple[list[float], bool]:
    """
    Gets a profile of pixel values from the specified georeferenced raster file(s). This function
    may be used to either get a terrain elevation, a surface elevation or a land cover profile
    depending on the actual content of the raster file(s). Pixel values are obtained using a
    nearest neighbor algorithm.

    Please note that this function is provided for convenience and runs much slower than the other
    Get...Profile() functions. It is advisable to use the other functions whenever possible if a 
    large number of profiles needs to be obtained.

    Args:
        pathnames (str|list[str]): Absolute or relative path to a georeferenced raster file, or a
            list of absolute or relative paths to georeferenced raster file(s) (useful when the
            specified latLonProfile extends over more than one file).
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (float): Pixel value to be used in the returned profile when no pixel data can
            be retrieved at a specific location.
        band (int): Band to use for the raster file(s), indexed from 1.

    Returns:
        profile (list[float]): Pixel value profile for the locations specified in latLonProfile.
        status (bool): True when all pixel values could be successfully retrieved, False otherwise.
    """
    if isinstance(pathnames, list) == False:
        return _GetProfileFromRasterFile(pathnames, latLonProfile, noDataValue, band)
    else:
        no_missing_data = False
        num_points = len(latLonProfile)
        merged_profile = [None]*num_points
        for pathname in pathnames:
            current_profile, _ = _GetProfileFromRasterFile(pathname, latLonProfile, None, band)
            for i, value in enumerate(merged_profile):
                if value is None:
                    merged_profile[i] = current_profile[i]
            if None not in merged_profile:
                no_missing_data = True
                break
        if no_missing_data == False:
            for i in range(0, num_points):
                if merged_profile[i] is None:
                    merged_profile[i] = noDataValue
        return (merged_profile, no_missing_data)


def _GetProfileFromRasterFile(pathname: str, latLonProfile: npt.ArrayLike, noDataValue: float=0,
                              band:int=1) -> tuple[list[float], bool]:
    """
    Gets a profile of pixel values from the specified georeferenced raster file. Pixel values are
    obtained using a nearest neighbor algorithm.

    Args:
        pathname (str): Absolute or relative path to a georeferenced raster file.
        latLonProfile (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form
            of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0]
            and its longitude at profile[0][1]. Such a profile may be obtained from the
            GetLatLonProfile() function.
        noDataValue (float): Pixel value to be used in the returned profile when no pixel data can
            be retrieved at a specific location.
        band (int): Band to use for the raster file(s), indexed from 1.

    Returns:
        profile (list[float]): Pixel value profile for the locations specified in latLonProfile.
        status (bool): True when all pixel values could be successfully retrieved, False otherwise.
    """
    no_missing_data = True
    num_points = len(latLonProfile)
    profile = [noDataValue]*num_points
    with rio.open(pathname) as src:
        file_no_data_val = src.nodatavals[0]
        wgs84 = rio.crs.CRS.from_string('EPSG:4326')
        lats = [0]*num_points
        lons = [0]*num_points
        for i in range(0, num_points):
            lats[i] = latLonProfile[i][0]
            lons[i] = latLonProfile[i][1]
        src_file_coords = rio.warp.transform(src_crs=wgs84, dst_crs=src.crs, xs=lons, ys=lats)
        rows, cols = rio.transform.rowcol(transform=src.transform, xs=src_file_coords[0],
                                          ys=src_file_coords[1], op=math.floor)
        for i in range(0, num_points):
            single_pixel_window = rio.windows.Window(cols[i], rows[i], 1, 1)
            value = src.read(indexes=band, window=single_pixel_window)
            if value.size == 1:
                value = value[0][0]
                if value == file_no_data_val:
                    no_missing_data = False
                    value = noDataValue
            else:
                no_missing_data = False
                value = noDataValue
            profile[i] = value
    return (profile, no_missing_data)