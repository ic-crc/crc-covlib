"""
Facilitate downloads of terrain, surface and land cover data from the Canada Centre for Mapping and
Earth Observation Data Cube Platform.

Publisher - Current Organization Name: Natural Resources Canada
Licence: Open Government Licence - Canada (https://open.canada.ca/en/open-government-licence-canada)

Additional links:
https://datacube.services.geo.ca/en/index.html
https://datacube.services.geo.ca/stac/api/
"""
import os
import pyproj
import rasterio
import urllib.request # part of the python standard library


__all__ = ['LatLonBox',
           'DownloadCdem',
           'DownloadCdsm',
           'DownloadMrdemDtm',
           'DownloadMrdemDsm',
           'DownloadHrdemDtm1m',
           'DownloadHrdemDsm1m',
           'DownloadHrdemDtm2m',
           'DownloadHrdemDsm2m',
           'DownloadLandcover2020'
          ]


class LatLonBox:
    """
    Bounding box in geographical coordinates (WGS84 or EPSG 4326 assumed).

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


def DownloadCdem(bounds: LatLonBox, outputPathname: str) -> None:
    """
    Downloads an extract from the Canadian Digital Elevation Model (CDEM).
    NOTE: CDEM is a legacy product, a better option is to use DownloadMrdemDtm() instead.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/cdem-cdsm/cdem/cdem-canada-dem.tif'
    _DownloadFromDatacube(bounds, outputPathname, cogUrl, 30.0, 30.0, False)


def DownloadCdsm(bounds: LatLonBox, outputPathname: str) -> None:
    """
    Downloads an extract from the Canadian Digital Surface Model (CDSM).
    NOTE: CDSM is a legacy product, a better option is to use DownloadMrdemDsm() instead.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/cdem-cdsm/cdsm/cdsm-canada-dem.tif'
    _DownloadFromDatacube(bounds, outputPathname, cogUrl, 30.0, 30.0, False)


def DownloadMrdemDtm(bounds: LatLonBox, outputPathname: str) -> None:
    """
    Downloads an extract from the Medium Resolution Digital Elevation Model (MRDEM) / Digital
    Terrain Model (DTM).

    For more details on this product:
    https://open.canada.ca/data/en/dataset/18752265-bda3-498c-a4ba-9dfe68cb98da

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/elevation/mrdem/mrdem-30/mrdem-30-dtm.tif'
    _DownloadFromDatacube(bounds, outputPathname, cogUrl, 30.0, 30.0, False)


def DownloadMrdemDsm(bounds: LatLonBox, outputPathname: str) -> None:
    """
    Downloads an extract from the Medium Resolution Digital Elevation Model (MRDEM) / Digital
    Surface Model (DSM).

    For more details on this product:
    https://open.canada.ca/data/en/dataset/18752265-bda3-498c-a4ba-9dfe68cb98da

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/elevation/mrdem/mrdem-30/mrdem-30-dsm.tif'
    _DownloadFromDatacube(bounds, outputPathname, cogUrl, 30.0, 30.0, False)


def DownloadHrdemDtm1m(bounds: LatLonBox, outputPathname: str) -> list[str]:
    """
    Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Terrain
    Model (DTM) at 1-meter resolution.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.

    Returns:
        (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain
            one file (i.e. outputPathname). However when the specified bounds overlap more than one
            tile of the HRDEM mosaic product, one file for each overlapping tile will be created.
            Created files may contain some or even only "no data" values depending on the
            availability of data for the requested area.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/hrdem/hrdem-mosaic-1m/___-mosaic-1m-dtm.tif'
    return _DownloadHrdem(bounds, outputPathname, cogUrl, 1.0, 1.0, False)


def DownloadHrdemDsm1m(bounds: LatLonBox, outputPathname: str) -> list[str]:
    """
    Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Surface
    Model (DSM) at 1-meter resolution.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.

    Returns:
        (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain
            one file (i.e. outputPathname). However when the specified bounds overlap more than one
            tile of the HRDEM mosaic product, one file for each overlapping tile will be created.
            Created files may contain some or even only "no data" values depending on the
            availability of data for the requested area.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/hrdem/hrdem-mosaic-1m/___-mosaic-1m-dsm.tif'
    return _DownloadHrdem(bounds, outputPathname, cogUrl, 1.0, 1.0, False)


def DownloadHrdemDtm2m(bounds: LatLonBox, outputPathname: str) -> list[str]:
    """
    Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Terrain
    Model (DTM) at 2-meter resolution.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.

    Returns:
        (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain
            one file (i.e. outputPathname). However when the specified bounds overlap more than one
            tile of the HRDEM mosaic product, one file for each overlapping tile will be created.
            Created files may contain some or even only "no data" values depending on the
            availability of data for the requested area.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/hrdem/hrdem-mosaic-2m/___-mosaic-2m-dtm.tif'
    return _DownloadHrdem(bounds, outputPathname, cogUrl, 2.0, 2.0, False)


def DownloadHrdemDsm2m(bounds: LatLonBox, outputPathname: str) -> list[str]:
    """
    Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Surface
    Model (DSM) at 2-meter resolution.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.

    Returns:
        (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain
            one file (i.e. outputPathname). However when the specified bounds overlap more than one
            tile of the HRDEM mosaic product, one file for each overlapping tile will be created.
            Created files may contain some or even only "no data" values depending on the
            availability of data for the requested area.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/elevation/hrdem/hrdem-mosaic-2m/___-mosaic-2m-dsm.tif'
    return _DownloadHrdem(bounds, outputPathname, cogUrl, 2.0, 2.0, False)


def DownloadLandcover2020(bounds: LatLonBox, outputPathname: str) -> None:
    """
    Downloads an extract from the 2020 Land Cover of Canada map.

    For more details on this product:
    https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47

    Args:
        bounds (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
        outputPathname (str): Destination file for the extract. The file and any missing directory
            will be created if non-existent, otherwise the file will be overwritten. The extract is
            saved in the GeoTIFF (.tif) format.
    """
    cogUrl = 'https://datacube-prod-data-public.s3.amazonaws.com/store/land/landcover/landcover-2020-classification.tif'
    _DownloadFromDatacube(bounds, outputPathname, cogUrl, 30.0, 30.0, False)


def _DownloadHrdem(bounds: LatLonBox, outputPathname: str, cogUrl: str, pixelSizeX: float,
                   pixelSizeY: float, compressOutput: bool) -> list[str]:
    epsg3979_bounds = _Epsg4326To3979(bounds)
    epsg3979_bounds.min_x -= pixelSizeX
    epsg3979_bounds.max_x += pixelSizeX
    epsg3979_bounds.min_y -= pixelSizeY
    epsg3979_bounds.max_y += pixelSizeY

    tile_names = _GetHrdemMosaicTiles(epsg3979_bounds)
    if len(tile_names) == 0:
        raise RuntimeError('Requested area is outside the bounds of the HRDEM Mosaic product.')

    output_pathnames = []
    for t in tile_names:
        cog_url_with_tile_no = cogUrl.replace('___', t)
        try:
            urllib.request.urlopen(cog_url_with_tile_no)
            url_exists = True
        except:
            url_exists = False # some HRDEM mosaic tiles have no associated GeoTIFF file (no data at all)
        if url_exists :
            i = len(output_pathnames)
            if i > 0:
                pathname_no_ext, ext = os.path.splitext(outputPathname)
                outputPathname = pathname_no_ext + '_{}'.format(i) + ext
            outputPathname = os.path.realpath(os.path.abspath(outputPathname)) # to return canonical paths
            _DownloadFromDatacube(bounds, outputPathname, cog_url_with_tile_no, pixelSizeX,
                                  pixelSizeY, compressOutput)
            output_pathnames.append(outputPathname)

    if len(output_pathnames) == 0:
        raise RuntimeError('No cloud optimized GeoTIFF file from the HRDEM Mosaic product is ' \
                           'currently available for the requested area.')

    return output_pathnames


def _DownloadFromDatacube(bounds: LatLonBox, outputPathname: str, cogUrl: str, pixelSizeX: float,
                          pixelSizeY: float, compressOutput: bool) -> None:
    epsg3979_bounds = _Epsg4326To3979(bounds)
    with rasterio.open(cogUrl) as src:
        window = src.window(epsg3979_bounds.min_x-pixelSizeX, epsg3979_bounds.min_y-pixelSizeY,
                            epsg3979_bounds.max_x+pixelSizeX, epsg3979_bounds.max_y+pixelSizeY)
        
        # ensure we request a window winthin the bounds of the tiff file
        col_off, row_off, width, height = window.flatten()
        if col_off < 0:
            width -= abs(col_off)
            col_off = 0
        if row_off < 0:
            height -= abs(row_off)
            row_off = 0
        width = min(width, src.width-col_off)
        height = min(height, src.height-row_off)
        if width <= 0 or height <= 0:
            raise RuntimeError('Requested area is outside the bounds of the cloud optimized GeoTFF file.')
        window = rasterio.windows.Window(col_off=col_off, row_off=row_off, height=height, width=width)

        raster_data = src.read(window=window)
        metadata = src.meta.copy()
        metadata.update({
            'height': raster_data.shape[1],
            'width': raster_data.shape[2],
            'count': raster_data.shape[0],
            'transform': rasterio.windows.transform(window, src.transform),
            # note: compression not making a big difference unless there are lots of "no data" values
            #       and takes more time to read/write the file
            'compress': 'LZW' if compressOutput else 'NONE'
        })

    os.makedirs(os.path.dirname(outputPathname), exist_ok=True)
    with rasterio.open(outputPathname, 'w', **metadata) as dst:
        dst.write(raster_data)


class _Epsg3979Box:
    def __init__(self, x0:float, y0:float, x1:float, y1:float):
        self.min_x:float = min(x0, x1)
        self.min_y:float = min(y0, y1)
        self.max_x:float = max(x0, x1)
        self.max_y:float = max(y0, y1)


def _Epsg4326To3979(bounds: LatLonBox) -> tuple[float, float, float, float]:
    crs_4326 = pyproj.CRS.from_epsg(4326)
    crs_3979 = pyproj.CRS.from_epsg(3979)
    transformer = pyproj.Transformer.from_crs(crs_from=crs_4326, crs_to=crs_3979, always_xy=True)
    bbox = transformer.transform_bounds(left=bounds.minLon, bottom=bounds.minLat,
                                        right=bounds.maxLon, top=bounds.maxLat)
    return _Epsg3979Box(*bbox)


def _Intersects(bounds0: _Epsg3979Box, bounds1: _Epsg3979Box) -> bool:
    if bounds0.min_y > bounds1.max_y:
        return False
    if bounds0.max_y < bounds1.min_y:
        return False
    if bounds0.min_x > bounds1.max_x:
        return False
    if bounds0.max_x < bounds1.min_x:
        return False
    return True


def _GetHrdemMosaicTiles(bounds: _Epsg3979Box) -> list[str]:
    tile_nos = []
    for t in _HRDEM_TILES:
        if _Intersects(bounds, t[1]):
            tile_nos.append(t[0])
    return tile_nos


# See page 12 of https://ftp.maps.canada.ca/pub/elevation/dem_mne/HRDEMmosaic_mosaiqueMNEHR/HRDEM_Mosaic_Product_Specification.pdf
_HRDEM_TILES = [
    ('8_1',  _Epsg3979Box(1000000, -1000000,  1500000, -500000)),

    ('6_2',  _Epsg3979Box( 500000,  -500000,        0,       0)),
    ('7_2',  _Epsg3979Box(1000000,  -500000,   500000,       0)),
    ('8_2',  _Epsg3979Box(1500000,  -500000,  1000000,       0)),
    ('9_2',  _Epsg3979Box(2000000,  -500000,  1500000,       0)),
    ('10_2', _Epsg3979Box(2500000,  -500000,  2000000,       0)),

    ('1_3',  _Epsg3979Box(-2000000,       0, -2500000,  500000)),
    ('2_3',  _Epsg3979Box(-1500000,       0, -2000000,  500000)),
    ('3_3',  _Epsg3979Box(-1000000,       0, -1500000,  500000)),
    ('4_3',  _Epsg3979Box( -500000,       0, -1000000,  500000)),
    ('5_3',  _Epsg3979Box(       0,  -50000,  -500000,  500000)),
    ('6_3',  _Epsg3979Box(  500000,       0,        0,  500000)),
    ('7_3',  _Epsg3979Box( 1000000,       0,   500000,  500000)),
    ('8_3',  _Epsg3979Box( 1500000,       0,  1000000,  500000)),
    ('9_3',  _Epsg3979Box( 2000000,       0,  1500000,  500000)),
    ('10_3', _Epsg3979Box( 2500000,       0,  2000000,  500000)),
    ('11_3', _Epsg3979Box( 3000000,       0,  2500000,  500000)),

    ('1_4',  _Epsg3979Box(-2000000,  500000, -2500000, 1000000)),
    ('2_4',  _Epsg3979Box(-1500000,  500000, -2000000, 1000000)),
    ('3_4',  _Epsg3979Box(-1000000,  500000, -1500000, 1000000)),
    ('4_4',  _Epsg3979Box( -500000,  500000, -1000000, 1000000)),
    ('5_4',  _Epsg3979Box(       0,  500000,  -500000, 1000000)),
    ('6_4',  _Epsg3979Box(  500000,  500000,        0, 1000000)),
    ('7_4',  _Epsg3979Box( 1000000,  500000,   500000, 1000000)),
    ('8_4',  _Epsg3979Box( 1500000,  500000,  1000000, 1000000)),
    ('9_4',  _Epsg3979Box( 2000000,  500000,  1500000, 1000000)),
    ('10_4', _Epsg3979Box( 2500000,  500000,  2000000, 1000000)),
    ('11_4', _Epsg3979Box( 3050000,  500000,  2500000, 1000000)),

    ('1_5',  _Epsg3979Box(-2000000, 1000000, -2500000, 1500000)),
    ('2_5',  _Epsg3979Box(-1500000, 1000000, -2000000, 1500000)),
    ('3_5',  _Epsg3979Box(-1000000, 1000000, -1500000, 1500000)),
    ('4_5',  _Epsg3979Box( -500000, 1000000, -1000000, 1500000)),
    ('5_5',  _Epsg3979Box(       0, 1000000,  -500000, 1500000)),
    ('6_5',  _Epsg3979Box(  500000, 1000000,        0, 1500000)),
    ('7_5',  _Epsg3979Box( 1000000, 1000000,   500000, 1500000)),
    ('8_5',  _Epsg3979Box( 1500000, 1000000,  1000000, 1500000)),
    ('9_5',  _Epsg3979Box( 2000000, 1000000,  1500000, 1500000)),
    ('10_5', _Epsg3979Box( 2500000, 1000000,  2000000, 1500000)),
    ('11_5', _Epsg3979Box( 3000000, 1000000,  2500000, 1500000)),

    ('1_6',  _Epsg3979Box(-2000000, 1500000, -2500000, 2000000)),
    ('2_6',  _Epsg3979Box(-1500000, 1500000, -2000000, 2000000)),
    ('3_6',  _Epsg3979Box(-1000000, 1500000, -1500000, 2000000)),
    ('4_6',  _Epsg3979Box( -500000, 1500000, -1000000, 2000000)),
    ('5_6',  _Epsg3979Box(       0, 1500000,  -500000, 2000000)),
    ('6_6',  _Epsg3979Box(  500000, 1500000,        0, 2000000)),
    ('7_6',  _Epsg3979Box( 1000000, 1500000,   500000, 2000000)),
    ('8_6',  _Epsg3979Box( 1500000, 1500000,  1000000, 2000000)),
    ('9_6',  _Epsg3979Box( 2000000, 1500000,  1500000, 2000000)),

    ('1_7',  _Epsg3979Box(-2000000, 2000000, -2500000, 2500000)),
    ('2_7',  _Epsg3979Box(-1500000, 2000000, -2000000, 2500000)),
    ('3_7',  _Epsg3979Box(-1000000, 2000000, -1500000, 2500000)),
    ('4_7',  _Epsg3979Box( -500000, 2000000, -1000000, 2500000)),
    ('5_7',  _Epsg3979Box(       0, 2000000,  -500000, 2500000)),
    ('6_7',  _Epsg3979Box(  500000, 2000000,        0, 2500000)),
    ('7_7',  _Epsg3979Box( 1000000, 2000000,   500000, 2500000)),
    ('8_7',  _Epsg3979Box( 1500000, 2000000,  1000000, 2500000)),

    ('2_8',  _Epsg3979Box(-1500000, 2500000, -2000000, 3000000)),
    ('3_8',  _Epsg3979Box(-1000000, 2500000, -1500000, 3000000)),
    ('4_8',  _Epsg3979Box( -500000, 2500000, -1000000, 3000000)),
    ('5_8',  _Epsg3979Box(       0, 2500000,  -500000, 3000000)),
    ('6_8',  _Epsg3979Box(  500000, 2500000,        0, 3000000)),
    ('7_8',  _Epsg3979Box( 1050000, 2500000,   500000, 3000000)),

    ('4_9',  _Epsg3979Box( -500000, 3000000, -1000000, 3500000)),
    ('5_9',  _Epsg3979Box(       0, 3000000,  -500000, 3500000)),
    ('6_9',  _Epsg3979Box(  500000, 3000000,        0, 3500000)),

    ('6_10', _Epsg3979Box(  550000, 3500000,   -50000, 4000000))
]
