# topography helper module
Additional topography related functionalities in support of crc-covlib.

```python
from crc_covlib.helper import topography 
```

- [CustomDataInfo (class)](#customdatainfo)
- [LatLonBox (class)](#latlonbox)
- [LoadFileAsCustomTerrainElevData](#loadfileascustomterrainelevdata)
- [LoadFileAsCustomLandCoverData](#loadfileascustomlandcoverdata)
- [LoadFileAsCustomSurfaceElevData](#loadfileascustomsurfaceelevdata)
- [GetDistanceProfile](#getdistanceprofile)
- [GetLatLonProfile](#getlatlonprofile)
- [GetTerrainElevProfile](#getterrainelevprofile)
- [GetSurfaceElevProfile](#getsurfaceelevprofile)
- [GetLandCoverProfile](#getlandcoverprofile)
- [GetMappedLandCoverProfile](#getmappedlandcoverprofile)
- [GetRasterProfile](#getrasterprofile)

***

### CustomDataInfo
#### crc_covlib.helper.topography.CustomDataInfo
```python
class CustomDataInfo()
```
Holds metadata for crc-covlib's custom terrain elevation data, land cover data or surface elevation data. Geographical coordinates in WGS84 (EPSG 4326).
    
Attributes:
- __lowerLeftCornerLat_degrees__ (float): crc-covlib Simulation object.
- __lowerLeftCornerLat_degrees__ (float): lower left corner latitude (i.e. minimum latitude) of the custom data (degrees).
- __lowerLeftCornerLon_degrees__ (float): lower left corner longitude (i.e. minimum longitude) of the custom data (degrees).
- __upperRightCornerLat_degrees__ (float): upper right corner latitude (i.e. maximum latitude) of the custom data (degrees).
- __upperRightCornerLon_degrees__ (float): upper right corner longitude (i.e. maximum longitude) of the custom data (degrees).
- __numHorizSamples__ (int): number of horizontal samples, or width, of the custom data.
- __numVertSamples__ (int): number of vertical samples, or height, of the custom data.
- __isNoDataValueDefined__ (bool): indicates whether a specific value within the custom data should be interpreted as "no data".
- __noDataValue__ (int | float): value to interpret as "no data", irrelevant when isNoDataValueDefined is set to False.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LatLonBox
#### crc_covlib.helper.topography.LatLonBox
```python
class LatLonBox(minLat:float, minLon:float, maxLat:float, maxLon:float)
```
Bounding box in geographical coordinates (WGS84 or EPSG 4326 assumed).
    
Attributes:
- __minLat__ (float): minimum latitude of bounding box (degrees).
- __minLon__ (float): minimum longitude of bounding box (degrees).
- __maxLat__ (float): maximum latitude of bounding box (degrees).
- __maxLon__ (float): maximum longutide of bounding box (degrees).

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadFileAsCustomTerrainElevData
#### crc_covlib.helper.topography.LoadFileAsCustomTerrainElevData
```python
def LoadFileAsCustomTerrainElevData(sim: Simulation, pathname: str, bounds: LatLonBox|None=None,
                                    setAsPrimary: bool=True, band: int=1) -> CustomDataInfo
```
Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326) geographic coordinate system if required, and passes the content as custom terrain elevation data to the crc-covlib Simulation object (calling AddCustomTerrainElevData).
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __pathname__ (str): Absolute or relative path to a georeferenced raster file containing terrain elevation data in meters.
- __bounds__ (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered by the specified bounds is read. Otherwise the whole content of the file is used.
- __setAsPrimary__ (bool): If set to True, the sim object's primary terrain elevation data source is set to TERR_ELEV_CUSTOM.
- __band__ (int): Band to use for the raster file, indexed from 1.

Returns:
- crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object containing the argument values used when calling AddCustomTerrainElevData(). Returns None on failure.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadFileAsCustomLandCoverData
#### crc_covlib.helper.topography.LoadFileAsCustomLandCoverData
```python
def LoadFileAsCustomLandCoverData(sim: Simulation, pathname: str, bounds: LatLonBox|None=None,
                                  setAsPrimary: bool=True, band: int=1) -> CustomDataInfo
```
Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326) geographic coordinate system if required, and passes the content as custom land cover data to the crc-covlib Simulation object (calling AddCustomLandCoverData).
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __pathname__ (str): Absolute or relative path to a georeferenced raster file containing land cover (clutter) data.
- __bounds__ (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered by the specified bounds is read. Otherwise the whole content of the file is used.
- __setAsPrimary__ (bool): If set to True, the sim object's primary land cover data source is set to LAND_COVER_CUSTOM.
- __band__ (int): Band to use for the raster file, indexed from 1.

Returns:
- crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object containing the argument values used when calling AddCustomLandCoverData(). Returns None on failure.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadFileAsCustomSurfaceElevData
#### crc_covlib.helper.topography.LoadFileAsCustomSurfaceElevData
```python
def LoadFileAsCustomSurfaceElevData(sim: Simulation, pathname: str, bounds: LatLonBox|None=None,
                                    setAsPrimary: bool=True, band: int=1) -> CustomDataInfo
```
Reads the specified raster data file, reprojects it (in memory) to the WGS84 (EPSG 4326) geographic coordinate system if required, and passes the content as custom surface elevation data to the crc-covlib Simulation object (calling AddCustomSurfaceElevData).
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __pathname__ (str): Absolute or relative path to a georeferenced raster file containing surface elevation data in meters.
- __bounds__ (crc_covlib.helper.topography.LatLonBox|None): If specified, only the area covered by the specified bounds is read. Otherwise the whole content of the file is used.
- __setAsPrimary__ (bool): If set to True, the sim object's primary surface elevation data source is set to SURF_ELEV_CUSTOM.
- __band__ (int): Band to use for the raster file, indexed from 1.

Returns:
- crc_covlib.helper.topography.CustomDataInfo|None: If successful, returns an object containing the argument values used when calling AddCustomSurfaceElevData(). Returns None on failure.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetDistanceProfile
#### crc_covlib.helper.topography.GetDistanceProfile
```python
def GetDistanceProfile(lat0: float, lon0: float, lat1: float, lon1: float, res_m: float,
                       minNumPoints: int=3, maxNumPoints: int|None=None) -> NDArray[float64]
```
Gets a great-circle distance profile (km) of evenly spaced points between the two specified points on the surface of the Earth.

Note: Calling GetDistanceProfile() and GetLatLonProfile() with the same input parameters yields profiles for the same corresponding points.

Args:
- __lat0__ (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
- __lon0__ (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
- __lat1__ (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
- __lon1__ (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.
- __res_m__ (float): Resolution (meters). Indicates that the returned profile points must be approximately spaced by res_m meters.
- __minNumPoints__ (float): Minimum number of points (i.e. distances) that the returned profile must contain.
- __maxNumPoints__ (int|None): Maximum number of points (i.e. distances) allowed in the returned profile. No limit when set to None.

Returns:
- __distProfile__ (numpy.typing.NDArray[numpy.float64]): A great-circle distance profile (km) of evenly spaced points between the two specified points on the surface of the Earth.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetLatLonProfile
#### crc_covlib.helper.topography.GetLatLonProfile
```python
def GetLatLonProfile(lat0: float, lon0: float, lat1: float, lon1: float, res_m: float,
                     minNumPoints: int=3, maxNumPoints: int|None=None) -> NDArray[float64]
```
Gets a latitude/longitude profile (degrees) of evenly spaced points (using a great-circle distance algorithm) between the two specified points on the surface of the Earth.

Note: Calling GetDistanceProfile() and GetLatLonProfile() with the same input parameters yields profiles for the same corresponding points.

Args:
- __lat0__ (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
- __lon0__ (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
- __lat1__ (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
- __lon1__ (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.
- __res_m__ (float): Resolution (meters). Indicates that the returned profile points must be approximately spaced by res_m meters.
- __minNumPoints__ (float): Minimum number of points (i.e. distances) that the returned profile must contain.
- __maxNumPoints__ (int|None): Maximum number of points (i.e. distances) allowed in the returned profile. No limit when set to None.

Returns:
- __latLonProfile__ (numpy.typing.NDArray[numpy.float64]): A latitude/longitude profile (degrees) of evenly spaced points (using a great-circle distance algorithm) between the two specified points on the surface of the Earth. The returned profile is a 2D array of shape (numPoints, 2). The latitude of the first point is at latLonProfile[0][0] and its longitude is at latLonProfile[0][1].

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetTerrainElevProfile
#### crc_covlib.helper.topography.GetTerrainElevProfile
```python
def GetTerrainElevProfile(sim: Simulation, latLonProfile: ArrayLike, noDataValue: float=0
                          ) -> tuple[NDArray[float64], bool]
```
Gets a terrain elevation profile (meters) using the terrain elevation source(s) from the specified Simulation object.

Args:
- __sim__ (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the GetLatLonProfile() function.
- __noDataValue__ (float): Terrain elevation (m) value to be used in the returned profile when no terrain elevation data can be retrieved at a specific location.

Returns:
- __terrainElevProfile__ (numpy.typing.NDArray[numpy.float64]): The terrain elevation profile (meters) for the points specified in latLonProfile.
- __status__ (bool): True when all terrain elevation data could be successfully retrieved. False when the terrain elevation data could not be retrieved for at least one of the locations in latLonProfile.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetSurfaceElevProfile
#### crc_covlib.helper.topography.GetSurfaceElevProfile
```python
def GetSurfaceElevProfile(sim: Simulation, latLonProfile: ArrayLike, noDataValue: float=0
                          ) -> tuple[NDArray[float64], bool]
```
Gets a surface elevation profile (meters) using the surface elevation source(s) from the specified Simulation object.

Args:
- __sim__ (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the GetLatLonProfile() function.
- __noDataValue__ (float): Surface elevation (m) value to be used in the returned profile when no surface elevation data can be retrieved at a specific location.

Returns:
- __surfaceElevProfile__ (numpy.typing.NDArray[numpy.float64]): The surface elevation profile (meters) for the points specified in latLonProfile.
- __status__ (bool): True when all surface elevation data could be successfully retrieved. False when the surface elevation data could not be retrieved for at least one of the locations in latLonProfile.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetLandCoverProfile
#### crc_covlib.helper.topography.GetLandCoverProfile
```python
def GetLandCoverProfile(sim: Simulation, latLonProfile: ArrayLike, noDataValue: int=-1
                        ) -> tuple[NDArray[intc], bool]
```
Gets a land cover profile using the land cover source(s) from the specified Simulation object.

Args:
- __sim__ (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the GetLatLonProfile() function.
- __noDataValue__ (int): Value to be used in the returned profile when no land cover data can be retrieved at a specific location.

Returns:
- __landCoverProfile__ (numpy.typing.NDArray[numpy.intc]): The land cover profile for the points specified in latLonProfile.
- __status__ (bool): True when all land cover data could be successfully retrieved. False when the land cover data could not be retrieved for at least one of the locations in latLonProfile.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetMappedLandCoverProfile
#### crc_covlib.helper.topography.GetMappedLandCoverProfile
```python
def GetMappedLandCoverProfile(sim: Simulation, latLonProfile: ArrayLike) -> tuple[NDArray[intc], bool]
```
Gets a mapped land cover profile using the land cover source(s) and mappings from the specified Simulation object. A mapped land cover value is a land cover value that has been converted to a recognized and usable value by the currently selected propagation model in the Simulation object.

Args:
- __sim__ (crc_covlib.simulation.Simulation): A crc-covlib Simulation object.
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the GetLatLonProfile() function.

Returns:
- __mappedLandCoverProfile__ (numpy.typing.NDArray[numpy.intc]): The mapped land cover profile for the points specified in latLonProfile.
- __status__ (bool): True when all land cover data could be successfully retrieved. False when the land cover data could not be retrieved for at least one of the locations in latLonProfile.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetRasterProfile
#### crc_covlib.helper.topography.GetRasterProfile
```python
def GetRasterProfile(pathnames: str|list[str], latLonProfile: ArrayLike,
                     noDataValue: float=0, band:int=1) -> tuple[list[float], bool]
```
Gets a profile of pixel values from the specified georeferenced raster file(s). This function may be used to either get a terrain elevation, a surface elevation or a land cover profile depending on the actual content of the raster file(s). Pixel values are obtained using a nearest neighbor algorithm.

Please note that this function is provided for convenience and runs much slower than the other Get...Profile() functions. It is advisable to use the other functions whenever possible if a large number of profiles needs to be obtained.

Args:
- __pathnames__ (str|list[str]): Absolute or relative path to a georeferenced raster file, or a list of absolute or relative paths to georeferenced raster file(s) (useful when the specified latLonProfile extends over more than one file).
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the GetLatLonProfile() function.
- __noDataValue__ (float): Pixel value to be used in the returned profile when no pixel data can be retrieved at a specific location.
- __band__ (int): Band to use for the raster file(s), indexed from 1.

Returns:
- __profile__ (list[float]): Pixel value profile for the locations specified in latLonProfile.
- __status__ (bool): True when all pixel values could be successfully retrieved, False otherwise.

[Back to top](#topography-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***