# buildings helper module
Additional building related functionalities in support of crc-covlib.

This module allows to get building footprint and height data from either a shapefile or from OpenStreetMap web services (see GetBuildingsFromShapefile() and GetBuildingsFromOpenStreetMap()). From this data, additional building releted metrics can be computed along a direct transmission path (see GetP1411UrbanMetrics(), GetP1411ResidentialMetrics()).

```python
from crc_covlib.helper import buildings 
```

- [LatLonBox (class)](#latlonbox)
- [Building (class)](#building)
- [GetBuildingsFromShapefile](#getbuildingsfromshapefile)
- [GetBuildingsFromOpenStreetMap](#getbuildingsfromopenstreetmap)
- [ExportBuildingsToGeojsonFile](#exportbuildingstogeojsonfile)
- [GetBuildingHeightsProfile](#getbuildingheightsprofile)
- [GetP1411UrbanMetrics](#getp1411urbanmetrics)
- [GetP1411ResidentialMetrics](#getp1411residentialmetrics)

***

### LatLonBox
#### crc_covlib.helper.buildings.LatLonBox
```python
class LatLonBox(minLat:float, minLon:float, maxLat:float, maxLon:float)
```
Bounding box in geographical coordinates (WGS84 or EPSG 4326 assumed).
    
Attributes:
- __minLat__ (float): minimum latitude of bounding box (degrees).
- __minLon__ (float): minimum longitude of bounding box (degrees).
- __maxLat__ (float): maximum latitude of bounding box (degrees).
- __maxLon__ (float): maximum longutide of bounding box (degrees).

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Building
#### crc_covlib.helper.buildings.Building
```python
class Building
```
Data class for a building.

Attributes:
- __height_m__ (float): Height of the building (meters).
- __footprint__ (shapely.Polygon): Coordinates of the building's footprint.
- __osmid__ (int|None): The building's OpenStreetMap ID when applicable, None otherwise.
- __footprintCRS__ (pyproj.crs.CRS): The coordinate reference system of the building's footprint.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetBuildingsFromShapefile
#### crc_covlib.helper.buildings.GetBuildingsFromShapefile
```python
def GetBuildingsFromShapefile(pathname: str, heightAttribute: str,
                              bounds: LatLonBox|None=None,
                              toCRS: pyproj.crs.CRS|None=pyproj.CRS.from_epsg(4326),
                              defaultHeight_m: float=3.0) -> list[Building]
```
Extracts building height and footprint data from the specified shapefile. The footprints are assumed to be stored using the Polygon shape type (5) within the shapefile.

Args:
- __pathname__ (str): Absolute or relative path to a shapefile (.shp) containing building heights and footprints.
- __heightAttribute__ (str): Name of the shapefile attribute to get the building heights from.
- __bounds__ (crc_covlib.helper.buildings.LatLonBox|None): When specified, only the area covered by the specified bounds is read. Otherwise the whole content of the file is used.
- __toCRS__ (pyproj.crs.CRS|None): A coordinate reference system (CRS). When specified, the extracted footprints are converted to this CRS. Otherwise the footprints are returned in their original CRS from the shapefile.
- __defaultHeight_m__ (float): Default height value (meters) to be applied to any building whose height cannot be obtained from the heightAttribute value.

Returns:
- (list[crc_covlib.helper.buildings.Building]): A list of [Building](#building) objects.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetBuildingsFromOpenStreetMap
#### crc_covlib.helper.buildings.GetBuildingsFromOpenStreetMap
```python
def GetBuildingsFromOpenStreetMap(bounds: LatLonBox, toCRS: pyproj.crs.CRS|None=pyproj.CRS.from_epsg(4326),
                                  defaultHeight_m: float=3.0, avgFloorHeight_m: float=3.0) -> list[Building]
```
Downloads building footprint data from OpenStreetMap. Building heights are estimated from the number of above-ground floors ('building:levels' tag) when present.

Args:
- __bounds__ (crc_covlib.helper.buildings.LatLonBox): A bounding box from which to get the building data.
- __toCRS__ (pyproj.crs.CRS|None): A coordinate reference system (CRS). When specified, the downloaded footprints are converted to this CRS. Otherwise the footprints are returned in their original CRS from OpenStreetMap.
- __defaultHeight_m__ (float): Default height value (meters) to be applied to any building whose height cannot be estimated (i.e. when the 'building:levels' tag is missing).
- __avgFloorHeight_m__ (float): Average floor height (meters) for estimating building heights.

Returns:
- (list[crc_covlib.helper.buildings.Building]): A list of Building objects.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ExportBuildingsToGeojsonFile
#### crc_covlib.helper.buildings.ExportBuildingsToGeojsonFile
```python
def ExportBuildingsToGeojsonFile(buildings: list[Building], pathname: str) -> None
```
Exports a list of Building objects to a geojson file.

Args:
- __buildings__ (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
- __pathname__ (str): Absolute or relative path to the geojson file to create or overwrite.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetBuildingHeightsProfile
#### crc_covlib.helper.buildings.GetBuildingHeightsProfile
```python
def GetBuildingHeightsProfile(buildings: list[Building], latLonProfile: ArrayLike
                             ) -> tuple[NDArray[float64], list[Building]]
```
Gets a building heights profile (meters) from the specified building data.

Args:
- __buildings__ (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
- __latLonProfile__ (numpy.typing.ArrayLike): A latitude/longitude profile (degrees, EPSG:4326) in the form of a 2D list or array. The latitude of the first point should be at latLonProfile[0][0] and its longitude at profile[0][1]. Such a profile may be obtained from the topography.GetLatLonProfile() function.

Return:
- __bldgHeightsProfile_m__ (numpy.typing.NDArray[numpy.float64]): The building heights profile (meters) for the points specified in latLonProfile. A value of 0 is used at locations where there is no building.
- __encounteredBuildings__ (list[Building]): List of Building objects from buildings that were encountered along latLonProfile. Buildings are listed in the order they were encountered iterating over latLonProfile.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetP1411UrbanMetrics
#### crc_covlib.helper.buildings.GetP1411UrbanMetrics
```python
def GetP1411UrbanMetrics(buildings: list[Building], txLat: float, txLon: float,
                         rxLat: float, rxLon: float, res_m: float=1.0, pathExt_m: float=100.0
                         ) -> tuple[float, float, float, float, float]
```
Gets building related metrics for a direct transmitter to receiver path. The metrics mainly consists of input parameter values for site specific, over roof-tops urban/suburban propagation models to be found in the ITU-R P.1411 recommendation. See FIGURE 2 of ITU-R P.1411-12 for more details.

Args:
- __buildings__ (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
- __txLat__ (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
- __txLon__ (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
- __rxLat__ (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
- __rxLon__ (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
- __res_m__ (float): Resolution (meters). Presence of buildings along the path will be evaluated about every res_m meters.
- __pathExt_m__ (float): Path extension length (meters). Extra distance passed the receiver to look for buildings in order to calculate the street width and building separation distance values.
            
Returns:
- __d_m__ (float): Path length (great-circle distance) from the transmitter to the receiver (meters).
- __hr_m__ (float): Average height of buildings along the path (meters). Set to -1 when no building is found along the path.
- __b_m__ (float): Average building separation distance along the path (meters). Set to -1 when less than 2 buildings are found along the path.
- __l_m__ (float): Length of the path covered by buildings (meters). Set to zero when no building is found along the path.
- __w_m__ (float): Street width at the receiver location (meters), which is the distance between the two buildings encompassing the receiver. Set to -1 when two such buildings cannot be found along the path.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetP1411ResidentialMetrics
#### crc_covlib.helper.buildings.GetP1411ResidentialMetrics
```python
def GetP1411ResidentialMetrics(buildings: list[Building], txLat: float, txLon: float,
                               rxLat: float, rxLon: float, res_m: float=1.0
                               ) -> tuple[float, float, float, float, float, float]
```
Gets building related metrics for a direct transmitter to receiver path. The metrics mainly consists of input parameter values for the site specific, below roof-top to near street level residential propagation model to be found in the ITU-R P.1411 recommendation. See FIGURE 12 of ITU-R P.1411-12 for more details.

Args:
- __buildings__ (list[crc_covlib.helper.buildings.Building]): A list of Building objects obtained from the GetBuildingsFromShapefile() or GetBuildingsFromOpenStreetMap() functions.
- __txLat__ (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
- __txLon__ (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
- __rxLat__ (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
- __rxLon__ (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
- __res_m__ (float): Resolution (meters). Presence of buildings along the path will be evaluated about every res_m meters.
            
Returns:
- __d_m__ (float): Path length (great-circle distance) from the transmitter to the receiver (meters).
- __hbTx_m__ (float): Height of nearest building from transmitter in receiver direction (meters). Set to -1 when no building is found along the path.
- __hbRx_m__ (float): Feight of nearest building from receiver in transmitter direction (meters). Set to -1 when no building is found along the path.
- __a_m__ (float): Distance between transmitter and nearest building from transmitter (meters). Set to -1 when no building is found along the path.
- __b_m__ (float): Distance between nearest buildings from transmitter and receiver (meters). Set to -1 when no building is found along the path.
- __c_m__ (float): Distance between receiver and nearest building from receiver (meters). Set to -1 when no building is found along the path.

[Back to top](#buildings-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***