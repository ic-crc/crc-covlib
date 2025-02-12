# streets helper module
Additional street related functionalities using OpenStreetMap data in support of crc-covlib.

This module aims at determining potential propagation paths for scenarios where antennas are below rooftops and the radio path is understood to occur mainly through street canyons. Typical usage is the obtention of input data for propagation models such as those found in ITU-R P.1411.

Typical workflow:
  1. GetStreetGraphFromBox() or GetStreetGraphFromPoints() to download OpenStreetMap data for the area of interest. This data may be visualized using DisplayStreetGraph().
  2. Use GetStreetCanyonsRadioPaths() to compute the shortest path(s) along street lines from the transmitter to the receiver stations. GetStreetCanyonsRadioPaths() returns one or more StreetCanyonsRadioPath objects. ExportRadioPathToGeojsonFile() may be used to visualize those path objects from a GIS software.
  3. Optionally, computed StreetCanyonsRadioPath objects may be further simplified (i.e. that is, to reduce the number of vertices in the path) using SimplifyRadioPath() or SortRadioPathsByStreetCornerCount().
  4. The DistancesToStreetCorners(), StreetCornerAngles() and ReceiverStreetOrientation() functions, as well as attributes from the StreetCanyonsRadioPath objects (distances_m, turnAngles_deg, etc.) may be used for input parameters to propagation models (ITU-R P.1411 for instance).

```python
from crc_covlib.helper import streets 
```

- [StreetGraph (class)](#streetgraph)
- [StreetCanyonsRadioPath (class)](#streetcanyonsradiopath)
- [GetStreetGraphFromBox](#getstreetgraphfrombox)
- [GetStreetGraphFromPoints](#getstreetgraphfrompoints)
- [DisplayStreetGraph](#displaystreetgraph)
- [GetStreetCanyonsRadioPaths](#getstreetcanyonsradiopaths)
- [ExportRadioPathToGeojsonFile](#exportradiopathtogeojsonfile)
- [SimplifyRadioPath](#simplifyradiopath)
- [DistancesToStreetCorners](#distancestostreetcorners)
- [StreetCornerAngles](#streetcornerangles)
- [SortRadioPathsByStreetCornerCount](#sortradiopathsbystreetcornercount)
- [ReceiverStreetOrientation](#receiverstreetorientation)

***

### StreetGraph
#### crc_covlib.helper.streets.StreetGraph
```python
class StreetGraph()
```
Street graph generated from OpenStreetMap (OSM) data. An instance may be obtained from GetStreetGraphFromBox() or GetStreetGraphFromPoints() and is typically used as input for GetStreetCanyonsRadioPaths().

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StreetCanyonsRadioPath
#### crc_covlib.helper.streets.StreetCanyonsRadioPath
```python
class StreetCanyonsRadioPath()
```
Data class for storing details about a propagation path occuring through street canyons (i.e. propagation along street lines). One ore more instance(s) may be obtained from GetStreetCanyonsRadioPaths().

Attributes:
- __latLonPath__ (list[tuple[float, float]]): List of n (latitude, longitude) tuples constituting the vertices for the propagation path along street segments, from transmitter to receiver (degrees, EPSG:4326).
- __turnAngles_deg__ (list[float]): List of n-2 turn angles between street segments in latLonPath (degrees). The first angle is at latLonPath[1] and the last angle is at latLonPath[-2]. Angle values are from 0 (going straight forward) to 180 (u-turn). Note that turn angles are always positive so there is no differentiation between right-hand and left-hand turns at the same angle.
- __isIntersection__ (list[bool]): List of n-2 boolean values indicating whether points between street segments in latLonPath are street intersections. The first boolean value is at latLonPath[1] and the last one is at latLonPath[-2].
- __distances_m__ (list[float]): List of n-1 street segment distances (meters) in latLonPath. The first value is the distance between latLonPath[0] and latLonPath[1], the last value is the distance between latLonPath[-2] and latLonPath[-1].
- __txLatLon__ (tuple[float, float]): Transmitter's location as a (latitude, longitude) tuple (degrees, EPSG:4326). latLonPath[0] is the closest point from txLatLon that is located on a street segment.
- __rxLatLon__ (tuple[float, float]): Receiver's location as a (latitude, longitude) tuple (degrees, EPSG:4326). latLonPath[-1] is the closest point from rxLatLon that is located on a street segment.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetStreetGraphFromBox
#### crc_covlib.helper.streets.GetStreetGraphFromBox
```python
def GetStreetGraphFromBox(minLat: float, minLon: float, maxLat: float, maxLon: float) -> StreetGraph
```
Downloads OpenStreetMap (OSM) data for the specified boundaries and creates a graph using [OSMnx](https://osmnx.readthedocs.io/en/stable/). The returned graph object may be used as input for the GetStreetCanyonsRadioPaths() function.

Args:
- __minLat__ (float): Minimum latitude boundary for the OSM data download (degrees, EPSG:4326), with -90 <= minLat <= 90.
- __minLon__ (float): Minimum longitude boundary for the OSM data download (degrees, EPSG:4326), with -180 <= minLon <= 180.
- __maxLat__ (float): Maximum latitude boundary for the OSM data download (degrees, EPSG:4326), with -90 <= maxLat <= 90.
- __maxLon__ (float): Maximum longitude boundary for the OSM data download (degrees, EPSG:4326), with -180 <= maxLon <= 180.

Returns:
- (crc_covlib.helper.streets.StreetGraph): OSM data street graph.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetStreetGraphFromPoints
#### crc_covlib.helper.streets.GetStreetGraphFromPoints
```python
def GetStreetGraphFromPoints(latLonPoints: list[tuple[float,float]], bufferDist_m: float=250) -> StreetGraph
```
Downloads OpenStreetMap (OSM) data for a boundary box containing the specified buffered points and creates a graph using [OSMnx](https://osmnx.readthedocs.io/en/stable/). The returned graph object may be used as input for the GetStreetCanyonsRadioPaths() function.

Args:
- __latLonPoints__ (list[tuple[float,float]]): List of (latitude, longitude) tuples for determining the OSM data download boundary box (degrees, EPSG:4326).
- __bufferDist_m__ (float): Buffer distance around the specified points (meters).

Returns:
- (crc_covlib.helper.streets.StreetGraph): OSM data street graph.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DisplayStreetGraph
#### crc_covlib.helper.streets.DisplayStreetGraph
```python
def DisplayStreetGraph(graph: StreetGraph) -> None
```
Displays a StreetGraph object.

Args:
- __graph__ (crc_covlib.helper.streets.StreetGraph): The StreetGraph object to be displayed.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetStreetCanyonsRadioPaths
#### crc_covlib.helper.streets.GetStreetCanyonsRadioPaths
```python
def GetStreetCanyonsRadioPaths(graph: StreetGraph|None, txLat: float, txLon: float,
                               rxLat: float, rxLon: float, numPaths: int
                               ) -> list[StreetCanyonsRadioPath]
```
Computes one or more radio paths along street lines from the transmitter location to thereceiver location. Returned paths are those to be found having the smallest travel distances. The information contained in the returned object(s) may be used as input for some propagation models (ITU-R P.1411 for instance) where antennas are below rooftops and the radio path is understood to occur mainly through street canyons.

Args:
- __graph__ (crc_covlib.helper.streets.StreetGraph|None): A StreetGraph object obtained from GetStreetGraphFromBox() or GetStreetGraphFromPoints(). When set to None, a new graph object is produced internally based on the transmitter and receiver locations.
- __txLat__ (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
- __txLon__ (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
- __rxLat__ (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
- __rxLon__ (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.
- __numPaths__ (int): Maximum number of paths to be computed and returned, with 1 <= numPaths.

Returns:
- (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): List of StreetCanyonsRadioPath objects. The number of items in the returned list will usually be equal to numPaths, but it can be less if less than numPaths paths could be computed.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ExportRadioPathToGeojsonFile
#### crc_covlib.helper.streets.ExportRadioPathToGeojsonFile
```python
def ExportRadioPathToGeojsonFile(radioPath: StreetCanyonsRadioPath, pathname: str) -> None
```
Exports a street canyons radio path to a geojson file.

Args:
- __radioPath__ (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath object obtained from the GetStreetCanyonsRadioPaths() function.
- __pathname__ (str): Absolute or relative path to the geojson file to create or overwrite.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SimplifyRadioPath
#### crc_covlib.helper.streets.SimplifyRadioPath
```python
def SimplifyRadioPath(radioPath: StreetCanyonsRadioPath, tolerance_m: float=12) -> StreetCanyonsRadioPath
```
Simplifies a street canyons radio path (i.e. reduces the number of vertices in the path).

Args:
- __radioPath__ (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath object obtained from the GetStreetCanyonsRadioPaths() function.
- __tolerance_m__ (float): Maximum allowed path geometry displacement (meters).

Returns:
- (crc_covlib.helper.streets.StreetCanyonsRadioPath): A new street canyons radio path that is a simplified version of the original.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DistancesToStreetCorners
#### crc_covlib.helper.streets.DistancesToStreetCorners
```python
def DistancesToStreetCorners(radioPath: StreetCanyonsRadioPath,
                             turnAngleThreshold_deg: float=20) -> list[float]
```
Gets a list of travel distances, in meters, to the next street corner (i.e. the next turn point at a turn angle of at least turnAngleThreshold_deg) or to the receiver, along the specified street canyons radio path.

Examples of return values:
- [50, 100, 30]:\
    Travel distance from tx station to first street corner is 50m.\
    Travel distance from first street corner to second street corner is 100m.\
    Travel distance from second street corner to rx station is 30m.
- [75]:\
    Travel distance from tx station to rx station is 75m (no street corner).

Args:
- __radioPath__ (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath object obtained from the GetStreetCanyonsRadioPaths() function.
- __turnAngleThreshold_deg__ (float): Turn angle threshold used to identify street corners (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to zero means the path is going straight through the intersection. A turn angle value around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle will vary depending on the angle at which the streets are crossing).

Returns:
- (list[float]): List of travel distances, in meters, to the next street corner or to the receiver.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StreetCornerAngles
#### crc_covlib.helper.streets.StreetCornerAngles
```python
def StreetCornerAngles(radioPath: StreetCanyonsRadioPath,
                       turnAngleThreshold_deg: float=20) -> list[float]
```
Gets the list of street corner angles from a street canyons radio path, that is, the list of street corner angles along the path where the turn angle is equal or above the specified threshold. Note that a street corner angle corresponds to 180 degrees minus what is referred to as the turn angle in this module. See FIGURE 3 of ITU-R P.1411-12 for a representation of the street corner angle.

Args:
- __radioPath__ (crc_covlib.helper.streets.StreetCanyonsRadioPath): A StreetCanyonsRadioPath object obtained from the GetStreetCanyonsRadioPaths() function.
- __turnAngleThreshold_deg__ (float): Turn angle threshold used to identify street corners (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to zero means the path is going straight through the intersection. A turn angle value around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle will vary depending on the angle at which the streets are crossing).
      
Returns:
-  (list[float]): List of street corner angles, in degrees, found along the specified street canyons radio path.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SortRadioPathsByStreetCornerCount
#### crc_covlib.helper.streets.SortRadioPathsByStreetCornerCount
```python
def SortRadioPathsByStreetCornerCount(radioPaths: list[StreetCanyonsRadioPath], simplifyPaths: bool,
                                      turnAngleThreshold_deg: float=20, tolerance_m: float=12
                                      ) -> list[StreetCanyonsRadioPath]
```
Sorts the specified street canyons radio paths ascendingly based on their number of turns at street corners. Radio paths having the same amount of turns at street corners are secondarily sorted ascendingly in order of travel distance.

Args:
- __radioPaths__ (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): List of StreetCanyonsRadioPath objects obtained from the GetStreetCanyonsRadioPaths() function.
- __simplifyPaths__ (bool): Indicates whether to simplify the radio paths before sorting and returning them. This is ususally desirable as it removes very small street sections that unnecessarily complexifies the path propagation-wise.
- __turnAngleThreshold_deg__ (float): Turn angle threshold used to identify street corners (degrees), with 0 <= turnAngleThreshold_deg <= 180. A turn angle of zero or close to zero means the path is going straight through the intersection. A turn angle value around 90 degrees is either a right-hand or a left-hand turn at the (the turn angle will vary depending on the angle at which the streets are crossing).
- __tolerance_m__ (float): Maximum allowed path geometry displacement (meters) when simplifying paths.

Returns:
- (list[crc_covlib.helper.streets.StreetCanyonsRadioPath]): Sorted list of possibly simplified street canyons radio paths.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ReceiverStreetOrientation
#### crc_covlib.helper.streets.ReceiverStreetOrientation
```python
def ReceiverStreetOrientation(graph: Union[StreetGraph,None], txLat: float, txLon: float,
                              rxLat: float, rxLon: float) -> float
```
Gets the receiver' street orientation with respect to the direct path between transmitter and receiver (degrees), from 0 to 90 degrees inclusively.

Args:
- __graph__ (crc_covlib.helper.streets.StreetGraph|None): A StreetGraph object obtained from GetStreetGraphFromBox() or GetStreetGraphFromPoints(). When set to None, a new graph object is produced internally based on the receiver location.
- __txLat__ (float): Transmitter latitude (degrees, EPSG:4326), with -90 <= txLat <= 90.
- __txLon__ (float): Transmitter longitude (degrees, EPSG:4326), with -180 <= txLon <= 180.
- __rxLat__ (float): Receiver latitude (degrees, EPSG:4326), with -90 <= rxLat <= 90.
- __rxLon__ (float): Receiver longitude (degrees, EPSG:4326), with -180 <= rxLon <= 180.

Returns:
- (float): The receiver' street orientation with respect to the direct path between transmitter and receiver (degrees), from 0 to 90 degrees inclusively.

[Back to top](#streets-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***