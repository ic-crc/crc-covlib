# itur_p2001 helper module
Implementation of Section 3.7 and Attachment H from ITU-R P.2001-5.

```python
from crc_covlib.helper import itur_p2001
```

- [IsLOS](#islos)
- [ElevationAngles](#elevationangles)
- [PathLength](#pathlength)
- [Bearing](#bearing)
- [IntermediatePathPoint](#intermediatepathpoint)

***

### IsLOS
#### crc_covlib.helper.itur_p2001.IsLOS
```python
def IsLOS(latt: float, lont: float, ht_mamsl: float, latr: float, lonr: float, hr_mamsl: float,
          dProfile_km: ArrayLike, hProfile_mamsl: ArrayLike) -> bool
```
ITU-R P.2001-5, Annex, Section 3.7\
Determines whether a path is line-of-sight or non-line-of-sight under median refractivity conditions.

Args:
- __latt__ (float): Latitude of transmitter (degrees), with -90 <= lat <= 90.
- __lont__ (float): Longitude of transmitter (degrees), with -180 <= lon <= 180.
- __ht_mamsl__ (float): Transmitter height (meters above mean sea level).
- __latr__ (float): Latitude of receiver (degrees), with -90 <= lat <= 90.
- __lonr__ (float): Longitude of receiver (degrees), with -180 <= lon <= 180.
- __hr_mamsl__ (float): Receiver height (meters above mean sea level).
- __dProfile_km__ (numpy.typing.ArrayLike): Great-cirlcle distance from transmitter (km) profile.
- __hProfile_mamsl__ (numpy.typing.ArrayLike): Terrain height profile (meters above mean sea level) from the transmitter to the receiver. hProfile and dProfile must have the same number of values.
    
Returns:
- (bool): True when the path is line-of-sight, False otherwise.

[Back to top](#itur_p2001-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ElevationAngles
#### crc_covlib.helper.itur_p2001.ElevationAngles
```python
def ElevationAngles(latt: float, lont: float, ht_mamsl: float,
                    latr: float, lonr: float, hr_mamsl: float,
                    dProfile_km: ArrayLike, hProfile_mamsl: ArrayLike,
                    useP1812Variation: bool=True) -> tuple[float, float]
```
ITU-R P.2001-5, Annex, Section 3.7\
Calculates the transmitter and receiver elevation angles (degrees) under median refractivity conditions.

Args:
- __latt__ (float): Latitude of transmitter (degrees), with -90 <= lat <= 90.
- __lont__ (float): Longitude of transmitter (degrees), with -180 <= lon <= 180.
- __ht_mamsl__ (float): Transmitter height (meters above mean sea level).
- __latr__ (float): Latitude of receiver (degrees), with -90 <= lat <= 90.
- __lonr__ (float): Longitude of receiver (degrees), with -180 <= lon <= 180.
- __hr_mamsl__ (float): Receiver height (meters above mean sea level).
- __dProfile_km__ (numpy.typing.ArrayLike): Great-cirlcle distance from transmitter (km) profile.
- __hProfile_mamsl__ (numpy.typing.ArrayLike): Terrain height profile (meters above mean sea level) from the transmitter to the receiver. hProfile and dProfile must have the same number of values.
- __useP1812Variation__ (bool): The formula provided in ITU-R P.2001 to calculate elevation angles does not appear to be suitable for short distances. When set to True, the formula from ITU-R P.1812-7 is used instead.
    
Returns:
- (float): Transmitter elevation angle (degrees). 0°=horizon, +90°=zenith, -90°=nadir.
- (float): Receiver elevation angle (degrees). 0°=horizon, +90°=zenith, -90°=nadir.

[Back to top](#itur_p2001-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PathLength
#### crc_covlib.helper.itur_p2001.PathLength
```python
def PathLength(lat0: float, lon0: float, lat1: float, lon1: float) -> float
```
ITU-R P.2001-5, Attachment H (H.2)\
Calculates the great-circle path length between two points on the surface of the Earth (km).
    
Args:
- __lat0__ (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
- __lon0__ (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
- __lat1__ (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
- __lon1__ (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.

Returns:
- (float): The great-circle path length between two points on the surface of the Earth (km).

[Back to top](#itur_p2001-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Bearing
#### crc_covlib.helper.itur_p2001.Bearing
```python
def Bearing(lat0: float, lon0: float, lat1: float, lon1: float) -> float
```
ITU-R P.2001-5, Attachment H (H.2)\
Calculates the bearing of the great-circle path between two points on the surface of the Earth, in degrees.

Args:
- __lat0__ (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
- __lon0__ (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
- __lat1__ (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
- __lon1__ (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.

Returns:
- (float): The bearing of the great-circle path from the first point towards the second point (degrees), with 0 <= bearing <= 360. Corresponds to the angle between due north at the first point eastwards (clockwise) to the direction of the path.

[Back to top](#itur_p2001-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IntermediatePathPoint
#### crc_covlib.helper.itur_p2001.IntermediatePathPoint
```python
def IntermediatePathPoint(lat0: float, lon0: float, lat1: float, lon1: float,
                          dist_km: float) -> tuple[float, float]
```
ITU-R P.2001-5, Attachment H (H.3)\
Calculates the latitude and longitude of any point along a great-circle path on the surface of the Earth.

Args:
- __lat0__ (float): Latitude at the first end of the path (degrees), with -90 <= lat0 <= 90.
- __lon0__ (float): Longitude at the first end of the path (degrees), with -180 <= lon0 <= 180.
- __lat1__ (float): Latitude at the second end of the path (degrees), with -90 <= lat1 <= 90.
- __lon1__ (float): Longitude at the second end of the path (degrees), with -180 <= lon1 <= 180.
- __dist_km__ (float): Great-circle distance of an intermediate point from the first end of the path (km).

Returns:
- __lat__ (float): Latitude of the intermediate point along the great-circle path (degrees).
- __lon__ (float): Longitude of the intermediate point along the great-circle path(degrees).

[Back to top](#itur_p2001-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***