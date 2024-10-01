# itur_p530 helper module
Implementation of ITU-R P.530-18 (partial).

```python
from crc_covlib.helper import itur_p530
```

- [GeoclimaticFactorK](#geoclimaticfactork)
- [DN75](#dn75)
- [AtmosphericAttenuation](#atmosphericattenuation)
- [FirstFresnelEllipsoidRadius](#firstfresnelellipsoidradius)
- [ApproximatedDiffractionLoss](#approximateddiffractionloss)
- [TimePeriod (enum)](#timeperiod)
- [SingleFrequencyFadingDistribution](#singlefrequencyfadingdistribution)
- [FadingDistribution](#fadingdistribution)
- [EnhancementDistribution](#enhancementdistribution)
- [InverseDistribution](#inversedistribution)
- [PathType (enum)](#pathtype)
- [AvgWorstMonthToShorterWorstPeriod](#avgworstmonthtoshorterworstperiod)
- [RainAttenuationLongTermStatistics](#rainattenuationlongtermstatistics)

***

### GeoclimaticFactorK
#### crc_covlib.helper.itur_p530.GeoclimaticFactorK
```python
def GeoclimaticFactorK(lat: float, lon: float) -> float
```
ITU-R P.530-18, Annex 1, Section 1.1\
Gets the geoclimatic factor K.

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.

Returns:
- __K__ (float): The geoclimatic factor K.

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DN75
#### crc_covlib.helper.itur_p530.DN75
```python
def DN75(lat: float, lon: float) -> float
```
ITU-R P.530-18, Annex 1, Section 1.1\
Gets parameter dN75, an empirical prediction of 0.1% of the average worst month refractivity increase with height over the lowest 75 m of the atmosphere from surface dewpoint data (N-units).

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.

Returns:
- __dN<sub>75</sub>__ (float): dN75, an empirical prediction of 0.1% of the average worst month refractivity increase with height over the lowest 75 m of the atmosphere from surface dewpoint data (N-units).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### AtmosphericAttenuation
#### crc_covlib.helper.itur_p530.AtmosphericAttenuation
```python
def AtmosphericAttenuation(d_km: float, f_GHz: float, P_hPa: float=1013.25,
                           T_K: float=288.15, rho_gm3: float=7.5) -> float
```
ITU-R P.530-18, Annex 1, Section 2.1\
Terrestrial path attenuation due to absorption by oxygen and water vapour (dB).

Args:
- __d_km__ (float): Path length (km), with 0 <= d_km.
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
- __P_hPa__ (float): Atmospheric pressure (hPa).
- __T_K__ (float): temperature (K).
- __rho_gm3__ (float): Water vapour density (g/m3).

Returns:
- (float): Terrestrial path attenuation due to absorption by oxygen and water vapour (dB).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FirstFresnelEllipsoidRadius
#### crc_covlib.helper.itur_p530.FirstFresnelEllipsoidRadius
```python
def FirstFresnelEllipsoidRadius(d_km: float, f_GHz: float, d1_km: float, d2_km: float) -> float
```
ITU-R P.530-18, Annex 1, Section 2.2.1\
Radius of the first Fresnel ellipsoid (m).

Args:
- __d_km__ (float): Path length (km), with 0 < d_km.
- __f_GHz__ (float): Frequency (GHz).
- __d1_km__ (float): Distance from the first terminal to the path obstruction (km).
- __d2_km__ (float): Distance from the second terminal to the path obstruction (km).

Returns:
- __F<sub>1</sub>__ (float): Radius of the first Fresnel ellipsoid (m).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ApproximatedDiffractionLoss
#### crc_covlib.helper.itur_p530.ApproximatedDiffractionLoss
```python
def ApproximatedDiffractionLoss(d_km: float, f_GHz: float, d1_km: float, d2_km: float,
                                h_m: float) -> float
```
ITU-R P.530-18, Annex 1, Section 2.2.1\
Approximation of the diffraction loss (dB) over average terrain (strictly valid for losses greater than about 15 dB).

Args:
- __d_km__ (float): Path length (km), with 0 < d_km.
- __f_GHz__ (float): Frequency (GHz).
- __d1_km__ (float): Distance from the first terminal to the path obstruction (km).
- __d2_km__ (float): Distance from the second terminal to the path obstruction (km).
- __h_m__ (float): Height difference (m) between most significant path blockage and the path trajectory (h_m is negative if the top of the obstruction of interest is above the virtual line-of-sight).

Returns:
- __A<sub>d</sub>__ (float): Approximation of the diffraction loss (dB) over average terrain (strictly valid for losses greater than about 15 dB).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TimePeriod
#### crc_covlib.helper.itur_p530.TimePeriod
```python
class TimePeriod(enum.Enum):
    AVG_WORST_MONTH = 1
    AVG_YEAR        = 2
```
Enumerates possible time periods for fading/enhancement distributions.

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SingleFrequencyFadingDistribution
#### crc_covlib.helper.itur_p530.SingleFrequencyFadingDistribution
```python
def SingleFrequencyFadingDistribution(A_dB: float, d_km: float, f_GHz: float, he_masl: float,
                                      hr_masl: float, ht_masl: float, lat: float, lon: float,
                                      timePeriod: TimePeriod) -> float
```
ITU-R P.530-18, Annex 1, Sections 2.3.1 and 2.3.4\
Calculates the percentage of time that fade depth A_dB is exceeded in the specified time period (%).

This implements a method for predicting the single-frequency (or narrow-band) fading distribution at large fade depths in the average worst month or average year in any part of the world. This method does not make use of the path profile and can be used for initial planning, licensing, or design purposes.

Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km, and can be set to zero for shorter paths.

Args:
- __A_dB__ (float): Fade depth (dB), with 0 <= A_dB.
- __d_km__ (float): Path length (km), with 0 < d_km.
- __f_GHz__ (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
- __he_masl__ (float): Emitter antenna height (meters above sea level).
- __hr_masl__ (float): Receiver antenna height (meters above sea level).
- __ht_masl__ (float): Mean terrain elevation along the path, excluding trees (meters above sea level).
- __lat__ (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
- __timePeriod__ (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or average year).

Returns:
- __p__ (float): The percentage of time that fade depth A_dB is exceeded in the specified time period (%).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FadingDistribution
#### crc_covlib.helper.itur_p530.FadingDistribution
```python
def FadingDistribution(A_dB: float, d_km: float, f_GHz: float, he_masl: float, hr_masl: float,
                       ht_masl: float, lat: float, lon: float, timePeriod: TimePeriod) -> float
```
ITU-R P.530-18, Annex 1, Sections 2.3.2 and 2.3.4\
Calculates the percentage of time that fade depth A_dB is exceeded in the specified time period (%).

This implementation is suitable for all fade depths and employs the method for large fade depths and an interpolation procedure for small fade depths. It combines the deep fading distribution given in section 2.3.1 and an empirical interpolation procedure for shallow fading down to 0 dB.

Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km, and can be set to zero for shorter paths.

Args:
- __A_dB__ (float): Fade depth (dB), with 0 <= A_dB.
- __d_km__ (float): Path length (km), with 0 < d_km.
- __f_GHz__ (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
- __he_masl__ (float): Emitter antenna height (meters above sea level).
- __hr_masl__ (float): Receiver antenna height (meters above sea level).
- __ht_masl__ (float): Mean terrain elevation along the path, excluding trees (meters above sea level).
- __lat__ (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
- __timePeriod__ (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or average year).
    
Returns:
- __p__ (float): The percentage of time that fade depth A_dB is exceeded in the specified time period (%).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EnhancementDistribution
#### crc_covlib.helper.itur_p530.EnhancementDistribution
```python
def EnhancementDistribution(E_dB: float, d_km: float, f_GHz: float, he_masl: float, hr_masl: float,
                            ht_masl: float, lat: float, lon: float, timePeriod: TimePeriod) -> float
```
ITU-R P.530-18, Annex 1, Sections 2.3.3 and 2.3.4\
Calculates the percentage of time that enhancement E_dB is not exceeded in the specified time period (%).

Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km, and can be set to zero for shorter paths.

Args:
- __E_dB__ (float): Enhancement (dB), with 0 <= E_dB.
- __d_km__ (float): Path length (km), with 0 < d_km.
- __f_GHz__ (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
- __he_masl__ (float): Emitter antenna height (meters above sea level).
- __hr_masl__ (float): Receiver antenna height (meters above sea level).
- __ht_masl__ (float): Mean terrain elevation along the path, excluding trees (meters above sea level).
- __lat__ (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
- __timePeriod__ (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or average year).

Returns:
- __p__ (float): The percentage of time that enhancement E_dB is not exceeded in the specified time period (%).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### InverseDistribution
#### crc_covlib.helper.itur_p530.InverseDistribution
```python
def InverseDistribution(distribFunc: Callable[..., float], p: float, *distribFuncArgs) -> float
```
Related to ITU-R P.530-18, Annex 1, Sections 2.3.1, 2.3.2, 2.3.3 and 2.3.4\
Calculates the fading (dB) exceeded for the specified time percentage, or the enhancement (dB) not exceeded for the specified percentage.

Note: Implemented algorithm is not part of the ITU recommendation, use with caution.

Args:
- __distribFunc__ (Callable[..., float]): A function, one of SingleFrequencyFadingDistribution, FadingDistribution or EnhancementDistribution.
- __p__ (float): Percentage of time (%).
- __distribFuncArgs__: Arguments for distribFunc, with the exception of A_dB or E_dB that should not be specified. 

Returns:
- (float): The fading (dB) exceeded for the specified time percentage, or the enhancement (dB) not exceeded for the specified percentage.

Example code:
```python
from crc_covlib.helper.itur_p530 import FadingDistribution, InverseDistribution, TimePeriod

A_dB = 10
d_km = 10
f_GHz = 29
he_masl = 100
hr_masl = 100
ht_masl = 50
lat = 51
lon = -53

p = FadingDistribution(A_dB, d_km, f_GHz, he_masl, hr_masl, ht_masl, lat, lon, TimePeriod.AVG_WORST_MONTH)
print(p) # 0.16938870177864995

A2_dB = InverseDistribution(FadingDistribution, p, d_km, f_GHz, he_masl, hr_masl, ht_masl, lat, lon, TimePeriod.AVG_WORST_MONTH)
print(A2_dB) # 9.999990463256836
```

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PathType
#### crc_covlib.helper.itur_p530.PathType
```python
class PathType(enum.Enum):
    RELATIVELY_FLAT = 1
    HILLY           = 2
    HILLY_LAND      = 3
```
Enumerates path types (to use with AvgWorstMonthToShorterWorstPeriod()).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### AvgWorstMonthToShorterWorstPeriod
#### crc_covlib.helper.itur_p530.AvgWorstMonthToShorterWorstPeriod
```python
def AvgWorstMonthToShorterWorstPeriod(pw: float, T_hours: float, pathType: PathType) -> float
```
ITU-R P.530-18, Annex 1, Sections 2.3.5\
Converts the percentage of time pw of exceeding a deep fade A in the average worst month to a percentage of time pws of exceeding the same deep fade during a shorter worst period of time T.

Args:
- __pw__ (float): The percentage of time of exceeding a deep fade A in the average worst month.
- __T_hours__ (float): Shorter than a month worst period of time (hours), with 1 <= T_hours < 720.
- __pathType__ (crc_covlib.helper.itur_p530.PathType): Type of path.

Returns:
- __p<sub>sw</sub>__ (float): Percentage of time of exceeding the deep fade A during the worst T hours.

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RainAttenuationLongTermStatistics
#### crc_covlib.helper.itur_p530.RainAttenuationLongTermStatistics
```python
def RainAttenuationLongTermStatistics(p: float, d_km: float, f_GHz: float,
                                      pathElevAngle_deg: float, polTiltAngle_deg: float,
                                      lat: float, lon: float) -> float
```
ITU-R P.530-18, Annex 1, Sections 2.4.1\
Calculates the rain attenuation over the specified path lengths, exceeded for p percent of
time, based on yearly statistics (dB).

Args:
- __p__ (float): Time percentage (%), with 0.001 <= p <= 1.
- __d_km__ (float): Path length (km), with 0 < d_km <= 60.
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 100.
- __pathElevAngle_deg__ (float): Path elevation angle (deg).
- __polTiltAngle_deg__ (float): Polarization tilt angle relative to the horizontal (deg). Use 45° for circular polarization.
- __lat__ (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the path location(degrees), with -180 <= lon <= 180.

Returns:
- __A<sub>p</sub>__ (float): Rain attenuation over the specified path lengths, exceeded for p percent of time, based on yearly statistics (dB).

[Back to top](#itur_p530-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***