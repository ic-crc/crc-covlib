# itur_p681 helper module
Implementation of ITU-R P.681-11 (partial).

```python
from crc_covlib.helper import itur_p681
```

- [ShadowingLevel (Enum)](#shadowinglevel)
- [RoadsideTreesShadowingFade](#roadsidetreesshadowingfade)
- [RoadsideTreesShadowingFadeEx](#roadsidetreesshadowingfadeex)
- [NonGsoRoadsideTreesShadowingUnavail](#nongsoroadsidetreesshadowingunavail)
- [FadeDurationDistribution](#fadedurationdistribution)
- [NonFadeDurationDistribution](#nonfadedurationdistribution)
- [BuildingBlockageProbability](#buildingblockageprobability)
- [StreetCanyonMaskingFunction](#streetcanyonmaskingfunction)
- [SingleWallMaskingFunction](#singlewallmaskingfunction)
- [StreetCrossingMaskingFunction](#streetcrossingmaskingfunction)
- [TJunctionMaskingFunction](#tjunctionmaskingfunction)
- [MountainousMultipathFadingDistribution](#mountainousmultipathfadingdistribution)
- [RoadsideTreesMultipathFadingDistribution](#roadsidetreesmultipathfadingdistribution)
- [ShadowingCrossCorrelationCoefficient](#shadowingcrosscorrelationcoefficient)
- [AvailabilityImprobability](#availabilityimprobability)

***

### ShadowingLevel
#### crc_covlib.helper.itur_p681.ShadowingLevel
```python
class ShadowingLevel(enum.Enum):
    MODERATE = 1
    EXTREME  = 2
```

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RoadsideTreesShadowingFade
#### crc_covlib.helper.itur_p681.RoadsideTreesShadowingFade
```python
def RoadsideTreesShadowingFade(f_GHz: float, theta_deg: float, p: float) -> float
```
ITU-R P.681-5, Annex 1, Section 4.1.1\
Estimates fade du to roadside tree-shadowing (dB).

"The predicted fade distributions apply for highways and rural roads where the overall aspect of the propagation path is, for the most part, orthogonal to the lines of roadside trees and utility poles and it is assumed that the dominant cause of LMSS signal fading is tree canopy shadowing".

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.8 <= f_GHz <= 20.
- __theta_deg__ (float): Path elevation angle to the satellite (degrees), with 7 <= theta_deg <= 60.
- __p__ (float): Percentage of distance travelled over which fade is exceeded (%), with 1 <= p <= 80.

Returns:
- __A__ (float): Fade exceeded for the specified percentage of distance travelled (dB).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RoadsideTreesShadowingFadeEx
#### crc_covlib.helper.itur_p681.RoadsideTreesShadowingFadeEx
```python
def RoadsideTreesShadowingFadeEx(f_GHz: float, theta_deg: float, p: int) -> float
```
ITU-R P.681-5, Annex 1, Section 4.1.1.1\
Extension of the roadside trees model for elevation angles greater than 60 degrees at frequencies of 1.6 GHz and 2.6 GHz.

Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz = 1.6 or f_GHz = 2.6.
- __theta_deg__ (float): Path elevation angle to the satellite (degrees), with 60 <= theta_deg <= 90.
- __p__ (int): Percentage of distance travelled over which fade is exceeded (%). p must be set to one of: 1, 5, 10, 15, 20 or 30.

Returns:
- __A__ (float): Fade exceeded for the specified percentage of distance travelled (dB).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### NonGsoRoadsideTreesShadowingUnavail
#### crc_covlib.helper.itur_p681.NonGsoRoadsideTreesShadowingUnavail
```python
def NonGsoRoadsideTreesShadowingUnavail(f_GHz: float, theta_list_deg: list[float],
                                        p_time_list: list[float], Gm_list_dBi: list[float],
                                        A_dB: float) -> float
```
ITU-R P.681-5, Annex 1, Section 4.1.1.2\
Estimates the percentage of unavailability due to roadside trees shadowing for non-geostationary (non-GSO) and mobile-satellite systems.

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.8 <= f_GHz <= 20.
- __theta_list_deg__ (list[float]): List of path elevation angles under which the terminal will see the satellite (degrees), with 7 <= theta_list_deg[i] <= 60 for any i.
- __p_time_list__ (list[float]): For each elevation angle in theta_list_deg, the percentage of time (%) for which the terminal will see the satellite at that angle, with 0 <= p_time_list[i] <= 100 for any i.
- __Gm_list_dBi__ (list[float]): For each elevation angle in theta_list_deg, the mobile terminal's antenna gain (dBi) at the corresponding elevation angle.
- __A_dB__ (float): Fade margin (dB).

Returns:
- __p_unavail__ (float): The total system unavailability (%), from 0 to 100.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FadeDurationDistribution
#### crc_covlib.helper.itur_p681.FadeDurationDistribution
```python
def FadeDurationDistribution(dd_m: float) -> float
```
ITU-R P.681-5, Annex 1, Section 4.1.2\
Estimates the probability that the distance fade duration exceeds the distance dd (m), under the condition that the attenuation exceeds 5 dB.

The model "is based on measurements at an elevation angle of 51 degrees and is applicable for [roads that exhibit] moderate to severe shadowing (percentage of optical shadowing between 55% and 90%)".

Args:
- __dd_m__ (float): Distance fade duration (meters), with 0.02 <= dd_m.

Returns:
- __p__ (float): Probability (%) that the distance fade duration exceeds the distance dd_m, under the condition that the attenuation exceeds 5 dB. Returned value will be between 0 and 100.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### NonFadeDurationDistribution
#### crc_covlib.helper.itur_p681.NonFadeDurationDistribution
```python
def NonFadeDurationDistribution(dd_m: float, shadowingLevel: ShadowingLevel) -> float
```
ITU-R P.681-5, Annex 1, Section 4.1.3\
Estimates the probability that a continuous non-fade distance duration exceeds the distance dd (m), given that the fade is smaller than a 5 dB threshold.

The model "is based on measurements at an elevation angle of 51 degrees and is applicable for [roads that exhibit] moderate to severe shadowing (percentage of optical shadowing between 55% and 90%)".

Args:
- __dd_m__ (float): Distance fade duration (meters).
- __shadowingLevel__ (crc_covlib.helper.itur_p681.ShadowingLevel): One of MODERATE (percentage of optical shadowing between 55% and 75%) or EXTREME (percentage of optical shadowing between 75% and 90%).

Returns:
- __p__ (float): Probability that a continuous non-fade distance duration exceeds the distance dd (m), given that the fade is smaller than a 5 dB threshold. Returned value will be between 0 and 100.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BuildingBlockageProbability
#### crc_covlib.helper.itur_p681.BuildingBlockageProbability
```python
def BuildingBlockageProbability(f_GHz: float, Cf: float, theta_deg: float, hm_m: float,
                                dm_m: float, hb_m: float, phi_deg: float) -> float
```
ITU-R P.681-11, Annex 1, Section 4.2\
Estimates the percentage probability of blockage due to the buildings (%).

Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz from about 0.8 to 20.
- __Cf__ (float): Required clearance as a fraction of the first Fresnel zone.
- __theta_deg__ (float): Elevation angle of the ray to the satellite above horizontal (degrees), with 0 < theta_deg < 90.
- __hm_m__ (float): Height of mobile above ground (meters).
- __dm_m__ (float): Distance of the mobile from the front of the buildings (meters).
- __hb_m__ (float): The most common (modal) building heigh (meters).
- __phi_deg__ (float): Azimuth angle of the ray relative to street direction (degrees), with 0 < phi_deg < 180.

Returns:
- __p__ (float): Percentage probability of blockage due to the buildings (%).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StreetCanyonMaskingFunction
#### crc_covlib.helper.itur_p681.StreetCanyonMaskingFunction
```python
def StreetCanyonMaskingFunction(theta_deg: float, h_m: float, w_m: float, phi_deg: float) -> bool
```
ITU-R P.681-11, Annex 1, Section 4.4\
Estimates whether a link can be completed for the street canyon scenario.

Args:
- __theta_deg__ (float): Path elevation angle, with 0 <= theta_deg <= 90.
- __h_m__ (float): Average building height (meters).
- __w_m__ (float): Street width (meters).
- __phi_deg__ (float): Street orientation with respect to the link (degrees), with -180 <= phi_deg <= 180.

Returns:
- (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SingleWallMaskingFunction
#### crc_covlib.helper.itur_p681.SingleWallMaskingFunction
```python
def SingleWallMaskingFunction(theta_deg: float, h_m: float, w_m: float, phi_deg: float) -> bool
```
ITU-R P.681-11, Annex 1, Section 4.4\
Estimates whether a link can be completed for the single wall scenario.

Args:
- __theta_deg__ (float): Path elevation angle, with 0 <= theta_deg <= 90.
- __h_m__ (float): Average building height (meters).
- __w_m__ (float): Street width (meters).
- __phi_deg__ (float): Street orientation with respect to the link (degrees), with -180 <= phi_deg <= 180.

Returns:
- (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StreetCrossingMaskingFunction
#### crc_covlib.helper.itur_p681.StreetCrossingMaskingFunction
```python
def StreetCrossingMaskingFunction(theta_deg: float, h_m: float, w1_m: float, w2_m: float,
                                  phi_deg: float) -> bool:
```
ITU-R P.681-11, Annex 1, Section 4.4\
Estimates whether a link can be completed for the street crossing scenario.

Args:
- __theta_deg__ (float): Path elevation angle, with 0 <= theta_deg <= 90.
- __h_m__ (float): Average building height (meters).
- __w1_m__ (float): Width of first street (meters).
- __w2_m__ (float): Width of second street (meters).
- __phi_deg__ (float): Street orientation with respect to the link (degrees), with -180 <= phi_deg <= 180.

Returns:
- (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TJunctionMaskingFunction
#### crc_covlib.helper.itur_p681.TJunctionMaskingFunction
```python
def TJunctionMaskingFunction(theta_deg: float, h_m: float, w1_m: float, w2_m: float,
                             phi_deg: float) -> bool
```
ITU-R P.681-11, Annex 1, Section 4.4\
Estimates whether a link can be completed for the T-junction scenario.

Args:
- __theta_deg__ (float): Path elevation angle, with 0 <= theta_deg <= 90.
- __h_m__ (float): Average building height (meters).
- __w1_m__ (float): Width of first street (meters).
- __w2_m__ (float): Width of second street (meters).
- __phi_deg__ (float): Street orientation with respect to the link (degrees), with -180 <= phi_deg <= 180.

Returns:
- (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MountainousMultipathFadingDistribution
#### crc_covlib.helper.itur_p681.MountainousMultipathFadingDistribution
```python
def MountainousMultipathFadingDistribution(f_GHz: float, theta_deg: float, A_dB: float,
                                           warningsOn: bool=True) -> float
```
ITU-R P.681-11, Annex 1, Section 5.1\
Distribution of fade depths due to multipath in mountainous terrain. The model is valid when the effect of shadowing is negligible.

Supported input values (from the recommendation's TABLE 3):
| f_GHz  | theta_deg | A_dB  |
|--------|-----------|-------|
| 0.87   | 30        | 2-7   |
| 1.5    | 30        | 2-8   |
| 0.87   | 45        | 2-4   |
| 1.5    | 45        | 2-5   |

Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz = 0.87 or f_GHz = 1.5.
- __theta_deg__ (float): Path elevation angle to the satellite (degrees), with theta_deg = 30 or theta_deg = 45.
- __A_dB__ (float): Fade exceeded (dB). See table above for ranges of valid values.
- __warningsOn__ (bool): Indicates whether to raise a warning when A_dB is out of validity range.

Returns:
- __p__ (float): Percentage of distance over which the fade is exceeded (%), from 1 to 10.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RoadsideTreesMultipathFadingDistribution
#### crc_covlib.helper.itur_p681.RoadsideTreesMultipathFadingDistribution
```python
def RoadsideTreesMultipathFadingDistribution(f_GHz: float, A_dB: float, warningsOn: bool=True
                                             ) -> float:
```
ITU-R P.681-11, Annex 1, Section 5.2\
Distribution of fade depths due to multipath in a roadsite trees environment. The model assumes negligible shadowing.

"Experiments conducted along tree-lined roads in the United States of America have shown that multipath fading is relatively insensitive to path elevation over the range of 30° to 60°".

Supported input values (from the recommendation's TABLE 4):
| f_GHz  | A_dB  |
|--------|-------|
| 0.87   | 1-4.5 |
| 1.5    | 1-6   |

Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz = 0.87 or f_GHz = 1.5.
- __A_dB__ (float): Fade exceeded (dB). See table above for ranges of valid values.
- __warningsOn__ (bool): Indicates whether to raise a warning when A_dB is out of validity range.

Returns:
- __p__ (float): Percentage of distance over which the fade is exceeded (%), from 1 to 50.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ShadowingCrossCorrelationCoefficient
#### crc_covlib.helper.itur_p681.ShadowingCrossCorrelationCoefficient
```python
def ShadowingCrossCorrelationCoefficient(delta_phi_deg: float, theta1_deg: float,
                                         theta2_deg: float, h_m: float, w_m: float,
                                         l_m: float) -> float
```
ITU-R P.681-11, Annex 1, Section 9.2.1\
Quantifies the cross-correlation coefficient between shadowing events in urban areas (using a
"street canyon" area geometry).

Args:
- __delta_phi_deg__ (float): Azimuth spacing between two separate satellite-to-mobile links in street canyons (degrees), with -180 <= delta_phi_deg <= 180.
- __theta1_deg__ (float): Satellite 1 elevation angle (degrees), with 0 < theta1_deg < 90.
- __theta2_deg__ (float): Satellite 2 elevation angle (degrees), with 0 < theta2_deg < 90 and with theta2_deg >= theta1_deg.
- __h_m__ (float): Average building height (meters).
- __w_m__ (float): Average street width (meters).
- __l_m__ (float): Length of street under consideration (meters). A large value is advised for this parameter, i.e. l_m >= 200.

Returns:
- __rho__ (float): The cross-correlation coefficient (no units).

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### AvailabilityImprobability
#### crc_covlib.helper.itur_p681.AvailabilityImprobability
```python
def AvailabilityImprobability(rho: float, p1: float, p2: float) -> float
```
ITU-R P.681-11, Annex 1, Section 9.2.1\
Computes the overall availability improbability after satellite diversity.

Args:
- __rho__ (float): Cross-correlation coefficient. May be obtained from the shadowing_cross_correlation_coefficient() function.
- __p1__ (float): Unavailability probability for the first link (%), with 0 <= p1 <= 100.
- __p2__ (float): Unavailability probability for the second link (%), with 0 <= p2 <= 100. For urban areas, p1 and p2 values may be computed using the building_blockage_probability() function.

Returns:
- __p0__ (float): The overall availability improbability after satellite diversity (%). The probability of availability will be 100-p0.

[Back to top](#itur_p681-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***
