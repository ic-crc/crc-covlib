# itur_p2108 helper module
Implementation of ITU-R P.2108-1.

```python
from crc_covlib.helper import itur_p2108
```

- [ClutterType (Enum)](#cluttertype)
- [GetDefaultRepresentativeHeight](#getdefaultrepresentativeheight)
- [HeightGainModelClutterLoss](#heightgainmodelclutterloss)
- [TerrestrialPathClutterLoss](#terrestrialpathclutterloss)
- [EarthSpaceClutterLoss](#earthspaceclutterloss)

***

### ClutterType
#### crc_covlib.helper.itur_p2108.ClutterType
```python
class ClutterType(enum.Enum):
    WATER_SEA          = 1
    OPEN_RURAL         = 2
    SUBURBAN           = 3
    URBAN_TREES_FOREST = 4
    DENSE_URBAN        = 5
```

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetDefaultRepresentativeHeight
#### crc_covlib.helper.itur_p2108.GetDefaultRepresentativeHeight
```python
def GetDefaultRepresentativeHeight(clut: ClutterType) -> float
```
ITU-R P.2108-1, Annex 1, Section 3.1\
Gets the default representative clutter height, as defined in the recommendation's Table 3.

Args:
- __clut__ (crc_covlib.helper.itur_p2108.ClutterType): One of the clutter types from the recommendation's Table 3.

Returns:
- (float): Representative clutter height (m).

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### HeightGainModelClutterLoss
#### crc_covlib.helper.itur_p2108.HeightGainModelClutterLoss
```python
def HeightGainModelClutterLoss(f_GHz: float, h_m: float, clut: ClutterType,
                               R_m: float|None=None, ws_m: float=27.0) -> float
```
ITU-R P.2108-1, Annex 1, Section 3.1\
"An additional loss, A<sub>h</sub>, is calculated which can be added to the basic transmission loss of a path calculated above the clutter, therefore basic transmission loss should be calculated to/from the height of the representative clutter height used. This model can be applied to both transmitting and receiving ends of the path."

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.03 <= f_GHz <= 3.
- __h_m__ (float): Antenna height (m), with 0 <= h_m.
- __clut__ (crc_covlib.helper.itur_p2108.ClutterType): One of the clutter types from the recommendation's Table 3.
- __R_m__ (float|None): Representative clutter height (m), with 0 < R_m. When R_m is set to None, the default representative clutter height for the specified clutter type is used.
- __ws_m__ (float): Street width (m).
    
Returns:
- __A<sub>h</sub>__ (float): Clutter loss (dB).

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TerrestrialPathClutterLoss
#### crc_covlib.helper.itur_p2108.TerrestrialPathClutterLoss
```python
def TerrestrialPathClutterLoss(f_GHz: float, d_km: float, loc_percent: float) -> float
```
ITU-R P.2108-1, Annex 1, Section 3.2\
Statistical clutter loss model for terrestrial paths that can be applied for urban and suburban clutter loss modelling provided terminal heights are well below the clutter height.

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.5 <= f_GHz <= 67.
- __d_km__: Distance (km), with 0.25 <= d_km.
- __loc_percent__: Percentage of locations (%), with 0 < loc_percent < 100.
    
Returns:
- (float): Clutter loss (dB).

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EarthSpaceClutterLoss
#### crc_covlib.helper.itur_p2108.EarthSpaceClutterLoss
```python
def EarthSpaceClutterLoss(f_GHz: float, elevAngle_deg: float, loc_percent: float) -> float
```
ITU-R P.2108-1, Annex 1, Section 3.3\
Statistical distribution of clutter loss where one end of the interference path is within man-made clutter, and the other is a satellite, aeroplane, or other platform above the surface of the Earth. This model is applicable to urban and suburban environments.

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.5 <= f_GHz <= 67.
- __elevAngle_deg__ (float): Elevation angle (deg). The angle of the airborne platform or satellite as seen from the terminal, with 0 <= elevAngle_deg <= 90.
- __loc_percent__ (float): Percentage of locations (%), with 0 < loc_percent < 100.
    
Returns:
- (float): Clutter loss (dB).

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***