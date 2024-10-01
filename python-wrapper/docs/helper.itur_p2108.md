# itur_p2108 helper module
Implementation of ITU-R P.2108-1, Annex 1, Sections 3.2 and 3.3.

```python
from crc_covlib.helper import itur_p2108
```

- [TerrestrialPathClutterLoss](#terrestrialpathclutterloss)
- [EarthSpaceClutterLoss](#earthspaceclutterloss)

***

### TerrestrialPathClutterLoss
#### crc_covlib.helper.itur_p2108.TerrestrialPathClutterLoss
```python
def TerrestrialPathClutterLoss(f_GHz: float, d_km: float, loc_percent: float) -> float
```
ITU-R P.2108-1, Annex 1, Section 3.2\
Statistical clutter loss model for terrestrial paths that can be applied for urban and suburban clutter loss modelling provided terminal heights are well below the clutter height.

Args:
- __f_GHz__ (float): frequency (GHz), with 0.5 <= f_GHz <= 67
- __d_km__: distance (km), with 0.25 <= d_km
- __loc_percent__: percentage of locations (%), with 0 < loc_percent < 100
    
Returns:
- (float): clutter loss (dB)

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
- __f_GHz__ (float): frequency (GHz), with 0.5 <= f_GHz <= 67
- __elevAngle_deg__ (float): elevation angle (deg). The angle of the airborne platform or satellite as seen from the terminal, with 0 <= elevAngle_deg <= 90
- __loc_percent__ (float): percentage of locations (%), with 0 < loc_percent < 100
    
Returns:
- (float): clutter loss (dB)

[Back to top](#itur_p2108-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***