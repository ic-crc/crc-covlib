# itur_p838 helper module
Implementation of ITU-R P.838-3.

```python
from crc_covlib.helper import itur_p838
```

- [RainAttenuation](#rainattenuation)
- [Coefficients](#coefficients)

***

### RainAttenuation
#### crc_covlib.helper.itur_p838.RainAttenuation
```python
def RainAttenuation(f_GHz: float, rainRate_mmhr: float, pathElevAngle_deg: float,
                    polTiltAngle_deg: float) -> float
```
ITU-R P.838-3\
Attenuation due to rain (dB/km).

Args:
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
- __rainRate_mmhr__ (float): Rain rate (mm/hr).
- __pathElevAngle_deg__ (float): Path elevation angle (deg).
- __polTiltAngle_deg__ (float): Polarization tilt angle relative to the horizontal (deg). Use 45° for circular polarization.

Returns:
- __γ<sub>R</sub>__ (float): Attenuation due to rain (dB/km).

[Back to top](#itur_p838-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Coefficients
#### crc_covlib.helper.itur_p838.Coefficients
```python
def Coefficients(f_GHz: float, pathElevAngle_deg: float, polTiltAngle_deg: float) -> tuple[float, float]
```
ITU-R P.838-3\
Gets the coefficients k and α from equations (4) and (5).

Args:
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
- __pathElevAngle_deg__ (float): Path elevation angle (deg).
- __polTiltAngle_deg__ (float): Polarization tilt angle relative to the horizontal (deg). Use 45° for circular polarization.

Returns:
- __k__ (float): Coefficient k.
- __α__ (float): Coefficient α.

[Back to top](#itur_p838-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***