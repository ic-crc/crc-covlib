# itur_p835 helper module
Implementation of ITU-R P.835-6, Annex 1.

```python
from crc_covlib.helper import itur_p835 
```

- [ReferenceAtmosphere (Enum)](#referenceatmosphere)
- [Temperature](#temperature)
- [Pressure](#pressure)
- [WaterVapourDensity](#watervapourdensity)
- [DryPressure](#drypressure)
- [WaterVapourPressure](#watervapourpressure)

***

### ReferenceAtmosphere
#### crc_covlib.helper.itur_p835.ReferenceAtmosphere
```python
class ReferenceAtmosphere(enum.Enum):
    MEAN_ANNUAL_GLOBAL   = 1
    LOW_LATITUDE         = 2  # smaller than 22°
    MID_LATITUDE_SUMMER  = 3  # between 22° and 45°
    MID_LATITUDE_WINTER  = 4  # between 22° and 45°
    HIGH_LATITUDE_SUMMER = 5  # higher than 45°
    HIGH_LATITUDE_WINTER = 6  # higher than 45°

MAGRA = ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL # Mean Annual Global Reference Atmosphere
```
Enumerates available reference atmospheres.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Temperature
#### crc_covlib.helper.itur_p835.Temperature
```python
def Temperature(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float
```
ITU-R P.835-6, Annex 1\
Gets the atmospheric temperature (K) at geometric height h_km (km).

Args:
- __h_km__ (float): Geometric height (km), with 0 <= h_km <= 100.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.

Returns:
- __T__ (float): The atmospheric temperature (K) at geometric height h_km.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Pressure
#### crc_covlib.helper.itur_p835.Pressure
```python
def Pressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float
```
ITU-R P.835-6, Annex 1\
Gets the total atmospheric pressure (hPa) at geometric height h_km (km).

Args:
- __h_km__ (float): Geometric height (km), with 0 <= h_km <= 100.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.

Returns:
- __P__ (float): The total atmospheric pressure (hPa) at geometric height h_km.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WaterVapourDensity
#### crc_covlib.helper.itur_p835.WaterVapourDensity
```python
def WaterVapourDensity(h_km: float, refAtm: ReferenceAtmosphere=MAGRA,
                       rho0_gm3: float=7.5, h0_km: float=2) -> float
```
ITU-R P.835-6, Annex 1\
Gets the water-vapour density (g/m3) at geometric height h_km (km).

Args:
- __h_km__ (float): Geometric height (km), with 0 <= h_km <= 100.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).
- __h0_km__ (float): Scale height (km). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __ρ__ (float): The water-vapour density (g/m3) at geometric height h_km.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DryPressure
#### crc_covlib.helper.itur_p835.DryPressure
```python
def DryPressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float
```
ITU-R P.835-6, Annex 1\
Gets the dry atmospheric pressure (hPa) at geometric height h_km (km).

Args:
- __h_km__ (float): Geometric height (km), with 0 <= h_km <= 100.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.

Returns:
- __P<sub>dry</sub>__ (float): The dry atmospheric pressure (hPa) at geometric height h_km.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WaterVapourPressure
#### crc_covlib.helper.itur_p835.WaterVapourPressure
```python
def WaterVapourPressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA,
                        rho0_gm3: float=7.5, h0_km: float=2) -> float
```
ITU-R P.835-6, Annex 1\
Gets the water vapour pressure (hPa) at geometric height h_km (km).

Args:
- __h_km__ (float): Geometric height (km), with 0 <= h_km <= 100.
- __ref_atm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).
- __h0_km__ (float): Scale height (km). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __e__ (float): The water vapour pressure (hPa) at geometric height h_km.

[Back to top](#itur_p835-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***