# itur_p837 helper module
Implementation of ITU-R P.837-7 (partial).

```python
from crc_covlib.helper import itur_p837
```

- [RainfallRate001](#rainfallrate001)

***

### RainfallRate001
#### crc_covlib.helper.itur_p837.RainfallRate001
```python
def RainfallRate001(lat: float, lon: float) -> float
```
ITU-R P.837-7\
Gets the annual rainfall rate exceeded for 0.01% of an average year (mm/hr). This function uses bilinear interpolation on the precalculated R001.txt file.

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
    
Returns:
- __R<sub>0.01</sub>__ (float): The annual rainfall rate exceeded for 0.01% of an average year (mm/hr).

[Back to top](#itur_p837-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***