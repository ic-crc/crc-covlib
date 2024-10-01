# itur_p453 helper module
Implementation of ITU-R P.453-14 (partial).

```python
from crc_covlib.helper import itur_p453
```

- [MedianAnnualNwet](#medianannualnwet)

***

### MedianAnnualNwet
#### crc_covlib.helper.itur_p453.MedianAnnualNwet
```python
def MedianAnnualNwet(lat: float, lon: float) -> float
```
ITU-R P.453-14\
Gets the median value of the wet term of the surface refractivity exceeded for the average year (ppm).

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
    
Returns:
- __N<sub>wet</sub>__ (float): The median value of the wet term of the surface refractivity exceeded for the average year (ppm).

[Back to top](#itur_p453-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***