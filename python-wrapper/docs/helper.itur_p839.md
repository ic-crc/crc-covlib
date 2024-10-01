# itur_p839 helper module
Implementation of ITU-R P.839-4.

```python
from crc_covlib.helper import itur_p839 
```

- [MeanAnnualZeroCelsiusIsothermHeight](#meanannualzerocelsiusisothermheight)
- [MeanAnnualRainHeight](#meanannualrainheight)

***

### MeanAnnualZeroCelsiusIsothermHeight
#### crc_covlib.helper.itur_p839.MeanAnnualZeroCelsiusIsothermHeight
```python
def MeanAnnualZeroCelsiusIsothermHeight(lat: float, lon: float) -> float
```
ITU-R P.839-4\
Gets the mean annual 0°C isotherm height (km above mean sea level).

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
    
Returns:
- __h<sub>0</sub>__ (float): The mean annual 0°C isotherm height (km above mean sea level).

[Back to top](#itur_p839-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanAnnualRainHeight
#### crc_covlib.helper.itur_p839.MeanAnnualRainHeight
```python
def MeanAnnualRainHeight(lat: float, lon: float) -> float
```
ITU-R P.839-4\
Gets the mean annual rain height (km above mean sea level).

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
    
Returns:
- __h<sub>R</sub>__ (float): The mean annual rain height (km above mean sea level).

[Back to top](#itur_p839-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***