# itur_p1511 helper module
Implementation of ITU-R P.1511-3 (partial).

```python
from crc_covlib.helper import itur_p1511
```

- [TopographicHeightAMSL](#topographicheightamsl)

***

### TopographicHeightAMSL
#### crc_covlib.helper.itur_p1511.TopographicHeightAMSL
```python
def TopographicHeightAMSL(lat: float, lon: float) -> float
```
ITU-R P.1511-3, Annex 1, Section 1.1\
Gets the topographic height of the surface of the Earth above mean sea level (m).

Args:
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
    
Returns:
- (float): The topographic height of the surface of the Earth above mean sea level (m).

[Back to top](#itur_p1511-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***