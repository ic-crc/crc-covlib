# itur_p2109 helper module
Implementation of ITU-R P.2109-2.

```python
from crc_covlib.helper import itur_p2109
```

- [BuildingType  (Enum)](#buildingtype)
- [BuildingEntryLoss](#buildingentryloss)

***

### BuildingType
#### crc_covlib.helper.itur_p2109.BuildingType
```python
class BuildingType(enum.Enum):
    TRADITIONAL         = 1
    THERMALLY_EFFICIENT = 2
```
Enumerates available building types.

[Back to top](#itur_p2109-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BuildingEntryLoss
#### crc_covlib.helper.itur_p2109.BuildingEntryLoss
```python
def BuildingEntryLoss(f_GHz: float, prob_percent: float, bldgType: BuildingType,
                      elevAngle_deg: float) -> float
```
ITU-R P.2109-2, Annex 1, Section 3\
Building entry loss (dB).

Args:
- __f_GHz__ (float): Frequency (GHz), with 0.08 <= f_GHz <= 100.
- __prob_percent__ (float): The probability with which the loss is not exceeded (%), with 0 < prob_percent < 100.
- __bldgType__ (crc_covlib.helper.itur_p2109.BuildingType): Building type (traditional or thermally efficient).
- __elevAngle_deg__ (float): Elevation angle of the path at the building façade (degrees above/below the horizontal), with -90 < elevAngle_deg < 90.

Returns:
- (float): Building entry loss (dB).

[Back to top](#itur_p2109-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***