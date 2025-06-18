# itur_p528 helper module
Wrapper around NTIA's C++ implementation of ITU-R P.528-5.\
See https://github.com/NTIA/p528

```python
from crc_covlib.helper import itur_p528
```

- [Polarization (Enum)](#polarization)
- [PropagationMode (Enum)](#propagationmode)
- [ReturnCode (Enum)](#returncode)
- [WarningFlag (Enum)](#warningflag)
- [Results (Class)](#results)
- [FreeSpaceElevAngleToGreatCircleDistance](#freespaceelevangletogreatcircledistance)
- [BasicTransmissionLoss](#basictransmissionloss)
- [BasicTransmissionLossEx](#basictransmissionlossex)


***

### Polarization
#### crc_covlib.helper.itur_p528.Polarization
```python
class Polarization(enum.Enum):
    HORIZONTAL = 0
    VERTICAL   = 1
```

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PropagationMode
#### crc_covlib.helper.itur_p528.PropagationMode
```python
class PropagationMode(enum.Enum):
    NOT_SET       = 0
    LINE_OF_SIGHT = 1
    DIFFRATION    = 2
    TROPOSCATTER  = 3
```

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ReturnCode
#### crc_covlib.helper.itur_p528.ReturnCode
```python
class ReturnCode(enum.Enum):
    SUCCESS                        = 0
    ERROR_VALIDATION__D_KM         = 1
    ERROR_VALIDATION__H_1          = 2
    ERROR_VALIDATION__H_2          = 3
    ERROR_VALIDATION__TERM_GEO     = 4
    ERROR_VALIDATION__F_MHZ_LOW    = 5
    ERROR_VALIDATION__F_MHZ_HIGH   = 6
    ERROR_VALIDATION__PERCENT_LOW  = 7
    ERROR_VALIDATION__PERCENT_HIGH = 8
    ERROR_VALIDATION__POLARIZATION = 9
    ERROR_HEIGHT_AND_DISTANCE      = 10
    SUCCESS_WITH_WARNINGS          = 11
```
See https://github.com/NTIA/p528/blob/master/ERRORS_AND_WARNINGS.md for more details.

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WarningFlag
#### crc_covlib.helper.itur_p528.WarningFlag
```python
class WarningFlag(enum.IntEnum):
    WARNING__NO_WARNINGS        = 0x00
    WARNING__DFRAC_TROPO_REGION = 0x01
    WARNING__HEIGHT_LIMIT_H_1   = 0x02
    WARNING__HEIGHT_LIMIT_H_2   = 0x04
```
See https://github.com/NTIA/p528/blob/master/ERRORS_AND_WARNINGS.md for more details.

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Results
#### crc_covlib.helper.itur_p528.Results
```python
class Results()
```
Results from the BasicTransmissionLossEx() function.

Attributes:
- __A_dB__ (float): Basic transmission loss (dB).
- __retCode__ (crc_covlib.helper.itur_p528.ReturnCode): Return code.
- __warnings__ (int): Warning flags.
- __d_km__ (float): Great circle path distance (km). Could be slightly different than specified in input variable if within LOS region.
- __Afs_dB__ (float): Free space basic transmission loss (dB).
- __Aa_dB__ (float): Median atmospheric absorption loss (dB).
- __thetah1_rad__ (float): Elevation angle of the ray at the low terminal (rad).
- __propagMode__ (crc_covlib.helper.itur_p528.PropagationMode): Mode of propagation.

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FreeSpaceElevAngleToGreatCircleDistance
#### crc_covlib.helper.itur_p528.FreeSpaceElevAngleToGreatCircleDistance
```python
def FreeSpaceElevAngleToGreatCircleDistance(fsElevAngle_deg: float, hr1_m: float, hr2_m: float) -> float
```
ITU-R P.528-5, Annex 2, Section 1\
Computes the great-circle distance between two terminals, in km.

Args:
- __fsElevAngle_deg__ (float): Free space elevation angle of the low terminal to the high terminal (deg).
- __hr1_m__ (float): Height of the low terminal above mean sea level (m).
- __hr2_m__ (float): Height of the high terminal above mean sea level (m).

Returns:
- (float): Great-circle distance between the terminals (km).

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BasicTransmissionLoss
#### crc_covlib.helper.itur_p528.BasicTransmissionLoss
```python
def BasicTransmissionLoss(f_MHz: float, p: float, d_km: float, hr1_m: float, hr2_m: float,
                          Tpol: Polarization) -> float
```
ITU-R P.528-5, Annex 2, Section 3\
Computes the basic transmission loss, in dB.

Args:
- __f_MHz__ (float): Frequency (MHz), with 100 <= f_MHz <= 30000.
- __p__ (float): Time percentage (%), with 1 <= p <= 99.
- __d_km__ (float): Great-circle path distance between terminals (km), with 0 <= d_km.
- __hr1_m__ (float): Height of the low terminal above mean sea level (m), with 1.5 <= hr1_m <= 20000.
- __hr2_m__ (float): Height of the high terminal above mean sea level (m), with 1.5 <= hr2_m <= 20000, and with hr1_m <= hr2_m.
- __Tpol__ (crc_covlib.helper.itur_p528.Polarization): Parameter indicating either horizontal or vertical linear polarization.

Returns:
- __A__ (float): Basic transmission loss (dB). Returns 0 on error.

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BasicTransmissionLossEx
#### crc_covlib.helper.itur_p528.BasicTransmissionLossEx
```python
def BasicTransmissionLossEx(f_MHz: float, p: float, d_km: float, hr1_m: float, hr2_m: float,
                            Tpol: Polarization) -> Results
```
ITU-R P.528-5, Annex 2, Section 3\
Computes the basic transmission loss (extended results version), in dB.

Args:
- __f_MHz__ (float): Frequency (MHz), with 100 <= f_MHz <= 30000.
- __p__ (float): Time percentage (%), with 1 <= p <= 99.
- __d_km__ (float): Great-circle path distance between terminals (km), with 0 <= d_km.
- __hr1_m__ (float): Height of the low terminal above mean sea level (m), with 1.5 <= hr1_m <= 20000.
- __hr2_m__ (float): Height of the high terminal above mean sea level (m), with 1.5 <= hr2_m <= 20000, and with hr1_m <= hr2_m.
- __Tpol__ (crc_covlib.helper.itur_p528.Polarization): Parameter indicating either horizontal or vertical linear polarization.

Returns:
- (crc_covlib.helper.itur_p528.Results): Object of type Results containing the basic transmission loss value (dB) and other intermediate results. See the Results class definition for more details.

[Back to top](#itur_p528-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***