# itur_p1411 helper module
Implementation of ITU-R P.1411-12 (partial).

```python
from crc_covlib.helper import itur_p1411
```

- [EnvironmentA (Enum)](#environmenta)
- [EnvironmentB (Enum)](#environmentb)
- [EnvironmentC (Enum)](#environmentc)
- [EnvironmentD (Enum)](#environmentd)
- [BuildingsLayout (Enum)](#buildingslayout)
- [PathType (Enum)](#pathtype)
- [WarningFlag (Enum)](#warningflag)
- [SiteGeneralWithinStreetCanyons](#sitegeneralwithinstreetcanyons)
- [SiteGeneralOverRoofTops](#sitegeneraloverrooftops)
- [SiteGeneralNearStreetLevel](#sitegeneralnearstreetlevel)
- [SiteSpecificWithinStreetCanyonsUHFLoS](#sitespecificwithinstreetcanyonsuhflos)
- [SiteSpecificWithinStreetCanyonsSHFLoS](#sitespecificwithinstreetcanyonsshflos)
- [SiteSpecificWithinStreetCanyonsEHFLoS](#sitespecificwithinstreetcanyonsehflos)
- [SiteSpecificWithinStreetCanyonsUHFNonLoS](#sitespecificwithinstreetcanyonsuhfnonlos)
- [SiteSpecificWithinStreetCanyonsSHFNonLoS](#sitespecificwithinstreetcanyonsshfnonlos)
- [SiteSpecificOverRoofTopsUrban](#sitespecificoverrooftopsurban)
- [SiteSpecificOverRoofTopsSuburban](#sitespecificoverrooftopssuburban)
- [SiteSpecificNearStreetLevelUrban](#sitespecificnearstreetlevelurban)
- [SiteSpecificNearStreetLevelResidential](#sitespecificnearstreetlevelresidential)

***

### EnvironmentA
#### crc_covlib.helper.itur_p1411.EnvironmentA
```python
class EnvironmentA(enum.Enum):
    URBAN_HIGH_RISE            = 10
    URBAN_LOW_RISE_OR_SUBURBAN = 11
    RESIDENTIAL                = 12
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EnvironmentB
#### crc_covlib.helper.itur_p1411.EnvironmentB
```python
class EnvironmentB(enum.Enum):
    SUBURBAN              = 20
    URBAN                 = 21
    DENSE_URBAN_HIGH_RISE = 22
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EnvironmentC
#### crc_covlib.helper.itur_p1411.EnvironmentC
```python
class EnvironmentC(enum.Enum):
    URBAN       = 30
    RESIDENTIAL = 31
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EnvironmentD
#### crc_covlib.helper.itur_p1411.EnvironmentD
```python
class EnvironmentD(enum.Enum):
    MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE = 40
    METROPOLITAN_CENTRE                 = 41
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BuildingsLayout
#### crc_covlib.helper.itur_p1411.BuildingsLayout
```python
class BuildingsLayout(enum.Enum):
    WEDGE_SHAPED    = 1
    CHAMFERED_SHAPE = 2
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PathType
#### crc_covlib.helper.itur_p1411.PathType
```python
class PathType(enum.Enum):
    LOS  = 1
    NLOS = 2
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WarningFlag
#### crc_covlib.helper.itur_p1411.WarningFlag
```python
class WarningFlag(enum.IntEnum):
    NO_WARNINGS       = 0x00
    FREQ_OUT_OF_RANGE = 0x01
    DIST_OUT_OF_RANGE = 0x02
```

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteGeneralWithinStreetCanyons
#### crc_covlib.helper.itur_p1411.SiteGeneralWithinStreetCanyons
```python
def SiteGeneralWithinStreetCanyons(f_GHz: float, d_m: float, env: EnvironmentA, path: PathType,
                                   addGaussianRandomVar: bool) -> tuple[float, int]
```
ITU-R P.1411-12, Annex 1, Section 4.1.1\
"This site-general model is applicable to situations where both the transmitting and receiving stations are located below-rooftop, regardless of their antenna heights."

Supported configurations (from recommendation's TABLE 4):
| f_GHz      | d_m        | env                          | path    |
|------------|------------|------------------------------|---------|
| 0.8-82     | 5-660      | URBAN_HIGH_RISE              | LOS     |
| 0.8-82     | 5-660      | URBAN_LOW_RISE_OR_SUBURBAN   | LOS     |
| 0.8-82     | 30-715     | URBAN_HIGH_RISE              | NLOS    |
| 10-73      | 30-250     | URBAN_LOW_RISE_OR_SUBURBAN   | NLOS    |
| 0.8-73     | 30-170     | RESIDENTIAL                  | NLOS    |

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 82 generally, but see above table for additional constraints.
- __d_m__ (float): 3D direct distance between the transmitting and receiving stations (m), with 5 <= d_m <= 715 generally, but see above table for additional constraints.
- __env__ (crc_covlib.helper.itur_p1411.EnvironmentA): One of URBAN_HIGH_RISE, URBAN_LOW_RISE_OR_SUBURBAN or RESIDENTIAL.
- __path__ (crc_covlib.helper.itur_p1411.PathType): One of LOS or NLOS. Indicates whether the path is line-of-sight or non-line-of-sight.
- __addGaussianRandomVar__ (bool): When set to True, a gaussian random variable is added to the median basic transmission loss. Use this option for Monte Carlo simulations for example.

Returns:
- __L<sub>b</sub>__ (float): Basic transmission loss (dB). When addGaussianRandomVar is set to False, L<sub>b</sub> is the median basic transmission loss.
- __warnings__ (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteGeneralOverRoofTops
#### crc_covlib.helper.itur_p1411.SiteGeneralOverRoofTops
```python
def SiteGeneralOverRoofTops(f_GHz: float, d_m: float, env: EnvironmentA, path: PathType,
                            addGaussianRandomVar: bool) -> tuple[float, int]
```
ITU-R P.1411-12, Annex 1, Section 4.2.1\
"This site-general model is applicable to situations where one of the stations is located above rooftop and the other station is located below-rooftop, regardless of their antenna heights."

Supported configurations (from from recommendation's TABLE 8):
| f_GHz      | d_m        | env                         | path    |
|------------|------------|-----------------------------|---------|
| 2.2-73     | 55-1200    | URBAN_HIGH_RISE,            | LOS     |
| 2.2-73     | 55-1200    | URBAN_LOW_RISE_OR_SUBURBAN  | LOS     |
| 2.2-66.5   | 260-1200   | URBAN_HIGH_RISE             | NLOS    |

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 2.2 <= f_GHz <= 73 generally, but see above table for additional constraints.
- __d_m__ (float): 3D direct distance between the transmitting and receiving stations (m), with 55 <= d_m <= 1200 generally, but see above table for additional constraints.
- __env__ (crc_covlib.helper.itur_p1411.EnvironmentA): One of URBAN_HIGH_RISE or URBAN_LOW_RISE_OR_SUBURBAN.
- __path__ (crc_covlib.helper.itur_p1411.PathType): One of LOS or NLOS. Indicates whether the path is line-of-sight or non-line-of-sight.
- __addGaussianRandomVar__ (bool): When set to True, a gaussian random variable is added to the median basic transmission loss. Use this option for Monte Carlo simulations for example.

Returns:
- __L<sub>b</sub>__ (float): Basic transmission loss (dB). When addGaussianRandomVar is set to False, L<sub>b</sub> is the median basic transmission loss.
- __warnings__ (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteGeneralNearStreetLevel
#### crc_covlib.helper.itur_p1411.SiteGeneralNearStreetLevel
```python
def SiteGeneralNearStreetLevel(f_GHz: float, d_m: float, p: float, env: EnvironmentB,
                               w_m: float=20.0) -> tuple[float, int]
```
ITU-R P.1411-12, Annex 1, Section 4.3.1\
"[This site-general model is] recommended for propagation between low-height terminals where both terminal antenna heights are near street level well below roof-top height, but are otherwise unspecified."

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.3 <= f_GHz <= 3.
- __d_m__ (float): Distance between the transmitting and receiving stations (m), with 0 < d_m <= 3000.
- __p__ (float): Location percentage (%), with 0 < p < 100.
- __env__ (crc_covlib.helper.itur_p1411.EnvironmentB): One of SUBURBAN, URBAN or DENSE_URBAN_HIGH_RISE.
- __w_m__ (float): Transition region width between line-of-sight and non-line-of-sight (m).

Returns:
- __L__ (float): Basic transmission loss (dB).
- __warnings__ (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificWithinStreetCanyonsUHFLoS
#### crc_covlib.helper.itur_p1411.SiteSpecificWithinStreetCanyonsUHFLoS
```python
def SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz: float, d_m: float, h1_m: float, h2_m: float
                                          ) -> tuple[float, float, float]
```
ITU-R P.1411-12, Annex 1, Section 4.1.2, UHF propagation\
Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban environment (street canyons context) in the UHF frequency range.

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.3 <= f_GHz <= 3.
- __d_m__ (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
- __h1_m__ (float): Station 1 antenna height (m), with 0 < h1_m.
- __h2_m__ (float): Station 2 antenna height (m), with 0 < h2_m.

Returns:
- __L<sub>LoS_m</sub>__ (float): Median basic transmission loss (dB).
- __L<sub>LoS_l</sub>__ (float): Lower bound of the basic transmission loss (dB).
- __L<sub>LoS_u</sub>__ (float): Upper bound of the basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificWithinStreetCanyonsSHFLoS
#### crc_covlib.helper.itur_p1411.SiteSpecificWithinStreetCanyonsSHFLoS
```python
def SiteSpecificWithinStreetCanyonsSHFLoS(f_GHz: float, d_m: float, h1_m: float, h2_m: float,
                                          hs_m: float) -> tuple[float, float, float]
```
ITU-R P.1411-12, Annex 1, Section 4.1.2, SHF propagation up to 15 GHz\
Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban environment (street canyons context) in the SHF frequency range.
    
Args:
- __f_GHz__ (float): Operating frequency (GHz), with 3 <= f_GHz <= 15.
- __d_m__ (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
- __h1_m__ (float): Station 1 antenna height (m), with 0 < h1_m.
- __h2_m__ (float): Station 2 antenna height (m), with 0 < h2_m.
- __hs_m__ (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23 to 1.6 meters.

Returns:
- __L<sub>LoS_m</sub>__ (float): Median basic transmission loss (dB).
- __L<sub>LoS_l</sub>__ (float): Lower bound of the basic transmission loss (dB).
- __L<sub>LoS_u</sub>__ (float): Upper bound of the basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificWithinStreetCanyonsEHFLoS
#### crc_covlib.helper.itur_p1411.SiteSpecificWithinStreetCanyonsEHFLoS
```python
def SiteSpecificWithinStreetCanyonsEHFLoS(f_GHz: float, d_m: float, n: float, P_hPa: float=1013.25,
                                          T_K: float=288.15, rho_gm3: float=7.5) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.1.2, Millimetre-wave propagation\
Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban environment (street canyons context) in the EHF frequency range (millimetre-wave).

The computed loss includes attenuation by atmospheric gases but does not include attenuation due to rain. The RainAttenuationLongTermStatistics() function from itur_p530 may be used to calculate this value when required.

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 10 <= f_GHz <= 100.
- __d_m__ (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
- __n__ (float): Basic transmission loss exponent. The recommendations's TABLES 7 gives values ranging from 1.9 to 2.21.
- __P_hPa__ (float): Atmospheric pressure (hPa), for estimating the attenuation by atmospheric gases.
- __T_K__ (float): Temperature (K), for estimating the attenuation by atmospheric gases.
- __rho_gm3__ (float): Water vapour density (g/m3), for estimating the attenuation by atmospheric gases.

Returns:
- __L<sub>LoS</sub>__ (float): Basic transmission loss (dB). Does not include rain attenuation.

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificWithinStreetCanyonsUHFNonLoS
#### crc_covlib.helper.itur_p1411.SiteSpecificWithinStreetCanyonsUHFNonLoS
```python
def SiteSpecificWithinStreetCanyonsUHFNonLoS(f_GHz: float, x1_m: float, x2_m: float, w1_m: float,
                                             w2_m: float, alpha_rad: float) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.1.3.1\
Site-specific model that estimates the basic transmission loss (dB) for a non-line-of-sight urban environment (street canyons context) in the UHF frequency range. The model considers a non-line-of-sight situation "where the diffracted and reflected waves at the corners of the street crossings have to be considered."

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 2.
- __x1_m__ (float): Distance from station 1 to street crossing (m), with 0 < x1_m.
- __x2_m__ (float): Distance from station 2 to street crossing (m), with 0 < x2_m.
- __w1_m__ (float): Street width at the position of the station 1 (m), with 0 < w1_m.
- __w2_m__ (float): Street width at the position of the station 2 (m), with 0 < w2_m.
- __alpha_rad__ (float): Corner angle (rad), with 0.6 < alpha_rad < pi.

Returns:
- __L<sub>NLoS2</sub>__ (float): Basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificWithinStreetCanyonsSHFNonLoS
#### crc_covlib.helper.itur_p1411.SiteSpecificWithinStreetCanyonsSHFNonLoS
```python
def SiteSpecificWithinStreetCanyonsSHFNonLoS(f_GHz: float, x1_m: float, x2_m: float, w1_m: float,
                                             h1_m: float, h2_m: float, hs_m: float, n: float,
                                             env: EnvironmentC,
                                             bldgLayout: BuildingsLayout=BuildingsLayout.WEDGE_SHAPED,
                                             P_hPa: float=1013.25, T_K: float=288.15,
                                             rho_gm3: float=7.5) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.1.3.2\
Site-specific model that estimates the basic transmission loss (dB) for a non-line-of-sight urban or residential environment (street canyons context) in the SHF frequency range. The model considers a non-line-of-sight situation "where the diffracted and reflected waves at the corners of the street crossings have to be considered." It assumes street corner angles of pi/2 radians (90 degrees).

For frequencies from 10 GHz and up, the losses in the line-of-sight region do not include the attenuation due to rain (see eq.(13)). The RainAttenuationLongTermStatistics() function from itur_p530 may be used to calculate this value when required.

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 2 <= f_GHz <= 38.
- __x1_m__ (float): Distance from station 1 to street crossing (m), with 20 < x1_m.
- __x2_m__ (float): Distance from station 2 to street crossing (m), with 0 <= x2_m.
- __w1_m__ (float): Street width at the position of the station 1 (m), with 0 < w1_m.
- __h1_m__ (float): Station 1 antenna height (m), with 0 < h1_m.
- __h2_m__ (float): Station 2 antenna height (m), with 0 < h2_m.
- __hs_m__ (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23 to 1.6 meters.
- __n__ (float): Basic transmission loss exponent. The recommendations's TABLES 7 gives values ranging from 1.9 to 2.21.
- __env__ (crc_covlib.helper.itur_p1411.EnvironmentC): URBAN or RESIDENTIAL.
- __bldgLayout__ (crc_covlib.helper.itur_p1411.BuildingsLayout): WEDGE_SHAPED or CHAMFERED_SHAPE. See recommendatinon's FIGURE 5. Only applies when env is set to URBAN.
- __P_hPa__ (float): Atmospheric pressure (hPa), for estimating the attenuation by atmospheric gases.
- __T_K__ (float): Temperature (K), for estimating the attenuation by atmospheric gases.
- __rho_gm3__ (float): Water vapour density (g/m3), for estimating the attenuation by atmospheric gases.

Returns:
- __L<sub>NLoS2</sub>__ (float): Basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificOverRoofTopsUrban
#### crc_covlib.helper.itur_p1411.SiteSpecificOverRoofTopsUrban
```python
def SiteSpecificOverRoofTopsUrban(f_GHz: float, d_m: float, h1_m: float, h2_m: float, hr_m: float,
                                  l_m: float, b_m: float, w2_m: float, phi_deg: float,
                                  env: EnvironmentD) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.2.2.1 & FIGURE 2\
Site-specific model that uses a multi-screen diffraction model to estimate the basic transmission loss (dB) in an urban environment. "The multi-screen diffraction model ...is valid if the roof-tops are all about the same height". It assumes "the roof-top heights differ only by an amount less than the first Fresnel-zone radius over [the path length]". The model includes "free-space basic transmission loss, ...the diffraction loss from roof-top to street ...and the reduction due to multiple screen diffraction past rows of buildings".

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 26 in general, but with 2 <= f_GHz <= 16 for h1_m < hr_m and w2_m < 10.
- __d_m__ (float): Path length (m), with 20 <= d_m <= 5000.
- __h1_m__ (float): Station 1 antenna height (m), with 4 <= h1_m <= 55.
- __h2_m__ (float): Station 2 antenna height (m), with 1 <= h2_m <= 3.
- __hr_m__ (float): Average height of buildings (m), with h2_m < hr_m. 
- __l_m__ (float): Length of the path covered by buildings (m), with 0 <= l_m. When l_m is set to zero, only the free-space basic transmission loss component is computed and returned (using f_GHz and d_m while other parameters are ignored).
- __b_m__ (float): Average building separation (m), with 0 < b_m.
- __w2_m__ (float): Street width (m) at station 2's location, with 0 < w2_m.
- __phi_deg__ (float): Street orientation with respect to the direct path (deg), with 0 <= phi_deg <= 90 (i.e. for a street that is perpendicular to the direct path, phi_deg is 90 deg).
- __env__ (crc_covlib.helper.itur_p1411.EnvironmentD): MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE or METROPOLITAN_CENTRE. Only used when f_GHz <= 2.

Returns:
- __L<sub>NLoS1</sub>__ (float): Basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificOverRoofTopsSuburban
#### crc_covlib.helper.itur_p1411.SiteSpecificOverRoofTopsSuburban
```python
def SiteSpecificOverRoofTopsSuburban(f_GHz: float, d_m: float, h1_m: float, h2_m: float,
                                     hr_m: float, w2_m: float, phi_deg: float) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.2.2.2\
Site-specific model that estimates the basic transmission loss (dB) for a suburban environment. The estimated loss "can be divided into three regions in terms of the dominant arrival waves at station 2. These are the direct wave, reflected wave, and diffracted wave dominant regions."

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 38.
- __d_m (float)__: Path length (m), with 10 <= d_m <= 5000.
- __h1_m__ (float): Station 1 antenna height (m), with hr_m+1 <= h1_m <= hr_m+100.
- __h2_m__ (float): Station 2 antenna height (m), with hr_m-10 <= h2_m <= hr_m-4.
- __hr_m__ (float): Average height of buildings (m).
- __w2_m__ (float): Street width (m) at station 2's location, with 10 <= w2_m <= 25.
- __phi_deg__ (float): Street orientation with respect to the direct path (deg), with 0 < phi_deg <= 90 (i.e. for a street that is perpendicular to the direct path, phi_deg is 90 deg).

Returns:
- __L<sub>NLoS1</sub>__ (float): Basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificNearStreetLevelUrban
#### crc_covlib.helper.itur_p1411.SiteSpecificNearStreetLevelUrban
```python
def SiteSpecificNearStreetLevelUrban(f_GHz: float, x1_m: float, x2_m: float, x3_m: float,
                                     h1_m: float, h2_m: float, hs_m: float) -> float
```
ITU-R P.1411-12, Annex 1, Section 4.3.2\
Site-specific model that estimates the basic transmission loss (dB) in a rectilinear street grid urban environment.

When both x2_m and x3_m are set to zero, the estimate is based on the algorithm for line-of-sight situations (section 4.3.2.1).

When x3_m only is set to zero, the estimate is based on the "1-Turn NLoS propagation" algorithm (section 4.3.2.2).

When x1_m, x2_m and x3_m values are all greater than zero, the estimate is based on the "2-Turn NLoS propagation" algorithm (section 4.3.2.2). For a 2-Turn NLoS link, "it is possible to establish multiple travel route paths". Use equation (68) from the recommendation to consider all 2-Turn route paths in the overall path loss calculation.

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
- __x1_m__ (float): Distance between station 1 and the first street corner (m), with 0 < x1_m.
- __x2_m__ (float): Distance between the first street corner and the second street corner (m), with 0 <= x2_m.
- __x3_m__ (float): Distance between the second street corner and station 2 (m), with 0 <= x3_m.
- __h1_m__ (float): Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
- __h2_m__ (float): Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
- __hs_m__ (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23 to 1.6 meters. Only used when f_GHz >= 3.

Return:
- __L__ (float): Basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SiteSpecificNearStreetLevelResidential
#### crc_covlib.helper.itur_p1411.SiteSpecificNearStreetLevelResidential
```python
def SiteSpecificNearStreetLevelResidential(f_GHz: float, d_m: float, hTx_m: float, hRx_m: float,
                                           hbTx_m: float, hbRx_m: float, a_m: float, b_m: float,
                                           c_m: float, m_m: float, n_bldgkm2: float,
                                           thetaList_deg: list[float], x1List_m: list[float],
                                           x2List_m: list[float]
                                           ) -> tuple[float, float, float, float]
```
ITU-R P.1411-12, Annex 1, Section 4.3.3\
Site-specific model "that predicts whole path loss L between two terminals of low height in  residential environments ...by using path loss along a road L<sub>r</sub>, path loss between houses L<sub>b</sub>, and over-roof basic transmission loss L<sub>v</sub>". "Applicable areas are both [line-of-sight] and [non-line-of-sight] regions that include areas having two or more corners".

Args:
- __f_GHz__ (float): Operating frequency (GHz), with 2 <= f_GHz <= 26.
- __d_m__ (float): Distance between the two terminals (m), with 0 <= d_m <= 1000.
- __hTx_m__ (float): Transmitter antenna height (m), with 1.2 <= hTx_m <= 6.
- __hRx_m__ (float): Receiver antenna height (m), with 1.2 <= hRx_m <= 6.
- __hbTx_m__ (float): Height of nearest building from transmitter in receiver direction (m).
- __hbRx_m__ (float): Height of nearest building from receiver in transmitter direction (m).
- __a_m__ (float): Distance between transmitter and nearest building from transmitter (m).
- __b_m__ (float): Distance between nearest buildings from transmitter and receiver (m).
- __c_m__ (float): Distance between receiver and nearest building from receiver (m).
- __m_m__ (float): Average building height of the buildings with less than 3 stories (m). Typically between 6 and 12 meters.
- __n_bldgkm2__ (float): Building density (buildings/km<sup>2</sup>).
- __thetaList_deg__ (list[float]): List of theta_i, where theta_i is the road angle of i-th street corner (degrees), with 0 <= theta_i <= 90.
- __x1List_m__ (list[float]): List of x1_i, where x1_i is the road distance from transmitter to i-th street corner (m).
- __x2List_m__ (list[float]): List of x2_i, where x2_i is the road distance from i-th street corner to receiver (m).

Return:
- __L__ (float): Basic transmission loss (dB).
- __L<sub>r</sub>__ (float): Path loss along road (dB).
- __L<sub>b</sub>__ (float): Path loss between houses (dB).
- __L<sub>v</sub>__ (float): Over-roof basic transmission loss (dB).

[Back to top](#itur_p1411-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***