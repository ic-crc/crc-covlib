# itur_p676 helper module
Implementation of ITU-R P.676-13. Fully implemented except for Section 5 of Annex 1.

Note: Statistical (Stat) functions from Annex 2 support using both annual and monthly statistics from the itur_p2145.py module. See the [itur_p2145](./helper.itur_p2145.md) module for more details on how to install
statistics files.


```python
from crc_covlib.helper import itur_p676
```

- [GaseousAttenuation](#gaseousattenuation)
- [DryAirGaseousAttenuation](#dryairgaseousattenuation)
- [WaterVapourGaseousAttenuation](#watervapourgaseousattenuation)
- [TerrestrialPathGaseousAttenuation](#terrestrialpathgaseousattenuation)
- [RefractiveIndex](#refractiveindex)
- [SlantPathGaseousAttenuation](#slantpathgaseousattenuation)
- [EarthToSpaceReciprocalApparentElevAngle](#earthtospacereciprocalapparentelevangle)
- [SpaceToEarthReciprocalApparentElevAngle](#spacetoearthreciprocalapparentelevangle)
- [InterceptsEarth](#interceptsearth)
- [PhaseDispersion](#phasedispersion)
- [DownwellingMicrowaveBrightnessTemperature](#downwellingmicrowavebrightnesstemperature)
- [UpwellingMicrowaveBrightnessTemperature](#upwellingmicrowavebrightnesstemperature)
- [SlantPathInstantOxygenGaseousAttenuation](#slantpathinstantoxygengaseousattenuation)
- [SlantPathStatOxygenGaseousAttenuation](#slantpathstatoxygengaseousattenuation)
- [SlantPathInstantWaterVapourGaseousAttenuation1](#slantpathinstantwatervapourgaseousattenuation1)
- [SlantPathInstantWaterVapourGaseousAttenuation2](#slantpathinstantwatervapourgaseousattenuation2)
- [SlantPathStatWaterVapourGaseousAttenuation](#slantpathstatwatervapourgaseousattenuation)
- [SlantPathStatGaseousAttenuation](#slantpathstatgaseousattenuation)
- [WeibullApproxAttenuation](#weibullapproxattenuation)

***

### GaseousAttenuation
#### crc_covlib.helper.itur_p676.GaseousAttenuation
```python
def GaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                       rho_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 1\
Gaseous attenuation attributable to dry air and water vapour (dB/km).

Args:
- __f_GHz__ (float): frequency (GHz) [1, 1000]
- __P_hPa__ (float): atmospheric pressure (hPa)
- __T_K__ (float): temperature (K)
- __rho_gm3__ (float): water vapour density (g/m3)

Returns:
- __γ__ (float): Gaseous attenuation attributable to dry air and water vapour (dB/km).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DryAirGaseousAttenuation
#### crc_covlib.helper.itur_p676.DryAirGaseousAttenuation
```python
def DryAirGaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                             rho_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 1\
Specific gaseous attenuation attributable to dry air (dB/km).

Args:
- __f_GHz__ (float): frequency (GHz) [1, 1000]
- __P_hPa__ (float): atmospheric pressure (hPa)
- __T_K__ (float): temperature (K)
- __rho_gm3__ (float): water vapour density (g/m3)

Returns:
- __γ<sub>o</sub>__ (float): Specific gaseous attenuation attributable to dry air (oxygen, pressure-induced nitrogen and non-resonant Debye attenuation) (dB/km).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WaterVapourGaseousAttenuation
#### crc_covlib.helper.itur_p676.WaterVapourGaseousAttenuation
```python
def WaterVapourGaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                                  rho_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 1\
Specific gaseous attenuation attributable to water vapour (dB/km).

Args:
- __f_GHz__ (float): frequency (GHz) [1, 1000]
- __P_hPa__ (float): atmospheric pressure (hPa)
- __T_K__ (float): temperature (K)
- __rho_gm3__ (float): water vapour density (g/m3)

Returns:
- __γ<sub>w</sub>__ (float): Specific gaseous attenuation attributable to water vapour (dB/km).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TerrestrialPathGaseousAttenuation
#### crc_covlib.helper.itur_p676.TerrestrialPathGaseousAttenuation
```python
def TerrestrialPathGaseousAttenuation(pathLength_km: float, f_GHz: float, P_hPa: float=1013.25,
                                      T_K: float=288.15, rho_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 2.1\
Path attenuation for a terrestrial path, or for slightly inclined paths close to the ground (dB).

Args:
- __pathLength_km__ (float): path length (km)
- __f_GHz__ (float): frequency (GHz) [1, 1000]
- __P_hPa__ (float): atmospheric pressure (hPa)
- __T_K__ (float): temperature (K)
- __rho_gm3__ (float): water vapour density (g/m3)

Returns:
- __A__ (float): Path attenuation for a terrestrial path, or for slightly inclined paths close to the ground (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RefractiveIndex
#### crc_covlib.helper.itur_p676.RefractiveIndex
```python
def RefractiveIndex(h_km: float, refAtm: ReferenceAtmosphere=MAGRA, 
                    rho0_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 2.2.1\
Calculates the refractive index at any height for Earth-satellite paths.

Args:
- __h_km__ (float): Height in km, with h_km >= 0.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __n__ (float): The refractive index at height h_km (for Earth-satellite paths).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathGaseousAttenuation
#### crc_covlib.helper.itur_p676.SlantPathGaseousAttenuation
```python
def SlantPathGaseousAttenuation(f_GHz: float, h1_km: float, h2_km: float, phi1_deg: float,
                                refAtm: ReferenceAtmosphere=MAGRA,
                                rho0_gm3: float=7.5) -> tuple[float, float, float]
```
ITU-R P.676-13, Annex 1, Section 2.2 (Slant paths)\
Calculates the Earth-space slant path gaseous attenuation for an ascending path between a location on or near the surface of the Earth and a location above the surface of the Earth or in space.

Args:
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
- __h1_km__ (float): Height of the first station (km), with 0 <= h1_km < h2_km.
- __h2_km__ (float): Height of the second station (km), with 0 <= h1_km < h2_km.
- __phi1_deg__ (float): The local apparent elevation angle at height h1_km (degrees). 0°=horizon, +90°=zenith, -90°=nadir.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __A<sub>gas</sub>__ (float): The slant path gaseous attenuation (dB).
- __bending__ (float): The total atmosphering bending along the path (radians).
- __ΔL__ (float): The excess atmospheric path length (km).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EarthToSpaceReciprocalApparentElevAngle
#### crc_covlib.helper.itur_p676.EarthToSpaceReciprocalApparentElevAngle
```python
def EarthToSpaceReciprocalApparentElevAngle(He_km: float, Hs_km: float, phi_e_deg: float,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 2.2.3\
Calculates the reciprocal apparent elevation angle at the space-based station based on the apparent elevation angle at the Earth-based station.

The gaseous attenuation for a space-Earth path, where the apparent elevation angle at the space station is φ<sub>s</sub>, is identical to the gaseous attenuation for the reciprocal Earth-space path, where the apparent elevation angle at the earth station is φ<sub>e</sub>.

Args:
- __He_km__ (float): Height of the Earth-based station (km).
- __Hs_km__ (float): Height of the space-based station (km).
- __phi_e_deg__ (float): Apparent elevation angle at the Earth-based station (deg). 0°=horizon, +90°=zenith, -90°=nadir.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __φ<sub>s</sub>__ (float): Reciprocal apparent elevation angle at the space-based station (deg).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SpaceToEarthReciprocalApparentElevAngle
#### crc_covlib.helper.itur_p676.SpaceToEarthReciprocalApparentElevAngle
```python
def SpaceToEarthReciprocalApparentElevAngle(He_km: float, Hs_km: float, phi_s_deg: float,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 2.2.3\
Calculates the reciprocal apparent elevation angle at the Earth-based station based on the apparent elevation angle at the space-based station.

The gaseous attenuation for a space-Earth path, where the apparent elevation angle at the space
station is φ<sub>s</sub>, is identical to the gaseous attenuation for the reciprocal Earth-space
path, where the apparent elevation angle at the earth station is φ<sub>e</sub>.

Args:
- __He_km__ (float): Height of the Earth-based station (km).
- __Hs_km__ (float): Height of the space-based station (km).
- __phi_s_deg__ (float): Apparent elevation angle at the space-based station (deg). 0°=horizon, +90°=zenith, -90°=nadir.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __φ<sub>e</sub>__ (float): Reciprocal apparent elevation angle at the Earth-based station (deg).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### InterceptsEarth
#### crc_covlib.helper.itur_p676.InterceptsEarth
```python
def InterceptsEarth(He_km: float, Hs_km: float, phi_s_deg: float,
                    refAtm: ReferenceAtmosphere=MAGRA,
                    rho0_gm3: float=7.5) -> bool
```
ITU-R P.676-13, Annex 1, Section 2.2.3\
Evaluates whether a space-Earth path intercepts the Earth.

Args:
- __He_km__ (float): Height of the Earth-based station (km).
- __Hs_km__ (float): Height of the space-based station (km).
- __phi_s_deg__ (float): Apparent elevation angle at the space-based station (deg). 0°=horizon, +90°=zenith, -90°=nadir.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- (bool): True if the space-Earth path intercepts the Earth, False otherwise.

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PhaseDispersion
#### crc_covlib.helper.itur_p676.PhaseDispersion
```python
def PhaseDispersion(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                    rho_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 3\
Calculates the gaseous phase dispersion (deg/km).

Args:
- __f_GHz__ (float): frequency (GHz) [1, 1000]
- __P_hPa__ (float): atmospheric pressure (hPa)
- __T_K__ (float): temperature (K)
- __rho_gm3__ (float): water vapour density (g/m3)

Returns:
- __φ__ (float): The gaseous phase dispersion (deg/km).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownwellingMicrowaveBrightnessTemperature
#### crc_covlib.helper.itur_p676.DownwellingMicrowaveBrightnessTemperature
```python
def DownwellingMicrowaveBrightnessTemperature(f_GHz: float, phi1_deg: float,
                                              h1_km: float=0, hk_km=100.1,
                                              refAtm: ReferenceAtmosphere=MAGRA,
                                              rho0_gm3: float=7.5) -> float
```
ITU-R P.676-13, Annex 1, Section 4.1\
The downwelling atmospheric microwave brightness temperature (K).

Args:
- __f_GHz__ (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
- __phi1_deg__ (float): The local apparent elevation angle at height h1_km, in degrees, with phi1_deg >= 0. 0°=horizon, +90°=zenith.
- __h1_km__ (float): Height of atmospheric layer 1 (km).
- __hk_km__ (float): Height of atmospheric layer k (km).
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __T<sub>downwelling</sub>__ (float): The downwelling atmospheric microwave brightness temperature (K).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### UpwellingMicrowaveBrightnessTemperature
#### crc_covlib.helper.itur_p676.UpwellingMicrowaveBrightnessTemperature
```python
def UpwellingMicrowaveBrightnessTemperature(f_GHz: float, phi1_deg: float,
                                            h1_km: float=0, hk_km=100.1,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5, emissivity: float=0.95,
                                            TEarth_K: float=290) -> float
```
ITU-R P.676-13, Annex 1, Section 4.2\
The upwelling atmospheric microwave brightness temperature (K).

Args:
- __f_GHz__ (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
- __phi1_deg__ (float): The local apparent elevation angle at height h1_km, in degrees, with phi1_deg >= 0. 0°=horizon, +90°=zenith.
- __h1_km__ (float): Height of atmospheric layer 1 (km).
- __hk_km__ (float): Height of atmospheric layer k (km).
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).
- __emissivity__ (float): Emissivity of the Earth's surface.
- __TEarth_K__ (float): Temperature of the Earth's surface (K).

Returns:
- __T<sub>upwelling</sub>__ (float): The upwelling atmospheric microwave brightness temperature (K).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathInstantOxygenGaseousAttenuation
#### crc_covlib.helper.itur_p676.SlantPathInstantOxygenGaseousAttenuation
```python
def SlantPathInstantOxygenGaseousAttenuation(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                             Ts_K: float, rhoWs_gm3: float) -> float
```
ITU-R P.676-13, Annex 2, Section 1.1\
Gets the predicted slant path instantaneous gaseous attenuation attributable to oxygen (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __Ps_hPa__ (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired location.
- __Ts_K__ (float): Instantaneous surface temperature, in K, at the desired location.
- __rhoWs_gm3__ (float): Instantaneous surface water vapour density, in g/m3, at the desired location.

Returns:
- __A<sub>o</sub>__ (float): The predicted slant path instantaneous gaseous attenuation attributable to
        oxygen (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathStatOxygenGaseousAttenuation
#### crc_covlib.helper.itur_p676.SlantPathStatOxygenGaseousAttenuation
```python
def SlantPathStatOxygenGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                          lat: float, lon: float, month: int|None=None,
                                          h_mamsl: float|None=None) -> float
```
ITU-R P.676-13, Annex 2, Section 1.2\
Gets the predicted slant path statistical gaseous attenuation attributable to oxygen (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>o</sub>__ (float): The predicted slant path statistical gaseous attenuation attributable to oxygen (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathInstantWaterVapourGaseousAttenuation1
#### crc_covlib.helper.itur_p676.SlantPathInstantWaterVapourGaseousAttenuation1
```python
def SlantPathInstantWaterVapourGaseousAttenuation1(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                                   Ts_K: float, rhoWs_gm3: float) -> float
```
ITU-R P.676-13, Annex 2, Section 2.1\
Gets the predicted slant path instantaneous gaseous attenuation attributable to water vapour, in dB, using method 1.

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __Ps_hPa__ (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired location.
- __Ts_K__ (float): Instantaneous surface temperature, in K, at the desired location.
- __rhoWs_gm3__ (float): Instantaneous surface water vapour density, in g/m3, at the desired location.

Returns:
- __A<sub>w</sub>__ (float): The predicted slant path instantaneous gaseous attenuation attributable to water vapour (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathInstantWaterVapourGaseousAttenuation2
#### crc_covlib.helper.itur_p676.SlantPathInstantWaterVapourGaseousAttenuation2
```python
def SlantPathInstantWaterVapourGaseousAttenuation2(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                                   Ts_K: float, rhoWs_gm3: float, Vs_kgm2: float
                                                   ) -> float
```
ITU-R P.676-13, Annex 2, Section 2.2\
Gets the predicted slant path instantaneous gaseous attenuation attributable to water vapour, in dB, using method 2.

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __Ps_hPa__ (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired location.
- __Ts_K__ (float): Instantaneous surface temperature, in K, at the desired location.
- __rhoWs_gm3__ (float): Instantaneous surface water vapour density, in g/m3, at the desired location.
- __Vs_kgm2__ (float): Integrated water vapour content, in kg/m2 or mm, from the surface of the Earth at the desired location.

Returns:
- __A<sub>w</sub>__ (float): The predicted slant path instantaneous gaseous attenuation attributable to water vapour (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathStatWaterVapourGaseousAttenuation
#### crc_covlib.helper.itur_p676.SlantPathStatWaterVapourGaseousAttenuation
```python
def SlantPathStatWaterVapourGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                               lat: float, lon: float, month: int|None=None,
                                               h_mamsl: float|None=None) -> float
```
ITU-R P.676-13, Annex 2, Section 2.3\
Gets the predicted slant path statistical gaseous attenuation attributable to water vapour (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>w</sub>__ (float): The predicted slant path statistical gaseous attenuation attributable to water vapour (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SlantPathStatGaseousAttenuation
#### crc_covlib.helper.itur_p676.SlantPathStatGaseousAttenuation
```python
def SlantPathStatGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                    lat: float, lon: float, month: int|None=None,
                                    h_mamsl: float|None=None) -> float
```
ITU-R P.676-13, Annex 2, Sections 1.2 and 2.3\
Gets the predicted slant path statistical gaseous attenuation attributable to both water vapour and oxygen (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>total</sub>__ (float): The predicted slant path statistical gaseous attenuation attributable to
        both water vapour and oxygen (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WeibullApproxAttenuation
#### crc_covlib.helper.itur_p676.WeibullApproxAttenuation
```python
def WeibullApproxAttenuation(f_GHz: float, theta_deg: float, p: float,
                             lat: float, lon: float, h_mamsl: float|None=None) -> float
```
ITU-R P.676-13, Annex 2, Section 2.4\
Gets the Weibull approximation to the predicted slant path statistical gaseous attenuation attributable to water vapour (dB). This approximation is based on annual statistics.

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>w</sub>__ (float): The Weibull approximation to the predicted slant path statistical gaseous
        attenuation attributable to water vapour (dB).

[Back to top](#itur_p676-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***