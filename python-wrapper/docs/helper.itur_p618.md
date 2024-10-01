# itur_p618 helper module
Implementation of ITU-R P.618-14 (partial).

```python
from crc_covlib.helper import itur_p618
```

- [RainAttenuation](#rainattenuation)
- [ScintillationFading](#scintillationfading)
- [GaseousAttenuation](#gaseousattenuation)
- [CloudAttenuation](#cloudattenuation)
- [TotalAtmosphericAttenuation](#totalatmosphericattenuation)
- [MeanRadiatingTemperature](#meanradiatingtemperature)
- [SkyNoiseTemperature](#skynoisetemperature)
- [HydrometeorCrossPolDiscrimination](#hydrometeorcrosspoldiscrimination)

***

### RainAttenuation
#### crc_covlib.helper.itur_p618.RainAttenuation
```python
def RainAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                    polTilt_deg: float, hs_km: float|None=None) -> float
```
ITU-R P.618-14, Annex 1, Section 2.2.1.1\
Estimates the attenuation due to rain exceeded for p% of an average year (dB).
    
Args:
- __f_GHz__ (float): Frequency of interest (GHz), with f_GHz <= 55.
- __theta_deg__ (float): Elevation angle of the slant propagation path (degrees).
- __p__ (float): Time percentage, with 0.001 <= p <= 5.
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
- __polTilt_deg__ (float): Polarization tilt angle relative to the horizontal (deg). Use 45° for circular polarization.
- __hs_km__ (float): Height above mean sea level of the earth station (km). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>R</sub>__ (float): Attenuation due to rain exceeded for p% of an average year (dB).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ScintillationFading
#### crc_covlib.helper.itur_p618.ScintillationFading
```python
def ScintillationFading(f_GHz: float, theta0_deg: float, p: float, lat: float, lon: float,
                        D: float, eff: float=0.5) -> float
```
ITU-R P.618-14, Annex 1, Section 2.4.1\
Calculates the tropospheric scintillation fading, exceeded for p% of the time (dB).

Args:
- __f_GHz__ (float): Frequency (GHz), with 4 <= f_GHz <= 55.
- __theta0_deg__ (float): Free-space elevation angle (degrees), with 5 <= theta0_deg <= 90.
- __p__ (float): Time percentage, with 0.01 < p <= 50.
- __lat__ (float): Latitude (degrees), with -90 <= lat <= 90. Latitude and longitude values are used to obtain Nwet, the median value of the wet term of the surface refractivity exceeded for the average year, from the digital maps in Recommendation ITU-R P.453.
- __lon__ (float): Longitude (degrees), with -180 <= lon <= 180.
- __D__ (float): Physical diameter of the earth-station antenna (meters).
- __eff__ (float): Antenna efficiency; if unknown, 0.5 is a conservative estimate.
    
Returns:
- __A<sub>S</sub>__ (float): Tropospheric scintillation fading, exceeded for p% of the time (dB).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GaseousAttenuation
#### crc_covlib.helper.itur_p618.GaseousAttenuation
```python
def GaseousAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                       hs_km: float|None=None) -> float
```
ITU-R P.618-14, Annex 1, Section 2.5\
Gets the gaseous attenuation due to water vapour and oxygen for a fixed probability (dB), as estimated by Recommendation ITU-R P.676.

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __hs_km__ (float|None): Height of the desired location (km above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __A<sub>G</sub>__ (float): The gaseous attenuation due to water vapour and oxygen for a fixed probability (dB).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### CloudAttenuation
#### crc_covlib.helper.itur_p618.CloudAttenuation
```python
def CloudAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float) -> float
```
ITU-R P.618-14, Annex 1, Section 2.5\
Gets the attenuation due to clouds for a fixed probability (dB), as estimated by Recommendation ITU-R P.840.

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
- __theta_deg__ (float): Elevation angle (deg).
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.

Returns:
- __A<sub>C</sub>__ (float): The attenuation due to clouds for a fixed probability (dB).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TotalAtmosphericAttenuation
#### crc_covlib.helper.itur_p618.TotalAtmosphericAttenuation
```python
def TotalAtmosphericAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                                polTilt_deg: float, D: float, eff: float=0.5,
                                hs_km: float|None=None, excludeScintillation: bool=False
                                ) -> tuple[float, float, float, float, float]
```
ITU-R P.618-14, Annex 1, Section 2.5\
Estimation of total attenuation due to multiple sources of simultaneously occurring atmospheric attenuation (dB). Total attenuation represents the combined effect of rain, gas, clouds and scintillation.
    
Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz <= 55.
- __theta_deg__ (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
- __p__ (float): The probability the attenuation is exceeded (i.e. the CCDF), in %, with 0.001 <= p <= 50.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __polTilt_deg__ (float): Polarization tilt angle relative to the horizontal (deg). Use 45° for circular polarization.
- __D__ (float): Physical diameter of the earth-station antenna (meters).
- __eff__ (float): Antenna efficiency; if unknown, 0.5 is a conservative estimate.
- __hs_km__ (float|None): Height of the desired location (km above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.
- __excludeScintillation__ (bool): Whether to exclude tropospheric scintillation from the total attenuation. Relevant in the context of sky noise temperature calculation. When set to True, parameters D and eff are unused.
    
Returns:
- __A<sub>T</sub>__ (float): Estimation of total attenuation (dB) due to multiple sources of simultaneously occurring atmospheric attenuation (rain, gas, clouds and scintillation).
- __A<sub>R</sub>__ (float): Attenuation due to rain for a fixed probability (dB).
- __A<sub>C</sub>__ (float): Attenuation due to clouds for a fixed probability (dB), as estimated by Recommendation ITU-R P.840.
- __A<sub>G</sub>__ (float): Gaseous attenuation due to water vapour and oxygen for a fixed probability (dB), as estimated by Recommendation ITU-R P.676.
- __A<sub>S</sub>__ (float): Attenuation due to tropospheric scintillation for a fixed probability (dB).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanRadiatingTemperature
#### crc_covlib.helper.itur_p618.MeanRadiatingTemperature
```python
def MeanRadiatingTemperature(Ts_K: float) -> float
```
ITU-R P.618-14, Annex 1, Section 3\
Calculates the mean radiating temperature (K) from the surface temperature.

Args:
- __Ts_K__ (float): The surface temperature (K).

Returns:
- __T<sub>mr</sub>__ (float): The mean radiating temperature (K).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SkyNoiseTemperature
#### crc_covlib.helper.itur_p618.SkyNoiseTemperature
```python
def SkyNoiseTemperature(AT_dB: float, Tmr_K: float=275) -> float
```
ITU-R P.618-14, Annex 1, Section 3\
Gets the sky noise temperature at a ground station antenna (K).

Args:
- __AT_dB__ (float): The total atmospheric attenuation excluding scintillation fading (dB).
- __Tmr_K__ (float): The mean radiating temperature (K).

Returns:
- __T<sub>sky</sub>__ (float): The sky noise temperature at a ground station antenna (K).

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### HydrometeorCrossPolDiscrimination
#### crc_covlib.helper.itur_p618.HydrometeorCrossPolDiscrimination
```python
def HydrometeorCrossPolDiscrimination(f_GHz: float, theta_deg: float, p: float,
                                      polTilt_deg: float, AR_dB: float) -> float
```
ITU-R P.618-14, Annex 1, Section 4.1\
Gets the hydrometeor-induced cross-polarization discrimination (dB) not exceeded for p% of the time.

Args:
- __f_GHz__ (float): Frequency (GHz), with 6 <= f_GHz <= 55.
- __theta_deg__ (float): Path elevation angle (deg), with theta_deg <= 60.
- __p__ (float): Time percentage for which the cross-polarization discrimination is not exceeded, in %. Must be set to one of the four following values: 1, 0.1, 0.01 and 0.001.
- __polTilt_deg__ (float): Tilt angle of the linearly polarized electric field vector with respect to the horizontal (deg). For circular polarization use 45°.
- __AR_dB__ (float): Rain attenuation (dB) exceeded for the required percentage of time, p, for the path in question, commonly called co-polar attenuation (CPA). May be calculated using RainAttenuation() from this module.
    
Returns:
- __XPD<sub>p</sub>__ (float): The hydrometeor-induced cross-polarization discrimination (dB) not exceeded for p% of the time.

[Back to top](#itur_p618-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***