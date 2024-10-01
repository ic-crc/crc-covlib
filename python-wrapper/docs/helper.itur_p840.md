# itur_p840 helper module
Implementation of ITU-R P.840-9.

This module supports using both annual and monthly statistics. Statistics data files can be obtained at https://www.itu.int/rec/R-REC-P.840/en or they can be installed by running the install_ITU_data.py script (use the Custom mode for the option to install monthly statistics, the default mode will only install annual statistics). Another option is to use functions from the [itur_data](./helper.itur_data.md) module.

When installing files manually: \
Annual statistics files must be placed in the data/itu_proprietary/p840/annual/ directory. Statistics for the month of January must be placed in the helper/data/itu_proprietary/p840/monthly/01/ directory, for the month of February they must be placed in the helper/data/p840/monthly/02/ directory,
etc.


```python
from crc_covlib.helper import itur_p840
```

- [CloudLiquidWaterAttenuationCoefficient](#cloudliquidwaterattenuationcoefficient)
- [InstantaneousCloudAttenuation](#instantaneouscloudattenuation)
- [StatisticalCloudAttenuation](#statisticalcloudattenuation)
- [LogNormalApproxCloudAttenuation](#lognormalapproxcloudattenuation)
- [IntegratedCloudLiquidWaterContent](#integratedcloudliquidwatercontent)
- [IntegratedCloudLiquidWaterContentMean](#integratedcloudliquidwatercontentmean)
- [IntegratedCloudLiquidWaterContentStdDev](#integratedcloudliquidwatercontentstddev)

***

### CloudLiquidWaterAttenuationCoefficient
#### crc_covlib.helper.itur_p840.CloudLiquidWaterAttenuationCoefficient
```python
def CloudLiquidWaterAttenuationCoefficient(f_GHz, T_K) -> float
```
ITU-R P.840-9, Annex 1, Section 2 \
Gets the cloud liquid water specific attenuation coefficient (dB/km)/(g/m3).

Args:
- __f_GHz__ (float): Frequency (GHz), f_GHz <= 200.
- __T_K__ (float): Liquid water temperature (K).
    
Returns:
- __K<sub>l</sub>__ (float): The cloud liquid water specific attenuation coefficient (dB/km)/(g/m3).

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### InstantaneousCloudAttenuation
#### crc_covlib.helper.itur_p840.InstantaneousCloudAttenuation
```python
def InstantaneousCloudAttenuation(f_GHz: float, theta_deg: float, L_kgm2: float) -> float
```
ITU-R P.840-9, Annex 1, Section 3.1 \
Gets the predicted slant path instantaneous cloud attenuation (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
- __theta_deg__ (float): Elevation angle (deg).
- __L_kgm2__ (float): Integrated cloud liquid water content, in kg/m2 or mm, from the surface of the Earth at the desired location.

Returns:
- __A<sub>c</sub>__ (float): The predicted slant path instantaneous cloud attenuation (dB).

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StatisticalCloudAttenuation
#### crc_covlib.helper.itur_p840.StatisticalCloudAttenuation
```python
def StatisticalCloudAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                                month: int|None=None) -> float
```
ITU-R P.840-9, Annex 1, Section 3.2 \
Gets the predicted slant path statistical cloud attenuation (dB).

Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
- __theta_deg__ (float): Elevation angle (deg).
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual statistics and with 0.1 <= p <= 100 for monthly statisctics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual statistics.

Returns:
- __A<sub>c</sub>__ (float): The predicted slant path statistical cloud attenuation (dB).

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LogNormalApproxCloudAttenuation
#### crc_covlib.helper.itur_p840.LogNormalApproxCloudAttenuation
```python
def LogNormalApproxCloudAttenuation(f_GHz: float, theta_deg: float, p: float,
                                    lat: float, lon: float) -> float
```
ITU-R P.840-9, Annex 1, Section 3.3 \
Gets the log-normal approximation to the predicted slant path statistical cloud attenuation (dB). Uses annual statistics.
    
Args:
- __f_GHz__ (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
- __theta_deg__ (float): Elevation angle (deg).
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual statistics and with 0.1 <= p <= 100 for monthly statisctics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.

Returns:
- __A<sub>c</sub>__ (float): The log-normal approximation to the predicted slant path statistical cloud attenuation (dB).

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IntegratedCloudLiquidWaterContent
#### crc_covlib.helper.itur_p840.IntegratedCloudLiquidWaterContent
```python
def IntegratedCloudLiquidWaterContent(p: float, lat: float, lon: float,
                                      month: int|None=None) -> float
```
ITU-R P.840-9, Annex 1, Section 4 \
Gets the integrated cloud liquid water content from digital maps, in kg/m2, or, equivalently, mm.

Args:
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual statistics and with 0.1 <= p <= 100 for monthly statisctics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual statistics.

Returns:
- __L__ (float): The integrated cloud liquid water content from digital maps, in kg/m2, or, equivalently, mm.

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IntegratedCloudLiquidWaterContentMean
#### crc_covlib.helper.itur_p840.IntegratedCloudLiquidWaterContentMean
```python
def IntegratedCloudLiquidWaterContentMean(lat: float, lon: float, month: int|None=None
                                          ) -> float
```
ITU-R P.840-9, Annex 1, Section 4.2.2 \
Gets the mean of the integrated cloud liquid water content.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual statistics.

Returns:
- (float): The mean of the integrated cloud liquid water content.

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IntegratedCloudLiquidWaterContentStdDev
#### crc_covlib.helper.itur_p840.IntegratedCloudLiquidWaterContentStdDev
```python
def IntegratedCloudLiquidWaterContentStdDev(lat: float, lon: float, month: int|None=None
                                            ) -> float
```
ITU-R P.840-9, Annex 1, Section 4.2.2 \
Gets the standard deviation of the integrated cloud liquid water content.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual statistics.

Returns:
- (float): The standard deviation of the integrated cloud liquid water content.

[Back to top](#itur_p840-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***