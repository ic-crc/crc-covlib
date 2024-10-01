# itur_p2145 helper module
Implementation of ITU-R P.2145-0.

This module supports using both annual and monthly statistics. Statistics data files can be obtained at https://www.itu.int/rec/R-REC-P.2145/en or they can be installed by running the install_ITU_data.py script (use the Custom mode for the option to install monthly statistics, the default mode will only install annual statistics). Another option is to use functions from the [itur_data](./helper.itur_data.md) module.

When installing files manually: \
For each month, 4 directories must be added in the helper/data/itu_proprietary/p2145/ directory, namely RHO_Month03, P_Month03, T_Month03 and V_Month03 (using the month of March as an example). Annual statistics must be placed in RHO_Annual, P_Annual, T_Annual, V_Annual directories.


```python
from crc_covlib.helper import itur_p2145
```

- [SurfaceTotalPressure](#surfacetotalpressure)
- [SurfaceTemperature](#surfacetemperature)
- [SurfaceWaterVapourDensity](#surfacewatervapourdensity)
- [IntegratedWaterVapourContent](#integratedwatervapourcontent)
- [MeanSurfaceTotalPressure](#meansurfacetotalpressure)
- [MeanSurfaceTemperature](#meansurfacetemperature)
- [MeanSurfaceWaterVapourDensity](#meansurfacewatervapourdensity)
- [MeanIntegratedWaterVapourContent](#meanintegratedwatervapourcontent)
- [StdDevSurfaceTotalPressure](#stddevsurfacetotalpressure)
- [StdDevSurfaceTemperature](#stddevsurfacetemperature)
- [StdDevSurfaceWaterVapourDensity](#stddevsurfacewatervapourdensity)
- [StdDevIntegratedWaterVapourContent](#stddevintegratedwatervapourcontent)
- [WeibullParameters](#weibullparameters)

***

### SurfaceTotalPressure
#### crc_covlib.helper.itur_p2145.SurfaceTotalPressure
```python
def SurfaceTotalPressure(p: float, lat: float, lon: float, month: int|None=None,
                         h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.1 of Annex. \
Gets the surface total (barometric) pressure (hPa) for the specified exceedence probabiliby p (CCDF - complementary cumulative distribution function) at the specified location on the surface of the Earth.

Args:
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __P__ (float): The surface total (barometric) pressure (hPa) for the specified exceedence probabiliby p (CCDF) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SurfaceTemperature
#### crc_covlib.helper.itur_p2145.SurfaceTemperature
```python
def SurfaceTemperature(p: float, lat: float, lon: float, month: (int|None)=None,
                       h_mamsl: (float|None)=None) -> float
```
ITU-R P.2145-0, Section 2.1 of Annex. \
Gets the surface temperature (K) for the specified exceedence probabiliby p (CCDF - complementary cumulative distribution function) at the specified location on the surface of the Earth.

Args:
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __T__ (float): The surface temperature (K) for the specified exceedence probabiliby p (CCDF) at
        the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SurfaceWaterVapourDensity
#### crc_covlib.helper.itur_p2145.SurfaceWaterVapourDensity
```python
def SurfaceWaterVapourDensity(p: float, lat: float, lon: float, month: (int|None)=None,
                              h_mamsl: (float|None)=None) -> float
```
ITU-R P.2145-0, Section 2.1 of Annex. \
Gets the surface water vapour density (g/m3) for the specified exceedence probabiliby p (CCDF - complementary cumulative distribution function) at the specified location on the surface of the Earth.

Args:
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __rho__ (float): The surface water vapour density (g/m3) for the specified exceedence probabiliby p (CCDF) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IntegratedWaterVapourContent
#### crc_covlib.helper.itur_p2145.IntegratedWaterVapourContent
```python
def IntegratedWaterVapourContent(p: float, lat: float, lon: float, month: int|None=None,
                                 h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.1 of Annex. \
Gets the integrated water vapour content (kg/m2) for the specified exceedence probabiliby p (CCDF - complementary cumulative distribution function) at the specified location on the surface of the Earth.

Args:
- __p__ (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual statistics and with 0.1 <= p <= 99 for monthly statistics.
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __V__ (float): The integrated water vapour content (kg/m2) for the specified exceedence probabiliby p (CCDF) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanSurfaceTotalPressure
#### crc_covlib.helper.itur_p2145.MeanSurfaceTotalPressure
```python
def MeanSurfaceTotalPressure(lat: float, lon: float, month: int|None=None,
                             h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the mean of the surface total (barometric) pressure (hPa) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The mean of the surface total (barometric) pressure (hPa) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanSurfaceTemperature
#### crc_covlib.helper.itur_p2145.MeanSurfaceTemperature
```python
def MeanSurfaceTemperature(lat: float, lon: float, month: int|None=None,
                           h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the mean of the surface temperature (K) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The mean of the surface temperature (K) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanSurfaceWaterVapourDensity
#### crc_covlib.helper.itur_p2145.MeanSurfaceWaterVapourDensity
```python
def MeanSurfaceWaterVapourDensity(lat: float, lon: float, month: int|None=None,
                                  h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the mean of the surface water vapour density (g/m3) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The mean of the surface water vapour density (g/m3) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### MeanIntegratedWaterVapourContent
#### crc_covlib.helper.itur_p2145.MeanIntegratedWaterVapourContent
```python
def MeanIntegratedWaterVapourContent(lat: float, lon: float, month: int|None=None,
                                     h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the mean of the integrated water vapour content (kg/m2) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The mean of the integrated water vapour content (kg/m2) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StdDevSurfaceTotalPressure
#### crc_covlib.helper.itur_p2145.StdDevSurfaceTotalPressure
```python
def StdDevSurfaceTotalPressure(lat: float, lon: float, month: int|None=None,
                               h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the standard deviation of the surface total (barometric) pressure (hPa) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The standard deviation of the surface total (barometric) pressure (hPa) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StdDevSurfaceTemperature
#### crc_covlib.helper.itur_p2145.StdDevSurfaceTemperature
```python
def StdDevSurfaceTemperature(lat: float, lon: float, month: int|None=None,
                             h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the standard deviation of the surface temperature (K) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The standard deviation of the surface temperature (K) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StdDevSurfaceWaterVapourDensity
#### crc_covlib.helper.itur_p2145.StdDevSurfaceWaterVapourDensity
```python
def StdDevSurfaceWaterVapourDensity(lat: float, lon: float, month: int|None=None,
                                    h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the standard deviation of the surface water vapour density (g/m3) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The standard deviation of the surface water vapour density (g/m3) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StdDevIntegratedWaterVapourContent
#### crc_covlib.helper.itur_p2145.StdDevIntegratedWaterVapourContent
```python
def StdDevIntegratedWaterVapourContent(lat: float, lon: float, month: int|None=None,
                                       h_mamsl: float|None=None) -> float
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the standard deviation of the integrated water vapour content (kg/m2) at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __month__ (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None, annual statistics are used.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- (float): The standard deviation of the integrated water vapour content (kg/m2) at the specified location on the surface of the Earth.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### WeibullParameters
#### crc_covlib.helper.itur_p2145.WeibullParameters
```python
def WeibullParameters(lat: float, lon: float, h_mamsl: float|None=None
                      ) -> tuple[float, float]
```
ITU-R P.2145-0, Section 2.2 of Annex. \
Gets the Weibull integrated water vapour content shape and scale parameters at the specified location on the surface of the Earth.

Args:
- __lat__ (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
- __lon__ (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
- __h_mamsl__ (float|None): Height of the desired location (meters above mean sea level). When set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

Returns:
- __k<sub>VS</sub>__ (float): The Weibull integrated water vapour content shape parameters.
- __lambda<sub>VS</sub>__ (float): The Weibull integrated water vapour content scale parameters.

[Back to top](#itur_p2145-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***