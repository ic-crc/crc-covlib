# itur_m2101 helper module
Implementation of ITU-R M.2021-0, Annex 1, Section 5.

```python
from crc_covlib.helper import itur_m2101 
```

- [IMTAntennaElementGain](#imtantennaelementgain)
- [IMTCompositeAntennaGain](#imtcompositeantennagain)

***

### IMTAntennaElementGain
#### crc_covlib.helper.itur_m2101.IMTAntennaElementGain

```python
def IMTAntennaElementGain(phi: float, theta: float, phi_3dB: float, theta_3dB: float, 
                          Am: float, SLAv: float, GEmax: float) -> float
```
Gets the gain (dBi) of a single IMT (International Mobile Telecommunications) antenna element at the specified azimuth and elevation angle. See ITU-R M.2021-0, Annex 1, Section 5.1 for details.

Args:
- __phi__ (float): Azimuth in degrees, from -180 to 180.
- __theta__ (float): Elevation angle in degrees, from 0 (zenith) to 180 (nadir).
- __phi_3dB__ (float): Horizontal 3dB bandwidth of single element, in degrees.
- __theta_3dB__ (float): Vertical 3dB bandwidth of single element, in degrees.
- __Am__ (float): Front-to-back ratio, in dB.
- __SLAv__ (float): Vertical sidelobe attenuation, in dB.
- __GEmax__ (float): Maximum gain of single element, in dBi.

Returns:
- float: gain of a single antenna element, in dBi.

[Back to top](#itur_m2101-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### IMTCompositeAntennaGain
#### crc_covlib.helper.itur_m2101.IMTCompositeAntennaGain

```python
def IMTCompositeAntennaGain(phi: float, theta: float, phi_3dB: float, theta_3dB: float, 
                            Am: float, SLAv: float, GEmax: float, NH: int, NV: int,
                            dH_over_wl: float, dV_over_wl: float,
                            phi_i_escan: float, theta_i_etilt: float) -> float
```
Gets the gain (dBi) of an IMT (International Mobile Telecommunications) composite antenna (i.e. beamforming antenna) at the specified azimuth and elevation angle. See ITU-R M.2021-0, Annex 1, Section 5.2 for details.

Args:
- __phi__ (float): Azimuth at which to get the gain from, in degrees, from -180 to 180.
- __theta__ (float): Elevation angle at which to get the gain from, in degrees, from 0 (zenith) to 180 (nadir).
- __phi_3dB__ (float): Horizontal 3dB bandwidth of single element, in degrees.
- __theta_3dB__ (float): Vertical 3dB bandwidth of single element, in degrees.
- __Am__ (float): Front-to-back ratio, in dB.
- __SLAv__ (float): Vertical sidelobe attenuation, in dB.
- __GEmax__ (float): Maximum gain of single element, in dBi.
- __NH__ (int): Number of columns in the array of elements.
- __NV__ (int): Number of rows in the array of elements.
- __dH_over_wl__ (float): Horizontal elements spacing over wavelength (dH/ʎ).
- __dV_over_wl__ (float): Vertical elements spacing over wavelength (dV/ʎ).
- __phi_i_escan__ (float): Bearing (h angle) of formed beam, in degrees.
- __theta_i_etilt__ (float): Tilt (v angle) of formed beam, in degrees (positive value for downtilt, negative for uptilt).

Returns:
- float: Composite antenna gain, in dBi.

[Back to top](#itur_m2101-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***
