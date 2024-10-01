# itur_p619 helper module
Partial implementation of ITU-R P.619-5.

```python
from crc_covlib.helper import itur_p619
```

- [CalculationStatus (Enum)](#calculationstatus)
- [FreeSpaceBasicTransmissionLoss](#freespacebasictransmissionloss)
- [SpaceToEarthGaseousAttenuation](#spacetoearthgaseousattenuation)
- [EarthToSpaceGaseousAttenuation](#earthtospacegaseousattenuation)
- [BeamSpreadingLoss](#beamspreadingloss)
- [TroposphericScintillationAttenuation](#troposphericscintillationattenuation)
- [StraightLineEarthSpacePath](#straightlineearthspacepath)
- [FreeSpaceToApparentElevationAngle](#freespacetoapparentelevationangle)
- [ApparentToFreeSpaceElevationAngle](#apparenttofreespaceelevationangle)
- [RayHeights](#rayheights)

***

### CalculationStatus
#### crc_covlib.helper.itur_p619.CalculationStatus
```python
class CalculationStatus(enum.Enum):
    COMPLETED                 = 1
    OUTSIDE_VICTIMS_BEAMWIDTH = 2
    DUCTING_OR_NO_LOS         = 3
```
- COMPLETED: Calculations completed successfully.
- OUTSIDE_VICTIMS_BEAMWIDTH: Calculations stopped. The line-of-sight between the two antennas falls outside of the victim's antenna beamwidth.
- DUCTING_OR_NO_LOS: Calcuations stopped. Either the line-of-sight between the two antennas is not free from ducting, or there is no line-of-sight.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FreeSpaceBasicTransmissionLoss
#### crc_covlib.helper.itur_p619.FreeSpaceBasicTransmissionLoss
```python
def FreeSpaceBasicTransmissionLoss(f_GHz: float, d_km: float) -> float
```
ITU-R P.619-5, Annex 1, Section 2.1\
Calculates the basic transmission loss assuming the complete radio path is in a vacuum with no obstruction, in dB.

Args:
- __f_GHz__ (float): Frequency (GHz).
- __d_km__ (float): Path length (km).

Returns:
- __L<sub>bfs</sub>__ (float): The free space basic transmission loss (dB).

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SpaceToEarthGaseousAttenuation
#### crc_covlib.helper.itur_p619.SpaceToEarthGaseousAttenuation
```python
def SpaceToEarthGaseousAttenuation(f_GHz: float, He_km: float, Hs_km: float,
                                   phi_e_deg: float|None, phi_s_deg: float,
                                   delta_phi_e_deg: float|None,
                                   refAtm: ReferenceAtmosphere=MAGRA, rho0_gm3: float=7.5
                                   ) -> tuple[float, CalculationStatus]
```
ITU-R P.619-5, Attachment C to Annex 1 (C.4)\
Calculates the attenuation due to atmospheric gases along a space-Earth propagation path (descending ray). For a space-Earth propagation path, the antenna of the space-based station is the transmitting antenna.

Args:
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000. Gaseous attenuation can be ignored below 1 GHz.
- __He_km__ (float): Height of Earth station (km above mean sea level), with 0 <= He_km < Hs_km.
- __Hs_km__ (float): Height of space-based station (km above mean sea level), with 0 <= He_km < Hs_km.
- __phi_e_deg__ (float|None): Elevation angle of the main beam of the earth based station (deg). If set to None, assumes the incident elevation angle is within the half-power beamwidth of the earth station. 0°=horizon, +90°=zenith, -90°=nadir.
- __phi_s_deg__ (float): Elevation angle of the main beam of the space-based station antenna (deg). 0°=horizon, +90°=zenith, -90°=nadir.
- __delta_phi_e_deg__ (float|None): Half-power beamwidth of the Earth based station (deg). If set to None, assumes the incident elevation angle is within the half-power beamwidth of the earth station.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).
        
Returns:
- __A<sub>g</sub>__ (float): The total gaseous attenuation along the space-Earth slant path (dB). Returns 0 (zero) when the attenuation could not be calculated (see status value then for more details).
- __status__ (crc_covlib.helper.itur_p619.CalculationStatus): COMPLETED if Ag could be calculated successfully. Otherwise the status value indicates the reason for stopping calculations.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### EarthToSpaceGaseousAttenuation
#### crc_covlib.helper.itur_p619.EarthToSpaceGaseousAttenuation
```python
def EarthToSpaceGaseousAttenuation(f_GHz: float, He_km: float, Hs_km: float,
                                   phi_e_deg: float, phi_s_deg: float|None,
                                   delta_phi_s_deg: float|None,
                                   refAtm: ReferenceAtmosphere=MAGRA, rho0_gm3: float=7.5
                                   ) -> tuple[float, CalculationStatus]
```
ITU-R P.619-5, Attachment C to Annex 1 (C.5)\
Calculates the attenuation due to atmospheric gases along an Earth-space propagation paths (ascending ray). For an Earth-space propagation path, the antenna of the earth based station is the transmitting antenna.

Args:
- __f_GHz__ (float): Frequency (GHz), with 1 <= f_GHz <= 1000. Gaseous attenuation can be ignored below 1 GHz.
- __He_km__ (float): Height of Earth station (km above mean sea level), with 0 <= He_km < Hs_km.
- __Hs_km__ (float): Height of space-based station (km above mean sea level), with 0 <= He_km < Hs_km.
- __phi_e_deg__ (float): Elevation angle of the main beam of the earth based station (deg). 0°=horizon, +90°=zenith, -90°=nadir.
- __phi_s_deg__ (float|None): Elevation angle of the main beam of the space-based station antenna (deg). If set to None, assumes the calculated elevation angle at the space station is within its half-power beamwidth. 0°=horizon, +90°=zenith, -90°=nadir.
- __delta_phi_s_deg__ (float|None): Half-power beamwidth of the space-based station antenna (deg). If set to None, assumes the calculated elevation angle at the space station is within its half-power beamwidth.
- __refAtm__ (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from ITU-R P.835-6.
- __rho0_gm3__ (float): Ground-level water vapour density (g/m3). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL (MAGRA).

Returns:
- __A<sub>g</sub>__ (float): The total gaseous attenuation along the Earth-space slant path (dB). Returns 0 (zero) when the attenuation could not be calculated (see status value then for more details).
- __status__ (crc_covlib.helper.itur_p619.CalculationStatus): COMPLETED if Ag could be calculated successfully. Otherwise the status value indicates the reason for stopping calculations.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### BeamSpreadingLoss
#### crc_covlib.helper.itur_p619.BeamSpreadingLoss
```python
def BeamSpreadingLoss(theta0_deg: float, h_km: float) -> float
```
ITU-R P.619-5, Annex 1, Section 2.4.2\
The signal loss due to beam spreading for a wave propagating through the total atmosphere in the Earth-space and space-Earth (dB). This effect is insignificant for elevation angles above 5 degrees.
    
Args:
- __theta0_deg__ (float): Free-space elevation angle (deg), with theta0_deg < 10.
- __h_km__ (float): Altitude of the lower point above sea level (km), 0 <= h_km <= 5.

Returns:
- __A<sub>bs</sub>__ (float): The signal loss due to beam spreading for a wave propagating through the total atmosphere in the Earth-space and space-Earth (dB).

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### TroposphericScintillationAttenuation
#### crc_covlib.helper.itur_p619.TroposphericScintillationAttenuation
```python
def TroposphericScintillationAttenuation(f_GHz: float, p: float, theta0_deg: float,
                                         Nwet: float, D: float, Ga: float|None=None,
                                         ) -> float
```
ITU-R P.619-5, Attachment D to Annex 1\
Gets the tropospheric scintillation attenuation (dB).

Args:
- __f_GHz__ (float): Frequency (GHz), with f_GHz <= 100.
- __p__ (float): Time percentage, with 0.001 <= p <= 99.999.
- __theta0_deg__ (float): Free-space elevation angle (degrees), with 5 <= theta0_deg <= 90.
- __Nwet__ (float): The median value of the wet term of the surface refractivity exceeded for the average year. See crc_covlib.helper.itur_p453.MedianAnnualNwet(lat, lon) function.
- __D__ (float): Physical diameter of the earth-station antenna (meters).
- __Ga__ (float|None): Antenna gain of the earth-based antenna, in the direction of the path (dBi). If specified, D is unused.
        
Returns:
- __A<sub>st</sub>__ (float): Tropospheric scintillation attenuation not exceeded for p percent time, in dB. Can be positive (attenuation) or negative (enhancement).

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### StraightLineEarthSpacePath
#### crc_covlib.helper.itur_p619.StraightLineEarthSpacePath
```python
def StraightLineEarthSpacePath(He_km: float, lat_e: float, lon_e: float, 
                               Hs_km: float, lat_s: float, lon_s: float, 
                               ) -> tuple[float, float, float]
```
ITU-R P.619-5, Attachment A to Annex 1\
Calculates the distance, elevation angle and azimuth of a space station as viewed from an earth-based station. It is based on spherical Earth geometry, and ignores the effect of atmospheric refraction.

Args:
- __He_km__ (float): Altitude of earth-based station, in km above sea level.
- __lat_e__ (float): Latitude of earth-based station, in degrees, with -90 <= lat_e <= 90.
- __lon_e__ (float): Longitude of earth-based station, in degrees with -180 <= lon_e <= 180.
- __Hs_km__ (float): Altitude of space station, in km above sea level.
- __lat_s__ (float): Latitude of sub-satellite point, in degrees, with -90 <= lat_s <= 90.
- __lon_s__ (float): Longitude of sub-satellite point, in degrees with -180 <= lon_s <= 180.
        
Returns:
- __D<sub>ts</sub>__ (float): Straight-line distance between the earth-based station and the space station, in km.
- __Ѳ<sub>0</sub>__ (float): Elevation angle of the straight line from the earth-based station to the space station, in degrees. It is the elevation angle of the ray at the earth-based station which would exist in the absence of tropospheric refraction, sometimes referred to as the free-space elevation angle. 0°=horizon, +90°=zenith, -90°=nadir.
- __azm_deg__ (float): Azimuth of the space station as viewed from an earth-based station, in degrees from true North. 0°=North, 90°=East, 180°=South, 270°=West. The azimuth is indeterminate if the elevation angle represents a vertical path.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### FreeSpaceToApparentElevationAngle
#### crc_covlib.helper.itur_p619.FreeSpaceToApparentElevationAngle
```python
def FreeSpaceToApparentElevationAngle(He_km: float, theta0_deg: float) -> float
```
ITU-R P.619-5, Attachment B to Annex 1\
Converts free-space elevation angle (at Earth-based station) to apparent elevation angle (at Earth-based station).

Args:
- __He_km__ (float): Altitude of earth-based station, in km above sea level, with 0 <= He_km <= 3.
- __theta0_deg__ (float): Free-space elevation angle, in degrees, with -1 <= theta_deg <= 10. It is he elevation angle calculated without taking atmospheric refraction into account.

Returns:
- __Ѳ__ (float): Apparent or actual elevation, in degrees. It is the elevation angle calculated taking atmospheric refraction into account. This is the optimum elevation angle for a high-gain antenna at the earth-based station intended to provide a link to the space station.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### ApparentToFreeSpaceElevationAngle
#### crc_covlib.helper.itur_p619.ApparentToFreeSpaceElevationAngle
```python
def ApparentToFreeSpaceElevationAngle(He_km: float, theta_deg: float) -> float
```
ITU-R P.619-5, Attachment B to Annex 1\
Converts apparent elevation angle (at Earth-based station) to free-space elevation angle (at Earth-based station).
    
Args:
- __He_km__ (float): Altitude of earth-based station, in km above sea level, with 0 <= He_km <= 3.
- __theta_deg__ (float): Apparent or actual elevation angle, in degrees. It is the elevation angle calculated taking atmospheric refraction into account. This is the optimum elevation angle for a high-gain antenna at the earth-based station intended to provide a link to the space station.
    
Returns:
- __Ѳ<sub>0</sub>__ (float): Free-space elevation angle, in degrees. It is he elevation angle calculated without taking atmospheric refraction into account.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### RayHeights
#### crc_covlib.helper.itur_p619.RayHeights
```python
def RayHeights(He_km: float, theta_deg: float, deltaDist_km: float=1
               ) -> tuple[list[float], list[float]]
```
ITU-R P.619-5, Attachment E to Annex 1\
Provides a method for tracing a ray launched from an earth-based station, in order to test whether it encounters an obstruction. It can be used to compile a profile of ray height relative to sea level, which can then be compared with a terrain profile.
    
Args:
- __He_km__ (float): Altitude of earth-based station, in km above sea level.
- __theta_deg__ (float): Apparent elevation angle at the earth-based station, in degrees. 0°=horizon, +90°=zenith, -90°=nadir.
- __deltaDist_km__ (float): Increment in horizontal distance over curved Earth (km).

Returns:
- __Dc_list__ (list[float]): Horizontal distances over curved earth from the earth-based station, in km.
- __Hr_list__ (list[float]): Ray heights over sea level, in km.

[Back to top](#itur_p619-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***