## antennas helper module
Additional antenna related functionalities in support of crc-covlib.

```python
from crc_covlib.helper import antennas 
```

- [LoadRadioMobileV3File](#loadradiomobilev3file)
- [LoadMsiPlanetFile](#loadmsiplanetfile)
- [LoadEdxFile](#loadedxfile)
- [LoadNsmaFile](#loadnsmafile)
- [SaveAs3DCsvFile](#saveas3dcsvfile)
- [Load3DCsvFile](#load3dcsvfile)
- [GenerateBeamformingAntennaPattern](#generatebeamformingantennapattern)
- [GetTotalIntegratedGain](#gettotalintegratedgain)
- [PlotPolar](#plotpolar)
- [PlotCartesian](#plotcartesian)
- [Plot3D](#plot3d)

***

### LoadRadioMobileV3File
#### crc_covlib.helper.antennas.LoadRadioMobileV3File
```python
def LoadRadioMobileV3File(sim: Simulation, terminal: Terminal, pathname: str,
                          normalize: bool=True) -> None
```
Loads a Radio Mobile antenna pattern file version 3 (usually *.ant) for the specified terminal's antenna.
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.
- __normalize__ (bool): Indicates whether to normalize the antenna pattern.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadMsiPlanetFile
#### crc_covlib.helper.antennas.LoadMsiPlanetFile
```python
def LoadMsiPlanetFile(sim: Simulation, terminal: Terminal, pathname: str, 
                      normalize: bool=True) -> None:
```
Loads a MSI Planet antenna pattern file (usually *.msi or *.prn) for the specified terminal's antenna.
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.
- __normalize__ (bool): Indicates whether to normalize the antenna pattern.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadEdxFile
#### crc_covlib.helper.antennas.LoadEdxFile
```python
def LoadEdxFile(sim: Simulation, terminal: Terminal, pathname: str,
                normalize: bool=True) -> None
```
Loads an EDX antenna pattern file (usually *.pat) for the specified terminal's antenna.
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.
- __normalize__ (bool): Indicates whether to normalize the antenna pattern.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### LoadNsmaFile
#### crc_covlib.helper.antennas.LoadNsmaFile
```python
def LoadNsmaFile(sim: Simulation, terminal: Terminal, pathname: str,
                 polari: str='', normalize: bool=True) -> None
```
Loads a NSMA antenna pattern file (usually *.adf) for the specified terminal's antenna.
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.
- __polari__ (str): By default the function tries to load any co-polarization pattern. If more than one is present, one may be selected using the polari argument. Example: 'V/V'.
- __normalize__ (bool): Indicates whether to normalize the antenna pattern.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### SaveAs3DCsvFile
#### crc_covlib.helper.antennas.SaveAs3DCsvFile
```python
def SaveAs3DCsvFile(sim: Simulation, terminal: Terminal, pathname: str,
                    approxMethod: PatternApproximationMethod=None,
                    azmStep_deg: int=1, elvStep_deg: int=1) -> None
```
Saves 3D pattern information from the specified terminal's antenna into a .csv file.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.
- __approxMethod__ (crc_covlib.simulation.PatternApproximationMethod): Approximation method for getting the gain from the antenna's horizontal and vertical patterns. If set to None, the antenna's approximation method will be used.
- __azmStep_deg__ (int): Azimuthal step, in degrees.
- __elvStep_deg__ (int): Elevational angle step, in degrees.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Load3DCsvFile
#### crc_covlib.helper.antennas.Load3DCsvFile
```python
def Load3DCsvFile(sim: Simulation, terminal: Terminal, pathname: str) -> None
```
Loads a 3D antenna pattern file for the specified terminal's antenna.
    
Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __pathname__ (str): Absolute or relative path for the antenna pattern file.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GenerateBeamformingAntennaPattern
#### crc_covlib.helper.antennas.GenerateBeamformingAntennaPattern
```python
def GenerateBeamformingAntennaPattern(sim: Simulation, terminal: Terminal,
                                      phi_3dB: float, theta_3dB: float, 
                                      Am: float, SLAv: float, GEmax: float, NH: int, NV: int,
                                      dH_over_wl: float, dV_over_wl: float,
                                      phi_escan_list: list, theta_etilt_list: list) -> None
```
Generates a beamforming antenna pattern for the specified terminal's antenna in accordance with the ITU-R M.2101-0 recommendation, Section 5 of Annex 1.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __phi_3dB__ (float): Horizontal 3dB bandwidth of single element, in degrees.
- __theta_3dB__ (float): Vertical 3dB bandwidth of single element, in degrees.
- __Am__ (float): Front-to-back ratio, in dB.
- __SLAv__ (float): Vertical sidelobe attenuation, in dB.
- __GEmax__ (float): Maximum gain of single element, in dBi.
- __NH__ (int): Number of columns in the array of elements.
- __NV__ (int): Number of rows in the array of elements.
- __dH_over_wl__ (float): Horizontal elements spacing over wavelength (dH/ʎ).
- __dV_over_wl__ (float): Vertical elements spacing over wavelength (dV/ʎ).
- __phi_escan_list__ (list): List of bearings (h angles) of formed beams, in degrees.
- __theta_etilt_list__ (list): List of tilts (v angles) of formed beams, in degrees (positive value for uptilt, negative for downtilt).

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### GetTotalIntegratedGain
#### crc_covlib.helper.antennas.GetTotalIntegratedGain
```python
def GetTotalIntegratedGain(sim: Simulation, terminal: Terminal) -> float
```
Calculates the total integrated gain (dBi) for the specified terminal's antenna.

Reference:
Ofcom, Enabling mmWave spectrum for new uses Annexes 5-8: supporting information,
https://www.ofcom.org.uk/__data/assets/pdf_file/0026/237266/annexes-5-8.pdf, p.10-11.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.

Returns:
- float: the total integrated gain, in dBi.

Example code:
```python
# Apply correction on a pattern to achieve a total integrated gain of 0 dBi.
# Note: may not want to do this on envelope patterns.
TX = simulation.Terminal.TRANSMITTER
tig = antennas.GetTotalIntegratedGain(sim, TX)
sim.SetAntennaMaximumGain(TX, sim.GetAntennaMaximumGain(TX)-tig)
```

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PlotPolar
#### crc_covlib.helper.antennas.PlotPolar
```python
def PlotPolar(sim: Simulation, terminal: Terminal, includeTilt: bool=True, 
              includeFixedBearing: bool=False) -> None
```
Plots the horizontal and vertical patterns of the specified terminal's antenna unto polar grids.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __includeTilt__ (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt and/or SetAntennaMechanicalTilt) should be included in the plot.
- __includeFixedBearing__ (bool): Indicates whether the antenna fixed bearing (set using SetAntennaBearing with TRUE_NORTH as the bearing reference) should be included in the plot.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### PlotCartesian
#### crc_covlib.helper.antennas.PlotCartesian
```python
def PlotCartesian(sim: Simulation, terminal: Terminal, includeTilt: bool=True, 
                  includeFixedBearing: bool=False) -> None
```
Plots the horizontal and vertical patterns of the specified terminal's antenna unto cartesian planes.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __includeTilt__ (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt and/or SetAntennaMechanicalTilt) should be included in the plot.
- __includeFixedBearing__ (bool): Indicates whether the antenna fixed bearing (set using SetAntennaBearing with TRUE_NORTH as the bearing reference) should be included in the plot.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### Plot3D
#### crc_covlib.helper.antennas.Plot3D
```python
def Plot3D(sim: Simulation, terminal: Terminal, includeTilt: bool=True,
           includeMaxGain: bool=False) -> None
```
Plots the gain of the specified terminal's antenna in 3D.

Args:
- __sim__ (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
- __terminal__ (crc_covlib.simulation.Terminal): Indicates either the transmitter or receiver terminal of the simulation.
- __includeTilt__ (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt and/or SetAntennaMechanicalTilt) should be included in the plot.
- __includeMaxGain__ (bool): Indicates whether the maximum gain value (set using SetAntennaMaximumGain) should be included in the plot.

[Back to top](#antennas-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***
