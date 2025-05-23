"""Additional antenna related functionalities in support of crc-covlib.
"""

from math import sin, cos, log10, radians, pi
from ..simulation import Simulation, Terminal, PatternApproximationMethod, BearingReference
from . import itur_m2101

__all__ = ['LoadRadioMobileV3File',
           'LoadNetworkPlannerFile',
           'LoadMsiPlanetFile',
           'LoadEdxFile',
           'LoadNsmaFile',
           'SaveAs3DCsvFile',
           'Load3DCsvFile',
           'GenerateBeamformingAntennaPattern',
           'GetTotalIntegratedGain',
           'PlotPolar',
           'PlotCartesian',
           'Plot3D']


def _AddMainVerticalCutEntry(sim: Simulation, terminal: Terminal, elevAngle_degrees: float,
                             gain_dB: float) -> None:
    """
    input parameter elevAngle_degrees:
      0        = astronomical horizon of front lobe
      90       = nadir
      180/-180 = astronomical horizon of back lobe
      270/-90  = zenith

    ...will be converted to crc-covlib's format, which is:
      (azm=0, elv=0)     = astronomical horizon of front lobe
      (azm=0, elv=90)    = nadir
      (azm=180, elv=90)  = nadir
      (azm=180, elv=0)   = astronomical horizon of back lobe
      (azm=0, elv=-90)   = zenith
      (azm=180, elv=-90) = zenith
    """
    while elevAngle_degrees < 0:
        elevAngle_degrees += 360
    
    if elevAngle_degrees <= 90:
        sim.AddAntennaVerticalPatternEntry(terminal, 0, elevAngle_degrees, gain_dB)
    elif elevAngle_degrees >= 270:
        sim.AddAntennaVerticalPatternEntry(terminal, 0, elevAngle_degrees-360, gain_dB)

    if elevAngle_degrees >= 90 and elevAngle_degrees <= 270:
        sim.AddAntennaVerticalPatternEntry(terminal, 180, 180-elevAngle_degrees, gain_dB)


def LoadRadioMobileV3File(sim: Simulation, terminal: Terminal, pathname: str,
                          normalize: bool=True) -> None:
    """
    Loads a Radio Mobile antenna pattern file version 3 (usually *.ant) for the specified
    terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        normalize (bool): Indicates whether to normalize the antenna pattern.
    """
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        for azm in range(0, 360):
            sim.AddAntennaHorizontalPatternEntry(terminal, azm, float(f.readline().strip()))
        for elv in range(-90, 270):
            _AddMainVerticalCutEntry(sim, terminal, elv, float(f.readline().strip()))

    if normalize:
        sim.NormalizeAntennaHorizontalPattern(terminal)
        sim.NormalizeAntennaVerticalPattern(terminal)


def LoadNetworkPlannerFile(sim: Simulation, terminal: Terminal, pathname: str, 
                      normalize: bool=True) -> None:
    """
    Loads a Google Network Planner antenna pattern file (usually *.csv) for the specified
    terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        normalize (bool): Indicates whether to normalize the antenna pattern.
    """
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            if line.startswith('----SpecMetadata----'):
                line = f.readline()
                line = f.readline()
                tokens = line.split(',')
                if len(tokens) >= 6:
                    gain_dBi = float(tokens[5])
                    sim.SetAntennaMaximumGain(terminal, gain_dBi)
            elif line.startswith('----PatternData----'):
                line = f.readline()
                line = f.readline()
                while line:
                    tokens = line.split(',')
                    if len(tokens) >= 3:
                        angle_deg = float(tokens[0])
                        azm_gain_dBi = float(tokens[1])
                        elev_gain_dBi = float(tokens[2])
                        azm_deg = angle_deg
                        if azm_deg < 0:
                            azm_deg += 360
                        elev_angle_deg = angle_deg
                        sim.AddAntennaHorizontalPatternEntry(terminal, azm_deg, azm_gain_dBi)
                        _AddMainVerticalCutEntry(sim, terminal, elev_angle_deg, elev_gain_dBi)
                    line = f.readline()
            line = f.readline()

    if normalize:
        sim.NormalizeAntennaHorizontalPattern(terminal)
        sim.NormalizeAntennaVerticalPattern(terminal)


def LoadMsiPlanetFile(sim: Simulation, terminal: Terminal, pathname: str, 
                      normalize: bool=True) -> None:
    """
    Loads a MSI Planet antenna pattern file (usually *.msi or *.prn) for the specified
    terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        normalize (bool): Indicates whether to normalize the antenna pattern.
    """    
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        line = f.readline().upper()
        while line:
            if line.startswith('GAIN'):
                tokens = line.split()
                if len(tokens) >= 2:
                    gain_dBd = float(tokens[1])
                    if len(tokens) >= 3:
                        if tokens[2] == 'DBI':
                            gain_dBd -= 2.15
                    sim.SetAntennaMaximumGain(terminal, gain_dBd + 2.15)
            elif line.startswith('HORIZONTAL'):
                num_entries = int(line.split()[1])
                for _ in range(0, num_entries):
                    [azm_str, gain_str] = f.readline().split()
                    sim.AddAntennaHorizontalPatternEntry(terminal, float(azm_str), -float(gain_str))
            elif line.startswith('VERTICAL'):
                num_entries = int(line.split()[1])
                for _ in range(0, num_entries):
                    [elv_str, gain_str] = f.readline().split()
                    _AddMainVerticalCutEntry(sim, terminal, float(elv_str), -float(gain_str))
            line = f.readline().upper()

    if normalize:
        sim.NormalizeAntennaHorizontalPattern(terminal)
        sim.NormalizeAntennaVerticalPattern(terminal)


def LoadEdxFile(sim: Simulation, terminal: Terminal, pathname: str,
                normalize: bool=True) -> None:
    """
    Loads an EDX antenna pattern file (usually *.pat) for the specified terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        normalize (bool): Indicates whether to normalize the antenna pattern.
    """   
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        tokens = f.readline().split(',')
        max_gain_dBi = float(tokens[1])
        sim.SetAntennaMaximumGain(terminal, max_gain_dBi)
        kypat = int(tokens[2]) # 1 for relative field strength, 2 for relative dB

        line = f.readline()
        while line:
            tokens = line.split(',')
            if tokens[0].startswith('999'):
                break
            azm = float(tokens[0])
            if azm < 0:
                azm += 360
            if kypat == 1:
                sim.AddAntennaHorizontalPatternEntry(terminal, azm, 20*log10(float(tokens[1])))
            else:
                sim.AddAntennaHorizontalPatternEntry(terminal, azm, float(tokens[1]))
            line = f.readline()
        
        tokens = f.readline().split(',')
        num_slices = int(tokens[0])
        nelv = int(tokens[1])
        for _ in range(0, num_slices):
            azm = int(float(f.readline().split()[0]))
            min_elv = max_elv = 0
            for _ in range(0, nelv):
                tokens = f.readline().split(',')
                elv = -float(tokens[0])
                min_elv = min(elv, min_elv)
                max_elv = max(elv, max_elv)
                if kypat == 1:
                    sim.AddAntennaVerticalPatternEntry(terminal,azm, elv, 20*log10(float(tokens[1])))
                else:
                    sim.AddAntennaVerticalPatternEntry(terminal, azm, elv, float(tokens[1]))

            # many EDX files have incomplete vertical patterns
            if min_elv > -85:
                sim.AddAntennaVerticalPatternEntry(terminal, azm, min_elv-1, -40)
            if max_elv < 85:
                sim.AddAntennaVerticalPatternEntry(terminal, azm, max_elv+1, -40)

    if normalize:
        sim.NormalizeAntennaHorizontalPattern(terminal)
        sim.NormalizeAntennaVerticalPattern(terminal)


def LoadNsmaFile(sim: Simulation, terminal: Terminal, pathname: str,
                 polari: str='', normalize: bool=True) -> None:
    """
    Loads a NSMA antenna pattern file (usually *.adf) for the specified terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        polari (str): By default the function tries to load any co-polarization pattern. 
            If more than one is present, one may be selected using the polari argument.
            Example: 'V/V'.
        normalize (bool): Indicates whether to normalize the antenna pattern.
    """  
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        line = f.readline().rstrip()
        while line:
            if line.startswith('GUNITS'):
                tokens = line.replace('/', ',').split(',')
                max_gain_unit = tokens[1]
                pattern_gain_unit = tokens[2]
            elif line.startswith('MDGAIN'):
                max_gain_dbi = float(line.split(',')[1])
                if max_gain_unit == 'DBD':
                    max_gain_dbi += 2.15
                sim.SetAntennaMaximumGain(terminal, max_gain_dbi)
            elif line.startswith('PATCUT'):
                pat_cut = line.split(',')[1]
                pols = f.readline().rstrip().split(',')[1]
                tokens = pols.split('/')
                is_co_pol = tokens[0] == tokens[1]
                num_points = int(f.readline().rstrip().split(',')[1])
                f.readline() # FSTLST
                if polari != '' and polari != pols:
                    # skip data, not the wanted polarizations
                    for _ in range(0, num_points):
                        f.readline()
                elif polari == '' and is_co_pol == False:
                    # skip cross-polarizaton data
                    for _ in range(0, num_points):
                        f.readline()
                else:
                    # clear pattern as file may contain more than one horizontal or vertical patterns
                    if pat_cut == 'AZ' or pat_cut == 'H':
                        sim.ClearAntennaPatterns(terminal, True, False)
                    elif pat_cut == 'EL' or pat_cut == 'V':
                        sim.ClearAntennaPatterns(terminal, False, True)
                    
                    for _ in range(0, num_points):
                        tokens = f.readline().rstrip().split(',')
                        angle = float(tokens[0])
                        gain = float(tokens[1])
                        
                        if pattern_gain_unit == 'DBI':
                            gain = gain - max_gain_dbi
                        elif pattern_gain_unit == 'DBD':
                            gain = (gain + 2.15) - max_gain_dbi
                        elif pattern_gain_unit == 'LIN':
                            gain = 20*log10(gain)

                        if pat_cut == 'AZ' or pat_cut == 'H':
                            azm = angle if angle >= 0 else angle+360
                            sim.AddAntennaHorizontalPatternEntry(terminal, azm, gain)
                        elif pat_cut == 'EL' or pat_cut == 'V':
                            _AddMainVerticalCutEntry(sim, terminal, -angle, gain)
                        # TODO implement reading Phi Cut if can find example file
            line = f.readline().rstrip()

    if normalize:
        sim.NormalizeAntennaHorizontalPattern(terminal)
        sim.NormalizeAntennaVerticalPattern(terminal)


def SaveAs3DCsvFile(sim: Simulation, terminal: Terminal, pathname: str,
                    approxMethod: PatternApproximationMethod=None,
                    azmStep_deg: int=1, elvStep_deg: int=1) -> None:
    """
    Saves 3D pattern information from the specified terminal's antenna into a .csv file.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
        approxMethod (crc_covlib.simulation.PatternApproximationMethod): Approximation 
            method for getting the gain from the antenna's horizontal and vertical patterns.
            If set to None, the antenna's approximation method will be used.
        azmStep_deg (int): Azimuthal step, in degrees.
        elvStep_deg (int): Elevational angle step, in degrees.
    """
    params = _SaveAntennaParams(sim, terminal)
    sim.SetAntennaBearing(terminal, BearingReference.TRUE_NORTH, 0)
    if approxMethod is not None:
        sim.SetAntennaPatternApproximationMethod(terminal, approxMethod)

    azms = range(0, 360+1, azmStep_deg)
    elvs = range(0, 180+1, elvStep_deg)
    with open(pathname, 'w', encoding='UTF-8') as f:
        f.write('3D')
        for azm in azms:
            f.write(';{}'.format(azm))
        f.write('\n')

        for elv in elvs:
            f.write('{}'.format(elv))
            for azm in azms:
                gain = sim.GetAntennaGain(terminal, azm, elv-90)
                f.write(';{:.2f}'.format(gain))
            f.write('\n')

    _RestoreAntennaParams(sim, terminal, params)


def Load3DCsvFile(sim: Simulation, terminal: Terminal, pathname: str) -> None:
    """
    Loads a 3D antenna pattern file for the specified terminal's antenna.
    
    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        pathname (str): Absolute or relative path for the antenna pattern file.
    """
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        tokens = f.readline().rstrip().split(';')
        azms = [int(azm) for azm in tokens[1:]]
        tokens = f.readline().rstrip().split(';')
        while tokens:
            if len(tokens) != len(azms)+1:
                break
            elv = int(tokens[0])
            gains = [float(gain) for gain in tokens[1:]]
            for azm, gain in zip(azms, gains):
                sim.AddAntennaVerticalPatternEntry(terminal, azm, elv-90, gain)
            tokens = f.readline().rstrip().split(';')

    max_gain = -sim.NormalizeAntennaVerticalPattern(terminal)
    sim.SetAntennaMaximumGain(terminal, max_gain)
    sim.SetAntennaPatternApproximationMethod(terminal, PatternApproximationMethod.V_PATTERN_ONLY)


"""
def SaveTo3DNsmaFile(sim: Simulation, terminal: Terminal, pathname: str) -> None:
    params = _SaveAntennaParams(sim, terminal)
    sim.SetAntennaMaximumGain(terminal, 0)
    sim.SetAntennaBearing(terminal, BearingReference.TRUE_NORTH, 0)
    sim.SetAntennaPatternApproximationMethod(terminal, PatternApproximationMethod.V_PATTERN_ONLY)

    with open(pathname, 'w', encoding='UTF-8') as f:
        f.write('REVNUM:,NSMA WG16.99.050\n')
        f.write('REVDAT:,19990520\n')
        f.write('ANTMAN:,N/A\n')
        f.write('MODNUM:,N/A\n')
        f.write('LOWFRQ:,0.0\n')
        f.write('HGHFRQ:,0.0\n')
        f.write('GUNITS:,DBI/DBR\n')
        f.write('MDGAIN:,{:.1f}\n'.format(params.max_gain))
        f.write('ELTILT:,0.0\n')
        f.write('PATTYP:,typical\n')
        f.write('NOFREQ:,1\n')
        f.write('PATFRE:,0.0\n')
        
        azms = range(0, 360, 1)
        elvs = range(-90, 90+1, 1)
        f.write('NUMCUT:,{}\n'.format(len(azms)))
        for azm in azms:
            f.write('PATCUT:,{}\n'.format(azm))
            f.write('POLARI:,V/V\n')
            f.write('NUPOIN:,{}\n'.format(len(elvs)))
            f.write('FSTLST:,0.000,180.000\n')
            for elv in elvs:
                f.write('{:.3f},{:.3f},\n'.format(elv+90, sim.GetAntennaGain(terminal, azm, elv)))
        f.write('ENDFIL:,EOF\n')

        _RestoreAntennaParams(sim, terminal, params)
"""


def GenerateBeamformingAntennaPattern(sim: Simulation, terminal: Terminal,
                                      phi_3dB: float, theta_3dB: float, 
                                      Am: float, SLAv: float, GEmax: float, NH: int, NV: int,
                                      dH_over_wl: float, dV_over_wl: float,
                                      phi_escan_list: list, theta_etilt_list: list) -> None:
    """
    Generates a beamforming antenna pattern for the specified terminal's antenna in accordance
    with the ITU-R M.2101-0 recommendation, Section 5 of Annex 1.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        phi_3dB (float): Horizontal 3dB bandwidth of single element, in degrees.
        theta_3dB (float): Vertical 3dB bandwidth of single element, in degrees.
        Am (float): Front-to-back ratio, in dB.
        SLAv (float): Vertical sidelobe attenuation, in dB.
        GEmax (float): Maximum gain of single element, in dBi.
        NH (int): Number of columns in the array of elements.
        NV (int): Number of rows in the array of elements.
        dH_over_wl (float): Horizontal elements spacing over wavelength (dH/ʎ).
        dV_over_wl (float): Vertical elements spacing over wavelength (dV/ʎ).
        phi_escan_list (list): List of bearings (h angles) of formed beams, in degrees.
        theta_etilt_list (list): List of tilts (v angles) of formed beams, in degrees
            (positive value for uptilt, negative for downtilt).
    """
    if len(phi_escan_list) == 0:
        raise ValueError('phi_escan_list must not be empty')
    if len(theta_etilt_list) == 0:
        raise ValueError('theta_etilt_list must not be empty')
    if len(phi_escan_list) != len(theta_etilt_list):
        raise ValueError('phi_escan_list and theta_etilt_list must have same length')
    
    sim.ClearAntennaPatterns(terminal, True, True)

    for azm in range(-180, 179+1, 1):
        for elv in range(0, 180+1, 1):
            max_gain = -300
            for phi_i_escan, theta_i_etilt in zip(phi_escan_list, theta_etilt_list):
                # change phi_i_escan and theta_i_etilt sign to follow usual convention in covlib
                gain = itur_m2101.IMTCompositeAntennaGain(azm, elv, phi_3dB, theta_3dB, Am,
                                                          SLAv, GEmax, NH, NV, dH_over_wl,
                                                          dV_over_wl, -phi_i_escan, -theta_i_etilt)
                max_gain = max(max_gain, gain)
            sim.AddAntennaVerticalPatternEntry(terminal, azm if azm >= 0 else azm+360, elv-90, max_gain)
            
    max_gain_dBi = -sim.NormalizeAntennaVerticalPattern(terminal)
    sim.SetAntennaMaximumGain(terminal, max_gain_dBi)
    sim.SetAntennaPatternApproximationMethod(terminal, PatternApproximationMethod.V_PATTERN_ONLY)


def GetTotalIntegratedGain(sim: Simulation, terminal: Terminal) -> float:
    """
    Calculates the total integrated gain (dBi) for the specified terminal's antenna.

    Reference:
    Ofcom, Enabling mmWave spectrum for new uses Annexes 5-8: supporting information,
    https://www.ofcom.org.uk/__data/assets/pdf_file/0026/237266/annexes-5-8.pdf, p.10-11.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.

    Returns:
        float: the total integrated gain, in dBi.

    Example code:
        # Apply correction on a pattern to achieve a total integrated gain of 0 dBi.
        # Note: may not want to do this on envelope patterns.
        TX = simulation.Terminal.TRANSMITTER
        tig = antennas.GetTotalIntegratedGain(sim, TX)
        sim.SetAntennaMaximumGain(TX, sim.GetAntennaMaximumGain(TX)-tig)
    """
    sum_linear = 0
    pt_count = 0
    for azm in range(0, 359+1, 1):
        for elv in range(-89, 89+1, 1):
            gain_dBi = sim.GetAntennaGain(terminal, azm, elv)
            sum_linear += pow(10.0, gain_dBi/10.0)*sin(radians(elv+90))
            pt_count += 1
    tig_linear = (pi/(2*pt_count))*sum_linear
    tig_dBi = 10.0*log10(tig_linear)
    return tig_dBi


class _AntennaParams:
    def __init__(self):
        self.max_gain = 0
        self.e_tilt = 0
        self.m_tilt = 0
        self.m_tilt_azm = 0
        self.bearing = 0
        self.bearing_ref = 0
        self.method = 0
"""
# or alternately dataclasses supported in python 3.7 and higher
class _AntennaParams:
    max_gain: float
    e_tilt: float
    m_tilt: float
    m_tilt_azm: float
    bearing: float
    bearing_ref: covlib.BearingReference
    method: covlib.PatternApproximationMethod
"""

def _SaveAntennaParams(sim: Simulation, terminal: Terminal) -> _AntennaParams:
    p = _AntennaParams()
    p.max_gain = sim.GetAntennaMaximumGain(terminal)
    p.e_tilt = sim.GetAntennaElectricalTilt(terminal)
    p.m_tilt = sim.GetAntennaMechanicalTilt(terminal)
    p.m_tilt_azm = sim.GetAntennaMechanicalTiltAzimuth(terminal)
    p.bearing = sim.GetAntennaBearing(terminal)
    p.bearing_ref = sim.GetAntennaBearingReference(terminal)
    p.method = sim.GetAntennaPatternApproximationMethod(terminal)
    return p


def _RestoreAntennaParams(sim: Simulation, terminal: Terminal, params: _AntennaParams) -> None:
    sim.SetAntennaMaximumGain(terminal, params.max_gain)
    sim.SetAntennaElectricalTilt(terminal, params.e_tilt)
    sim.SetAntennaMechanicalTilt(terminal, params.m_tilt, params.m_tilt_azm)
    sim.SetAntennaBearing(terminal, params.bearing_ref, params.bearing)
    sim.SetAntennaPatternApproximationMethod(terminal, params.method)


def _Cuts(sim: Simulation, terminal: Terminal, includeTilt: bool, includeFixedBearing: bool) -> list:
    params = _SaveAntennaParams(sim, terminal)
    sim.SetAntennaMaximumGain(terminal, 0)
    if includeTilt == False:
        sim.SetAntennaElectricalTilt(terminal, 0)
        sim.SetAntennaMechanicalTilt(terminal, 0, 0)
    if includeFixedBearing == False or sim.GetAntennaBearingReference(terminal) == BearingReference.OTHER_TERMINAL:
        sim.SetAntennaBearing(terminal, BearingReference.TRUE_NORTH, 0)

    if params.method != PatternApproximationMethod.V_PATTERN_ONLY:
        sim.SetAntennaPatternApproximationMethod(terminal, PatternApproximationMethod.H_PATTERN_ONLY)
    azms = range(0, 361, 1)
    h_gains = []
    for azm in azms:
        h_gains.append(sim.GetAntennaGain(terminal, azm, 0))

    sim.SetAntennaPatternApproximationMethod(terminal, PatternApproximationMethod.V_PATTERN_ONLY)
    # bearing to 0 so that the shown V cut is always at 0 and 180 deg of azimuth
    sim.SetAntennaBearing(terminal, BearingReference.TRUE_NORTH, 0)
    elvs = range(-90, 271, 1)
    v_gains = []
    for elv in elvs:
        if elv > 90:
            v_gains.append(sim.GetAntennaGain(terminal, 180, 180-elv))
        else:
            v_gains.append(sim.GetAntennaGain(terminal, 0, elv))

    _RestoreAntennaParams(sim, terminal, params)
    return [azms, h_gains, elvs, v_gains]


def PlotPolar(sim: Simulation, terminal: Terminal, includeTilt: bool=True, 
              includeFixedBearing: bool=False) -> None:
    """
    Plots the horizontal and vertical patterns of the specified terminal's antenna
    unto polar grids.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        includeTilt (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt
            and/or SetAntennaMechanicalTilt) should be included in the plot.
        includeFixedBearing (bool): Indicates whether the antenna fixed bearing (set using 
            SetAntennaBearing with TRUE_NORTH as the bearing reference) should be included
            in the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    [azms, h_gains, elvs, v_gains] = _Cuts(sim, terminal, includeTilt, includeFixedBearing)

    max_gain = max(max(h_gains), max(v_gains))
    max_limit_db = max(0, max_gain)
    min_limit_db = -40
    ticks_db = range(min_limit_db, int(max_limit_db)+10, 10)
    ticks_db_labels = []
    for g in ticks_db:
        if g%20==0 and (g!=0 or max_limit_db>0):
            ticks_db_labels.append(str(g)+'dB')
        else:
            ticks_db_labels.append('')
    ticks_deg = np.deg2rad(range(0, 360, 10))
    ticks_deg_h_labels = ['0°','','','30°','','','60°','','','90°','','','120°','','','150°','','',
                          '180°','','','210°','','','240°','','','270°','','','300°','','','330°','','']
    ticks_deg_v_labels = ['0°','','','30°','','','60°','','','90°','','','','','','','','',
                          '','','','','','','','','','-90°','','','-60°','','','-30°','','']

    ax1 = plt.subplot(121, projection='polar')
    ax1.set_title('Horizontal Plane')
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    #ax1.set_theta_zero_location('E')
    ax1.set_xticks(ticks_deg)
    ax1.set_xticklabels(ticks_deg_h_labels)
    ax1.set_ylim(min_limit_db, max_limit_db)
    ax1.set_yticks(ticks_db)
    ax1.set_yticklabels(ticks_db_labels)

    ax2 = plt.subplot(122, projection='polar')
    ax2.set_title('Vertical Plane')
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location('E')
    ax2.set_xticks(ticks_deg)
    ax2.set_xticklabels(ticks_deg_v_labels)
    ax2.set_ylim(min_limit_db, max_limit_db)
    ax2.set_yticks(ticks_db)
    ax2.set_yticklabels(ticks_db_labels)

    h_gains = [max(g, min_limit_db) for g in h_gains]
    v_gains = [max(g, min_limit_db) for g in v_gains]
    ax1.plot(np.deg2rad(azms), h_gains)
    ax2.plot(np.deg2rad(elvs), v_gains)
    plt.show()


def PlotCartesian(sim: Simulation, terminal: Terminal, includeTilt: bool=True, 
                  includeFixedBearing: bool=False) -> None:
    """
    Plots the horizontal and vertical patterns of the specified terminal's antenna unto
    cartesian planes.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        includeTilt (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt
            and/or SetAntennaMechanicalTilt) should be included in the plot.
        includeFixedBearing (bool): Indicates whether the antenna fixed bearing (set using 
            SetAntennaBearing with TRUE_NORTH as the bearing reference) should be included
            in the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    [azms, h_gains, elvs, v_gains] = _Cuts(sim, terminal, includeTilt, includeFixedBearing)
    azms = [azm if azm <= 180 else azm-360 for azm in azms] # set azimuths to be from -180 to 180
    elvs = [elv if elv <= 180 else elv-360 for elv in elvs] # set elev angles to be from -180 to 180
    azms = np.roll(azms, 180) # have first element be -180
    h_gains = np.roll(h_gains, 180)
    elvs = np.roll(elvs, 90) # have first element be -180
    v_gains = np.roll(v_gains, 90)

    max_gain = max(max(h_gains), max(v_gains))
    max_limit_db = max(0, max_gain)
    min_limit_db = -40
    ticks_db = range(min_limit_db, int(max_limit_db)+10, 10)
    ticks_db_labels = [str(g)+'dB' for g in ticks_db]
    
    ticks_deg = range(-180, 180+1, 60)
    #ticks_deg_h_labels = ['-180°','-20°','-60°','0°','60°','120°','180°']
    ticks_deg_h_labels = ['180°','240°','300°','0°','60°','120°','180°']
    ticks_deg_v_labels = ['-180°','-120°','-60°','0°','60°','120°','180°']

    ax1 = plt.subplot(121)
    ax1.set_title('Horizontal Plane')
    ax1.set_xticks(ticks_deg)
    ax1.set_xticklabels(ticks_deg_h_labels)
    ax1.set_ylim(min_limit_db, max_limit_db)
    ax1.set_yticks(ticks_db)
    ax1.set_yticklabels(ticks_db_labels)
    ax1.grid()

    ax2 = plt.subplot(122)
    ax2.set_title('Vertical Plane')
    ax2.set_xticks(ticks_deg)
    ax2.set_xticklabels(ticks_deg_v_labels)
    ax2.set_ylim(min_limit_db, max_limit_db)
    ax2.set_yticks(ticks_db)
    ax2.set_yticklabels(ticks_db_labels)
    ax2.grid()

    h_gains = [max(g, min_limit_db) for g in h_gains]
    v_gains = [max(g, min_limit_db) for g in v_gains]
    ax1.plot(azms, h_gains)
    ax2.plot(elvs, v_gains)
    plt.show()


def Plot3D(sim: Simulation, terminal: Terminal, includeTilt: bool=True,
           includeMaxGain: bool=False) -> None:
    """
    Plots the gain of the specified terminal's antenna in 3D.

    Args:
        sim (crc_covlib.simulation.Simulation): crc-covlib Simulation object.
        terminal (crc_covlib.simulation.Terminal): Indicates either the transmitter or
            receiver terminal of the simulation.
        includeTilt (bool): Indicates whether the antenna tilt (set using SetAntennaElectricalTilt
            and/or SetAntennaMechanicalTilt) should be included in the plot.
        includeMaxGain (bool): Indicates whether the maximum gain value (set using 
            SetAntennaMaximumGain) should be included in the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    params = _SaveAntennaParams(sim, terminal)
    if includeMaxGain == False:
        sim.SetAntennaMaximumGain(terminal, 0)
    sim.SetAntennaBearing(terminal, BearingReference.TRUE_NORTH, 0)
    if includeTilt == False:
        sim.SetAntennaElectricalTilt(terminal, 0)
        sim.SetAntennaMechanicalTilt(terminal, 0, 0)

    min_gain = -40
    azms = np.arange(0, 360.1, 1)
    elvs = np.arange(-90, 90.1, 1)
    x = np.zeros((len(azms), len(elvs)))
    y = np.zeros((len(azms), len(elvs)))
    z = np.zeros((len(azms), len(elvs)))

    i = 0
    for azm in azms:
        j = 0
        for elv in elvs:
            gain_dB = sim.GetAntennaGain(terminal, azm, elv)
            gain_dB = max(0, gain_dB + abs(min_gain))
            # sperical to cartesian coordinates conversion
            x[i][j] = min_gain + (gain_dB * sin(radians(elv+90)) * cos(radians(-azm)))
            y[i][j] = min_gain + (gain_dB * sin(radians(elv+90)) * sin(radians(-azm)))
            z[i][j] = min_gain + (gain_dB * cos(radians(elv+90)))
            j += 1
        i += 1

    _RestoreAntennaParams(sim, terminal, params)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, rcount=100, ccount=100)
    try:
        ax.set_aspect('equal')
    except:
        ax.set_aspect('auto')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    
    plt.show()
