"""Implementation of ITU-R P.676-13. Fully implemented except for Section 5 of Annex 1.

Note: Statistical (Stat) functions from Annex 2 support using both annual and monthly statistics
from the itur_p2145.py module. See the itur_p2145.py module for more details on how to install
statistics files.
"""

import os
from math import cos, sin, radians, degrees, sqrt, acos, floor, ceil, log, asin, exp, pow
import math
from typing import Union
import numpy as np
import numpy.typing as npt
from numba import jit
from .itur_p835 import ReferenceAtmosphere, MAGRA
from . import itur_p835
from . import itur_p2145


__all__ = ['GaseousAttenuation',
           'DryAirGaseousAttenuation',
           'WaterVapourGaseousAttenuation',
           'TerrestrialPathGaseousAttenuation',
           'RefractiveIndex',
           'SlantPathGaseousAttenuation',
           'EarthToSpaceReciprocalApparentElevAngle',
           'SpaceToEarthReciprocalApparentElevAngle',
           'InterceptsEarth',
           'PhaseDispersion',
           'DownwellingMicrowaveBrightnessTemperature',
           'UpwellingMicrowaveBrightnessTemperature',
           'SlantPathInstantOxygenGaseousAttenuation',
           'SlantPathStatOxygenGaseousAttenuation',
           'SlantPathInstantWaterVapourGaseousAttenuation1',
           'SlantPathInstantWaterVapourGaseousAttenuation2',
           'SlantPathStatWaterVapourGaseousAttenuation',
           'SlantPathStatGaseousAttenuation',
           'WeibullApproxAttenuation',
           'FIGURE_1',
           'FIGURE_2',
           'FIGURE_4',
           'FIGURE_6',
           'FIGURE_8',
           'FIGURE_9']


@jit(nopython=True)
def GaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                       rho_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - gamma
    Gaseous attenuation attributable to dry air and water vapour (dB/km).

    Args:
        f_GHz (float): frequency (GHz) [1, 1000]
        P_hPa (float): atmospheric pressure (hPa)
        T_K (float): temperature (K)
        rho_gm3 (float): water vapour density (g/m3)

    Returns:
        (float): Gaseous attenuation attributable to dry air and water vapour (dB/km).
    """
    yo = DryAirGaseousAttenuation(f_GHz, P_hPa, T_K, rho_gm3)
    yw = WaterVapourGaseousAttenuation(f_GHz, P_hPa, T_K, rho_gm3)
    return yo+yw # eq. (1)


@jit(nopython=True)
def DryAirGaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                             rho_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - gamma_o
    Specific gaseous attenuation attributable to dry air (dB/km).

    Args:
        f_GHz (float): frequency (GHz) [1, 1000]
        P_hPa (float): atmospheric pressure (hPa)
        T_K (float): temperature (K)
        rho_gm3 (float): water vapour density (g/m3)

    Returns:
        (float): Specific gaseous attenuation attributable to dry air (oxygen, pressure-induced 
            nitrogen and non-resonant Debye attenuation) (dB/km).
    """
    yo = 0.1820*f_GHz*_NppOxygen(f_GHz, P_hPa, T_K, rho_gm3) # eq. (1)
    return yo


@jit(nopython=True)
def WaterVapourGaseousAttenuation(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                                  rho_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - gamma_w
    Specific gaseous attenuation attributable to water vapour (dB/km).

    Args:
        f_GHz (float): frequency (GHz) [1, 1000]
        P_hPa (float): atmospheric pressure (hPa)
        T_K (float): temperature (K)
        rho_gm3 (float): water vapour density (g/m3)

    Returns:
        (float): Specific gaseous attenuation attributable to water vapour (dB/km).
    """
    yw = 0.1820*f_GHz*_NppWaterVapour(f_GHz, P_hPa, T_K, rho_gm3); # eq. (1)
    return yw


@jit(nopython=True)
def _NppOxygen(f: float, P: float, TK: float, rho: float) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - N''Oxygen

    Args:
        f (float): frequency (GHz) [1, 1000]
        P (float): atmospheric pressure (hPa)
        TK (float): temperature (K)
        rho (float): water vapour density (g/m3)

    Returns:
        (float): Imaginary part of the complex refractivity attributable to oxygen.
    """
    NUM_LINES = 44
    theta = 300.0/TK
    e = rho*TK/216.7
    NppO = 0.0
    prod1 = 1.0E-7*P*theta*theta*theta
    prod2 = 1.1*e*theta
    prod3 = 1.0E-4*(P+e)*pow(theta,0.8)
    for i in range(0, NUM_LINES, 1):
        fi = _TABLE1[i][0]
        a = _TABLE1[i]
        Si = a[1] * prod1 * exp(a[2]*(1.0-theta)) # eq. (3) for oxygen
        df = a[3]*1.0E-4*((P*pow(theta,0.8-a[4]))+prod2) # eq. (6a) for oxygen
        df = sqrt((df*df)+2.25E-6) # eq. (6b) for oxygen
        delta = (a[5]+(a[6]*theta))*prod3 # eq. (7) for oxygen
        # eq. (5)
        Fi = (f/fi)*( ((df-(delta*(fi-f)))/(((fi-f)*(fi-f))+(df*df))) +
                      ((df-(delta*(fi+f)))/(((fi+f)*(fi+f))+(df*df))) )
        NppO += Si*Fi # eq. (2a)
    d = 5.6E-4*(P+e)*pow(theta, 0.8) # eq. (9)
    # eq. (8)
    NppD = f*P*theta*theta*( (6.14E-5/(d*(1.0+((f*f)/(d*d))))) + 
                             ((1.4E-12*P*pow(theta,1.5))/(1.0+(1.9E-5*pow(f,1.5)))) )
    NppO += NppD
    return NppO


@jit(nopython=True)
def _NppWaterVapour(f: float, P: float, TK: float, rho: float) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - N''Water Vapour

    Args:
        f (float): frequency (GHz) [1, 1000]
        P (float): atmospheric pressure (hPa)
        TK (float): temperature (K)
        rho (float): water vapour density (g/m3)

    Returns:
        (float): Imaginary part of the complex refractivity attributable to water vapour.
    """
    NUM_LINES = 35
    theta = 300.0/TK
    e = rho*TK/216.7
    NppWV = 0.0
    prod1 = 1.0E-1*e*pow(theta, 3.5)
    for i in range(0, NUM_LINES, 1):
        fi = _TABLE2[i][0]
        b = _TABLE2[i]
        Si = b[1] * prod1 * exp(b[2]*(1.0-theta)) # eq. (3) for water vapour
        df = b[3]*1.0E-4*((P*pow(theta,b[4])) + (b[5]*e*pow(theta,b[6]))) # eq. (6a) for water vapour
        df = 0.535*df + sqrt((0.217*df*df)+(2.1316E-12*fi*fi/theta)) # eq. (6b) for water vapour
        # eq. (5) with delta=0 (eq.(7))
        Fi = (f/fi)*( (df/(((fi-f)*(fi-f))+(df*df))) + (df/(((fi+f)*(fi+f))+(df*df))) )
        NppWV += Si*Fi
    return NppWV


@jit(nopython=True)
def TerrestrialPathGaseousAttenuation(pathLength_km: float, f_GHz: float, P_hPa: float=1013.25,
                                      T_K: float=288.15, rho_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 2.1
    Path attenuation for a terrestrial path, or for slightly inclined paths close to the ground (dB).

    Args:
        pathLength_km (float): path length (km)
        f_GHz (float): frequency (GHz) [1, 1000]
        P_hPa (float): atmospheric pressure (hPa)
        T_K (float): temperature (K)
        rho_gm3 (float): water vapour density (g/m3)

    Returns:
        (float): Path attenuation for a terrestrial path, or for slightly inclined paths close to
            the ground (dB).
    """
    gamma = GaseousAttenuation(f_GHz, P_hPa, T_K, rho_gm3)
    A = pathLength_km*gamma
    return A


@jit(nopython=True)
def RefractiveIndex(h_km: float, refAtm: ReferenceAtmosphere=MAGRA, 
                    rho0_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 2.2.1
    Calculates the refractive index at any height for Earth-satellite paths.

    Args:
        h_km (float): Height in km, with h_km >= 0.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        (float): The refractive index at height h_km (for Earth-satellite paths).
    """
    # Outer space conventionnally starts at 100 km above sea level.
    # See Section 2.2.1 of Annex 1 from ITU-R P.676-13.
    if h_km > 100:
        return 1

    # Extract from ITU-R P.453-14:
    # "For Earth-satellite paths, the refractive index at any height is obtained using equations (1),
    # (2) and (10) above, together with the appropriate values for the parameters given in 
    # Recommendation ITU-R P.835, Annex 1."    

    P = itur_p835.Pressure(h_km, refAtm) # total atmospheric pressure (hPa)
    T = itur_p835.Temperature(h_km, refAtm) 
    rho = itur_p835.WaterVapourDensity(h_km, refAtm, rho0_gm3)
    e = rho*T/216.7 # water vapour pressure, in hPa. eq. (10) of P.453-14
    Pd = P - e # dry atmospheric pressure (hPa).  eq. (5) of P.453-14
    N = (77.6*Pd/T) + (72*e/T) + (3.75E5*e/(T*T)) # eq. (2) of P.453-14
    n = 1 + (N*1E-6) # eq. (1) of P.453-14
    return n


@jit(nopython=True)
def _LayerIndices(h_lower: float, h_upper: float) -> tuple[float, float]:
    """ITU-R P.676-13, Section 2.2.1 of Annex 1"""
    h_lower = max(0, h_lower)
    h_upper = min(100, h_upper)
    i_lower = floor(100*log(1E4*h_lower*((math.e**0.01)-1)+1)+1)
    i_upper = ceil(100*log(1E4*h_upper*((math.e**0.01)-1)+1)+1)
    return (i_lower, i_upper)


@jit(nopython=True)
def _LayerThickness(i: int, i_lower: int, i_upper: int, h_lower: float, h_upper: float) -> float:
    """ITU-R P.676-13, Section 2.2.1 of Annex 1"""
    if h_lower == 0 and h_upper > 100:
        delta_i_km =0.0001*math.e**((i-1)/100)
    else:
        m = ((math.e**0.02-math.e**0.01)/(math.e**(i_upper/100)-math.e**(i_lower/100)))*(h_upper-h_lower)
        delta_i_km = m*(math.e**((i-1)/100))
    return delta_i_km


@jit(nopython=True)
def _LayerBottomHeight(i: int, i_lower: int, i_upper: int, h_lower: float, h_upper: float) -> float:
    """ITU-R P.676-13, Section 2.2.1 of Annex 1"""
    if h_lower == 0 and h_upper > 100:
        h_i_km = 0.0001*(((math.e**((i-1)/100))-1)/((math.e**0.01)-1))
    else:
        m = ((math.e**0.02-math.e**0.01)/(math.e**(i_upper/100)-math.e**(i_lower/100)))*(h_upper-h_lower)
        h_i_km = h_lower + m*(((math.e**((i-1)/100))-(math.e**((i_lower-1)/100)))/((math.e**0.01)-1))
    return h_i_km


@jit(nopython=True)
def SlantPathGaseousAttenuation(f_GHz: float, h1_km: float, h2_km: float, phi1_deg: float,
                                refAtm: ReferenceAtmosphere=MAGRA,
                                rho0_gm3: float=7.5) -> tuple[float, float, float]:
    """
    ITU-R P.676-13, Annex 1, Section 2.2 (Slant paths)
    Calculates the Earth-space slant path gaseous attenuation for an ascending path between a
    location on or near the surface of the Earth and a location above the surface of the Earth or
    in space.

    Args:
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
        h1_km (float): Height of the first station (km), with 0 <= h1_km < h2_km.
        h2_km (float): Height of the second station (km), with 0 <= h1_km < h2_km.
        phi1_deg (float): The local apparent elevation angle at height h1_km (degrees).
            0°=horizon, +90°=zenith, -90°=nadir.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        Agas (float): The slant path gaseous attenuation (dB).
        bending (float): The total atmosphering bending along the path (radians).
        deltaL (float): The excess atmospheric path length (km).
    """
    if phi1_deg >= 0:
        return _NonNegSlantPathGaseousAtt(f_GHz, h1_km, h2_km, phi1_deg, refAtm, rho0_gm3)
    else:
        return _NegSlantPathGaseousAtt(f_GHz, h1_km, h2_km, phi1_deg, refAtm, rho0_gm3)


@jit(nopython=True)
def _NonNegSlantPathGaseousAtt(f_GHz: float, h1_km: float, h2_km: float, phi1_deg: float,
                               refAtm: ReferenceAtmosphere=MAGRA,
                               rho0_gm3: float=7.5) -> tuple[float, float, float]:
    """
    ITU-R P.676-13, Section 2.2.1 of Annex 1 (Non-negative apparent elevation angles)
    Calculates the Earth-space slant path gaseous attenuation for an ascending path between a
    location on or near the surface of the Earth and a location above the surface of the Earth or
    in space.

    Args:
        f_GHz (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
        h1_km (float): Height of the first station (km), with 0 <= h1_km < h2_km.
        h2_km (float): Height of the second station (km), with 0 <= h1_km < h2_km.
        phi1_deg (float): The local apparent elevation angle at height h1_km, in degrees, with
            phi1_deg >= 0°. 0°=horizon, +90°=zenith.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        Agas (float): The slant path gaseous attenuation (dB).
        bending (float): The total atmosphering bending along the path (radians).
        deltaL (float): The excess atmospheric path length (km).
    """
    Agas = 0
    bending = 0
    deltaL = 0
    Re = 6371
    r_1 = None
    n_1 = None
    beta_1_deg = 90-phi1_deg
    sin_beta_1 = sin(radians(beta_1_deg))

    h_lower = h1_km
    h_upper = h2_km
    (i_lower, i_upper) = _LayerIndices(h_lower, h_upper)
    for i in range(i_lower, i_upper, 1):
        h_i_bot = _LayerBottomHeight(i, i_lower, i_upper, h_lower, h_upper)
        delta_i = _LayerThickness(i, i_lower, i_upper, h_lower, h_upper)
        h_i_mid = h_i_bot+(delta_i/2)
        n_i = RefractiveIndex(h_i_mid, refAtm, rho0_gm3)
        r_i = Re + h_i_bot
        if r_1 is None:
            r_1 = r_i
            n_1 = n_i
        beta_i_rad = asin(((n_1*r_1)/(n_i*r_i))*sin_beta_1)
        cos_beta_i = cos(beta_i_rad)
        a_i = (-r_i*cos_beta_i)+sqrt((r_i*r_i*cos_beta_i*cos_beta_i)+(2*r_i*delta_i)+(delta_i*delta_i))
        y_i = GaseousAttenuation(f_GHz=f_GHz,
                                 # examples from CG-3M3J-13-ValEx-Rev8.1.1.xlsx use dry pressure
                                 P_hPa=itur_p835.DryPressure(h_i_mid, refAtm),
                                 T_K=itur_p835.Temperature(h_i_mid, refAtm),
                                 rho_gm3=itur_p835.WaterVapourDensity(h_i_mid, refAtm, rho0_gm3))
        Agas += a_i*y_i

        # For calculating the atmospheric bending along the path 
        if i != i_upper-1:
            h_ip1_bot = h_i_bot + delta_i
            delta_ip1 = _LayerThickness(i+1, i_lower, i_upper, h_lower, h_upper)
            h_ip1_mid = h_ip1_bot+(delta_ip1/2)
            n_ip1 = RefractiveIndex(h_ip1_mid, refAtm, rho0_gm3)
            r_ip1 = r_i + delta_i
            beta_ip1_rad = asin(((n_1*r_1)/(n_ip1*r_ip1))*sin_beta_1)
            alpha_i_rad = asin(((n_1*r_1)/(n_i*r_ip1))*sin_beta_1)
            bending += beta_ip1_rad-alpha_i_rad # ITU-R P.676-13, Section 2.2.4 of Annex 1

        # For calculating the excess atmospheric path length
        deltaL += a_i*(n_i-1) # ITU-R P.676-13, Section 2.2.5 of Annex 1

    return (Agas, bending, deltaL)


@jit(nopython=True)
def _NegSlantPathGaseousAtt(f_GHz: float, h1_km: float, h2_km: float, phi1_deg: float,
                            refAtm: ReferenceAtmosphere=MAGRA,
                            rho0_gm3: float=7.5) -> tuple[float, float, float]:
    """
    ITU-R P.676-13, Section 2.2.2 of Annex 1 (Negative apparent elevation angles)
    Calculates the Earth-space slant path gaseous attenuation for an ascending path between a
    location on or near the surface of the Earth and a location above the surface of the Earth or
    in space.

    Args:
        f_GHz (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
        h1_km (float): Height of the first station (km), with 0 <= h1_km < h2_km.
        h2_km (float): Height of the second station (km), with 0 <= h1_km < h2_km.
        phi1_deg (float): The local apparent elevation angle at height h1_km, in degrees, with
            phi1_deg < 0°. 0°=horizon, -90°=nadir.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        Agas (float): The slant path gaseous attenuation (dB).
        bending (float): The total atmosphering bending along the path (radians).
        deltaL (float): The excess atmospheric path length (km).
    """
    Re = 6371
    beta_1_deg = 90-phi1_deg
    sin_beta_1 = sin(radians(beta_1_deg))
    n_h1 = RefractiveIndex(h1_km, refAtm, rho0_gm3)

    hG_low = 0
    hG_high = h2_km
    hG = ((hG_high-hG_low)/2)+hG_low
    for _ in range (1, 24):
        n_hG = RefractiveIndex(hG, refAtm, rho0_gm3)
        gazing_test = (n_hG*(Re+hG))-(n_h1*(Re+h1_km)*sin_beta_1)
        if gazing_test > 0:
            hG_high = hG
        else:
            hG_low = hG
        hG = ((hG_high-hG_low)/2)+hG_low

    Agas1, bending1, deltaL1 = _NonNegSlantPathGaseousAtt(f_GHz, hG, h1_km, 0, refAtm, rho0_gm3)
    Agas2, bending2, deltaL2 = _NonNegSlantPathGaseousAtt(f_GHz, hG, h2_km, 0, refAtm, rho0_gm3)
    return (Agas1+Agas2, bending1+bending2, deltaL1+deltaL2)


@jit(nopython=True)
def EarthToSpaceReciprocalApparentElevAngle(He_km: float, Hs_km: float, phi_e_deg: float,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 2.2.3
    Calculates the reciprocal apparent elevation angle at the space-based station based on the
    apparent elevation angle at the Earth-based station.

    The gaseous attenuation for a space-Earth path, where the apparent elevation angle at the space
    station is phi_s_deg, is identical to the gaseous attenuation for the reciprocal Earth-space
    path, where the apparent elevation angle at the earth station is phi_e_deg.
    
    Args:
        He_km (float): Height of the Earth-based station (km).
        Hs_km (float): Height of the space-based station (km).
        phi_e_deg (float): Apparent elevation angle at the Earth-based station (deg).
            0°=horizon, +90°=zenith, -90°=nadir.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        phi_s_deg (float): Reciprocal apparent elevation angle at the space-based station (deg).
    """
    Re = 6371
    phi_e_rad = radians(phi_e_deg)
    rs = Re+Hs_km
    ns = RefractiveIndex(Hs_km, refAtm, rho0_gm3)
    re = Re+He_km
    ne = RefractiveIndex(He_km, refAtm, rho0_gm3)
    phi_s_rad = -acos(((re*ne)/(rs*ns))*cos(phi_e_rad))
    phi_s_deg = degrees(phi_s_rad)
    return phi_s_deg


@jit(nopython=True)
def SpaceToEarthReciprocalApparentElevAngle(He_km: float, Hs_km: float, phi_s_deg: float,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 2.2.3
    Calculates the reciprocal apparent elevation angle at the Earth-based station based on the
    apparent elevation angle at the space-based station.

    The gaseous attenuation for a space-Earth path, where the apparent elevation angle at the space
    station is phi_s_deg, is identical to the gaseous attenuation for the reciprocal Earth-space
    path, where the apparent elevation angle at the earth station is phi_e_deg.
    
    Args:
        He_km (float): Height of the Earth-based station (km).
        Hs_km (float): Height of the space-based station (km).
        phi_s_deg (float): Apparent elevation angle at the space-based station (deg).
            0°=horizon, +90°=zenith, -90°=nadir.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        phi_e_deg (float): Reciprocal apparent elevation angle at the Earth-based station (deg).
    """
    Re = 6371
    phi_s_rad = radians(phi_s_deg)
    rs = Re+Hs_km
    ns = RefractiveIndex(Hs_km, refAtm, rho0_gm3)
    re = Re+He_km
    ne = RefractiveIndex(He_km, refAtm, rho0_gm3)
    phi_e_rad = acos(((rs*ns)/(re*ne))*cos(phi_s_rad))
    phi_e_deg = degrees(phi_e_rad)
    return phi_e_deg


@jit(nopython=True)
def InterceptsEarth(He_km: float, Hs_km: float, phi_s_deg: float,
                    refAtm: ReferenceAtmosphere=MAGRA,
                    rho0_gm3: float=7.5) -> bool:
    """
    ITU-R P.676-13, Annex 1, Section 2.2.3
    Evaluates whether a space-Earth path intercepts the Earth.

    Args:
        He_km (float): Height of the Earth-based station (km).
        Hs_km (float): Height of the space-based station (km).
        phi_s_deg (float): Apparent elevation angle at the space-based station (deg).
            0°=horizon, +90°=zenith, -90°=nadir.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        (bool): True if the space-Earth path intercepts the Earth, False otherwise.
    """
    Re = 6371
    phi_s_rad = radians(phi_s_deg)
    rs = Re+Hs_km
    ns = RefractiveIndex(Hs_km, refAtm, rho0_gm3)
    re = Re+He_km
    ne = RefractiveIndex(He_km, refAtm, rho0_gm3)
    if ((rs*ns)/(re*ne))*cos(phi_s_rad) > 1:
        return False
    else:
        return True


@jit(nopython=True)
def PhaseDispersion(f_GHz: float, P_hPa: float=1013.25, T_K: float=288.15,
                    rho_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 3
    Calculates the gaseous phase dispersion (deg/km).

    Args:
        f_GHz (float): frequency (GHz) [1, 1000]
        P_hPa (float): atmospheric pressure (hPa)
        T_K (float): temperature (K)
        rho_gm3 (float): water vapour density (g/m3)

    Returns:
        (float): The gaseous phase dispersion (deg/km).

    """
    NpOxygen = _NpOxygen(f_GHz, P_hPa, T_K, rho_gm3)
    NpWaterVapour = _NpWaterVapour(f_GHz, P_hPa, T_K, rho_gm3)
    phi = -1.2008*f_GHz*(NpOxygen+NpWaterVapour) # eq. (24)
    return phi


@jit(nopython=True)
def _NpOxygen(f: float, P: float, TK: float, rho: float) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 1 - N'Oxygen

    Args:
        f (float): frequency (GHz) [1, 1000]
        P (float): atmospheric pressure (hPa)
        TK (float): temperature (K)
        rho (float): water vapour density (g/m3)

    Returns:
        (float): Real part of the complex refractivity attributable to oxygen.
    """
    NUM_LINES = 44
    theta = 300.0/TK
    e = rho*TK/216.7
    NpO = 0.0
    prod1 = 1.0E-7*P*theta*theta*theta
    prod2 = 1.1*e*theta
    prod3 = 1.0E-4*(P+e)*pow(theta,0.8)
    for i in range(0, NUM_LINES, 1):
        fi = _TABLE1[i][0]
        a = _TABLE1[i]
        Si = a[1] * prod1 * exp(a[2]*(1.0-theta)) # eq. (3) for oxygen
        df = a[3]*1.0E-4*((P*pow(theta,0.8-a[4]))+prod2) # eq. (6a) for oxygen
        df = sqrt((df*df)+2.25E-6) # eq. (6b) for oxygen
        delta = (a[5]+(a[6]*theta))*prod3 # eq. (7) for oxygen
        # eq. (25c)
        Fpi = (f/fi)*( (((fi-f)+(delta*df))/(((fi-f)*(fi-f))+(df*df))) - 
                       (((fi+f)+(delta*df))/(((fi+f)*(fi+f))+(df*df))) )
        NpO += Si*Fpi # eq. (25a)
    d = 5.6E-4*(P+e)*pow(theta, 0.8) # eq. (9)
    NpD = (-6.14E-5*P*theta*theta*f*f)/((f*f)+(d*d)) # eq. (25d)
    NpO += NpD # eq. (25a)
    return NpO


@jit(nopython=True)
def _NpWaterVapour(f: float, P: float, TK: float, rho: float) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 3 - N'Water Vapour

    Args:
        f (float): frequency (GHz) [1, 1000]
        P (float): atmospheric pressure (hPa)
        TK (float): temperature (K)
        rho (float): water vapour density (g/m3)

    Returns:
        (float): Real part of the complex refractivity attributable to water vapour.
    """
    NUM_LINES = 35
    theta = 300.0/TK
    e = rho*TK/216.7
    NpWV = 0.0
    prod1 = 1.0E-1*e*pow(theta, 3.5)
    for i in range(0, NUM_LINES, 1):
        fi = _TABLE2[i][0]
        b = _TABLE2[i]
        Si = b[1] * prod1 * exp(b[2]*(1.0-theta)) # eq. (3) for water vapour
        df = b[3]*1.0E-4*((P*pow(theta,b[4])) + (b[5]*e*pow(theta,b[6]))) # eq. (6a) for water vapour
        df = 0.535*df + sqrt((0.217*df*df)+(2.1316E-12*fi*fi/theta)) # eq. (6b) for water vapour
        # eq. (25c) with delta=0 (eq. (7))
        Fpi = (f/fi)*( ((fi-f)/(((fi-f)*(fi-f))+(df*df))) - ((fi+f)/(((fi+f)*(fi+f))+(df*df))) )
        NpWV += Si*Fpi # eq. (25b)
    return NpWV


@jit(nopython=True)
def _TB(f_GHz: float, Tj: float) -> float:
    TB = 0.048*f_GHz/(exp(0.048*f_GHz/Tj)-1) # eq. (26)
    return TB


@jit(nopython=True)
def DownwellingMicrowaveBrightnessTemperature(f_GHz: float, phi1_deg: float,
                                              h1_km: float=0, hk_km=100.1,
                                              refAtm: ReferenceAtmosphere=MAGRA,
                                              rho0_gm3: float=7.5) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 4.1
    The downwelling atmospheric microwave brightness temperature (K).
    
    Args:
        f_GHz (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
        phi1_deg (float): The local apparent elevation angle at height h1_km, in degrees, 
            with phi1_deg >= 0. 0°=horizon, +90°=zenith.
        h1_km (float): Height of atmospheric layer 1 (km).
        hk_km (float): Height of atmospheric layer k (km).
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        (float): The downwelling atmospheric microwave brightness temperature (K).
    """
    Re = 6371
    r_1 = None
    n_1 = None
    beta_1_deg = 90-phi1_deg
    sin_beta_1 = sin(radians(beta_1_deg))

    # step 1
    TB_dw = _TB(f_GHz, 2.73)

    h_lower = h1_km
    h_upper = hk_km
    (i_lower, i_upper) = _LayerIndices(h_lower, h_upper)
    for i in range(i_lower, i_upper, 1):
        j = i_upper-i

        # step 2
        TB_dw_last = TB_dw

        # step 3
        h_j_bot = _LayerBottomHeight(j, i_lower, i_upper, h_lower, h_upper)
        delta_j = _LayerThickness(j, i_lower, i_upper, h_lower, h_upper)
        h_j_mid = h_j_bot+(delta_j/2)
        Tj = itur_p835.Temperature(h_j_mid, refAtm)
        TB = _TB(f_GHz, Tj)

        # step 4
        n_j = RefractiveIndex(h_j_mid, refAtm, rho0_gm3)
        r_j = Re + h_j_bot
        if r_1 is None:
            r_1 = r_j
            n_1 = n_j
        beta_j_rad = asin(((n_1*r_1)/(n_j*r_j))*sin_beta_1)
        cos_beta_j = cos(beta_j_rad)
        a_j = (-r_j*cos_beta_j)+sqrt((r_j*r_j*cos_beta_j*cos_beta_j)+(2*r_j*delta_j)+(delta_j*delta_j))
        y_j = GaseousAttenuation(f_GHz=f_GHz,
                                 P_hPa=itur_p835.DryPressure(h_j_mid, refAtm),
                                 T_K=itur_p835.Temperature(h_j_mid, refAtm),
                                 rho_gm3=itur_p835.WaterVapourDensity(h_j_mid, refAtm, rho0_gm3))

        # step 4
        Lj = pow(10, (-a_j*y_j)/10)

        # step 5
        TB_dw = (TB_dw_last*Lj) + ((1-Lj)*TB)

    return TB_dw


@jit(nopython=True)
def UpwellingMicrowaveBrightnessTemperature(f_GHz: float, phi1_deg: float,
                                            h1_km: float=0, hk_km=100.1,
                                            refAtm: ReferenceAtmosphere=MAGRA,
                                            rho0_gm3: float=7.5, emissivity: float=0.95,
                                            TEarth_K: float=290) -> float:
    """
    ITU-R P.676-13, Annex 1, Section 4.2
    The upwelling atmospheric microwave brightness temperature (K).
    
    Args:
        f_GHz (float): Frequency in GHz, with 1 <= f_GHz <= 1000.
        phi1_deg (float): The local apparent elevation angle at height h1_km, in degrees, 
            with phi1_deg >= 0. 0°=horizon, +90°=zenith.
        h1_km (float): Height of atmospheric layer 1 (km).
        hk_km (float): Height of atmospheric layer k (km).
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).
        emissivity (float): Emissivity of the Earth's surface.
        TEarth_K (float): Temperature of the Earth's surface (K).

    Returns:
        (float): The upwelling atmospheric microwave brightness temperature (K).
    """
    # Note: In ITU-R P.676-13, it seems there is an error at equations (28a) and (28c). The -1
    # value is placed outside of the denominator, contrary to eq.(26). Using eq. (28a) and (28c)
    # as is yield results that are very different than those shown in FIGURE 9. However when using
    # -1 within the denominator like in eq.(26), results matches those in FIRGURE 9. 

    Re = 6371
    r_1 = None
    n_1 = None
    beta_1_deg = 90-phi1_deg
    sin_beta_1 = sin(radians(beta_1_deg))

    # step 1
    reflectivity = 1 - emissivity
    TB_dw = DownwellingMicrowaveBrightnessTemperature(f_GHz, phi1_deg, h1_km, hk_km, refAtm,
                                                      rho0_gm3)
    TB_uw = (emissivity*_TB(f_GHz, TEarth_K)) + (reflectivity*TB_dw)

    h_lower = h1_km
    h_upper = hk_km
    (j_lower, j_upper) = _LayerIndices(h_lower, h_upper)
    for j in range(j_lower, j_upper, 1):
        # step 2
        TB_uw_last = TB_uw

        # step 3
        h_j_bot = _LayerBottomHeight(j, j_lower, j_upper, h_lower, h_upper)
        delta_j = _LayerThickness(j, j_lower, j_upper, h_lower, h_upper)
        h_j_mid = h_j_bot+(delta_j/2)
        Tj = itur_p835.Temperature(h_j_mid, refAtm)
        TB = _TB(f_GHz, Tj)

        # step 4
        n_j = RefractiveIndex(h_j_mid, refAtm, rho0_gm3)
        r_j = Re + h_j_bot
        if r_1 is None:
            r_1 = r_j
            n_1 = n_j
        beta_j_rad = asin(((n_1*r_1)/(n_j*r_j))*sin_beta_1)
        cos_beta_j = cos(beta_j_rad)
        a_j = (-r_j*cos_beta_j)+sqrt((r_j*r_j*cos_beta_j*cos_beta_j)+(2*r_j*delta_j)+(delta_j*delta_j))
        y_j = GaseousAttenuation(f_GHz=f_GHz,
                                 P_hPa=itur_p835.DryPressure(h_j_mid, refAtm),
                                 T_K=itur_p835.Temperature(h_j_mid, refAtm),
                                 rho_gm3=itur_p835.WaterVapourDensity(h_j_mid, refAtm, rho0_gm3))

        # step 4
        Lj = pow(10, (-a_j*y_j)/10)

        # step 5
        TB_uw = (TB_uw_last*Lj) + ((1-Lj)*TB)

    return TB_uw


@jit(nopython=True)
def SlantPathInstantOxygenGaseousAttenuation(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                             Ts_K: float, rhoWs_gm3: float) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 1.1
    Gets the predicted slant path instantaneous gaseous attenuation attributable to oxygen (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        Ps_hPa (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired
            location.
        Ts_K (float): Instantaneous surface temperature, in K, at the desired location.
        rhoWs_gm3 (float): Instantaneous surface water vapour density, in g/m3, at the desired
            location.

    Returns:
        Ao (float): The predicted slant path instantaneous gaseous attenuation attributable to
            oxygen (dB).
    """
    es = rhoWs_gm3*Ts_K/216.7
    ps = Ps_hPa - es
    yo = DryAirGaseousAttenuation(f_GHz=f_GHz, P_hPa=ps, T_K=Ts_K, rho_gm3=rhoWs_gm3)
    ao, bo, co, do = _Part1Coeffs(f_GHz)
    ho = ao + bo*Ts_K + co*Ps_hPa + do*rhoWs_gm3
    Ao = yo*ho/sin(radians(theta_deg))
    return Ao


@jit(nopython=True)
def _SlantPathStatOxygenGaseousAttenuation(f_GHz: float, theta_deg: float,
                                           mPs_hPa: float, mTs_K: float, mRhoWs_gm3: float,
                                           Psp_hPa: float, Tsp_K: float, rhoWsp_gm3: float) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 1.2
    Gets the predicted slant path statistical gaseous attenuation attributable to oxygen (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        mPs_hPa (float): Mean total (barometric) surface pressure, in hPa, at the desired location.
        mTs_K (float): Mean surface temperature, in K, at the desired location.
        mRhoWs_gm3 (float): Mean surface water vapour density, in g/m3, at the desired location.
        Psp_hPa (float): Total (barometric) surface pressure at the exceedance probability p, in
            hPa, at the desired location.
        Tsp_K (float): Surface temperature at the exceedance probability p, in K, at the desired
            location.
        rhoWsp_gm3 (float): Surface water vapour density at the exceedance probability, p, in g/m3,
            at the desired location.

    Returns:
        Ao (float): The predicted slant path statistical gaseous attenuation attributable to
            oxygen (dB).
    """
    m_es = mRhoWs_gm3*mTs_K/216.7
    m_ps = mPs_hPa - m_es
    yo = DryAirGaseousAttenuation(f_GHz=f_GHz, P_hPa=m_ps, T_K=mTs_K, rho_gm3=mRhoWs_gm3)
    ao, bo, co, do = _Part1Coeffs(f_GHz)
    ho = ao + bo*Tsp_K + co*Psp_hPa + do*rhoWsp_gm3
    Ao = yo*ho/sin(radians(theta_deg))
    return Ao


def SlantPathStatOxygenGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                          lat: float, lon: float, month: Union[int, None]=None,
                                          h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 1.2
    Gets the predicted slant path statistical gaseous attenuation attributable to oxygen (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        Ao (float): The predicted slant path statistical gaseous attenuation attributable to
            oxygen (dB).
    """
    Psp_hPa = itur_p2145.SurfaceTotalPressure(p, lat, lon, month, h_mamsl)
    Tsp_K = itur_p2145.SurfaceTemperature(p, lat, lon, month, h_mamsl)
    rhoWsp_gm3 = itur_p2145.SurfaceWaterVapourDensity(p, lat, lon, month, h_mamsl)
    mPs_hPa = itur_p2145.MeanSurfaceTotalPressure(lat, lon, month, h_mamsl)
    mTs_K = itur_p2145.MeanSurfaceTemperature(lat, lon, month, h_mamsl)
    mRhoWs_gm3 = itur_p2145.MeanSurfaceWaterVapourDensity(lat, lon, month, h_mamsl)
    Ao = _SlantPathStatOxygenGaseousAttenuation(f_GHz, theta_deg, mPs_hPa, mTs_K, mRhoWs_gm3,
                                                Psp_hPa, Tsp_K, rhoWsp_gm3)
    return Ao


@jit(nopython=True)
def SlantPathInstantWaterVapourGaseousAttenuation1(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                                   Ts_K: float, rhoWs_gm3: float) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.1
    Gets the predicted slant path instantaneous gaseous attenuation attributable to water vapour,
    in dB, using method 1.
    
    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        Ps_hPa (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired
            location.
        Ts_K (float): Instantaneous surface temperature, in K, at the desired location.
        rhoWs_gm3 (float): Instantaneous surface water vapour density, in g/m3, at the desired
            location.

    Returns:
        Aw (float): The predicted slant path instantaneous gaseous attenuation attributable to
            water vapour (dB).
    """
    es = rhoWs_gm3*Ts_K/216.7
    ps = Ps_hPa - es
    yw = WaterVapourGaseousAttenuation(f_GHz=f_GHz, P_hPa=ps, T_K=Ts_K, rho_gm3=rhoWs_gm3)
    hw = 5.6585E-5*f_GHz + 1.8348
    hw += 2.6846/(((f_GHz-22.235080)**2)+2.7649)
    hw += 5.8905/(((f_GHz-183.310087)**2)+4.9219)
    hw += 2.9810/(((f_GHz-325.152888)**2)+3.0748)
    Aw = yw*hw/sin(radians(theta_deg))
    return Aw


@jit(nopython=True)
def SlantPathInstantWaterVapourGaseousAttenuation2(f_GHz: float, theta_deg: float, Ps_hPa: float,
                                                   Ts_K: float, rhoWs_gm3: float, Vs_kgm2: float
                                                   ) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.2
    Gets the predicted slant path instantaneous gaseous attenuation attributable to water vapour,
    in dB, using method 2.
    
    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        Ps_hPa (float): Instantaneous total (barometric) surface pressure, in hPa, at the desired
            location.
        Ts_K (float): Instantaneous surface temperature, in K, at the desired location.
        rhoWs_gm3 (float): Instantaneous surface water vapour density, in g/m3, at the desired
            location.
        Vs_kgm2 (float): Integrated water vapour content, in kg/m2 or mm, from the surface of the
            Earth at the desired location.

    Returns:
        Aw (float): The predicted slant path instantaneous gaseous attenuation attributable to
            water vapour (dB).
    """
    aV, bV, cV, dV = _Part2Coeffs(f_GHz)
    KV = aV + bV*rhoWs_gm3 + cV*Ts_K + dV*Ps_hPa
    Aw = KV*Vs_kgm2/sin(radians(theta_deg))
    return Aw


@jit(nopython=True)
def _SlantPathStatWaterVapourGaseousAttenuation(f_GHz: float, theta_deg: float,
                                                mPs_hPa: float, mTs_K: float, mRhoWs_gm3: float,
                                                Vsp_kgm2: float) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.3
    Gets the predicted slant path statistical gaseous attenuation attributable to water vapour (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        mPs_hPa (float): Mean total (barometric) surface pressure, in hPa, at the desired location.
        mTs_K (float): Mean surface temperature, in K, at the desired location.
        mRhoWs_gm3 (float): Mean surface water vapour density, in g/m3, at the desired location.
        Vsp_kgm2 (float): Integrated water vapour content at the exceedance probability p, in kg/m2
            or mm, from the surface of the Earth at the desired location.

    Returns:
        Aw (float): The predicted slant path statistical gaseous attenuation attributable to water
            vapour (dB).
    """
    aV, bV, cV, dV = _Part2Coeffs(f_GHz)
    KV = aV + bV*mRhoWs_gm3 + cV*mTs_K + dV*mPs_hPa
    Aw = KV*Vsp_kgm2/sin(radians(theta_deg))
    return Aw


def SlantPathStatWaterVapourGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                               lat: float, lon: float, month: Union[int, None]=None,
                                               h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.3
    Gets the predicted slant path statistical gaseous attenuation attributable to water vapour (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        Aw (float): The predicted slant path statistical gaseous attenuation attributable to water
            vapour (dB).
    """
    Vsp_kgm2 = itur_p2145.IntegratedWaterVapourContent(p, lat, lon, month, h_mamsl)
    mPs_hPa = itur_p2145.MeanSurfaceTotalPressure(lat, lon, month, h_mamsl)
    mTs_K = itur_p2145.MeanSurfaceTemperature(lat, lon, month, h_mamsl)
    mRhoWs_gm3 = itur_p2145.MeanSurfaceWaterVapourDensity(lat, lon, month, h_mamsl)
    Aw = _SlantPathStatWaterVapourGaseousAttenuation(f_GHz, theta_deg, mPs_hPa, mTs_K, mRhoWs_gm3,
                                                      Vsp_kgm2)
    return Aw


def SlantPathStatGaseousAttenuation(f_GHz: float, theta_deg: float, p: float,
                                    lat: float, lon: float, month: Union[int, None]=None,
                                    h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.676-13, Annex 2, Sections 1.2 and 2.3
    Gets the predicted slant path statistical gaseous attenuation attributable to both water vapour
    and oxygen (dB).
    
    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        Atotal (float): The predicted slant path statistical gaseous attenuation attributable to
            both water vapour and oxygen (dB).
    """
    Ao = SlantPathStatOxygenGaseousAttenuation(f_GHz, theta_deg, p, lat, lon, month, h_mamsl)
    Aw = SlantPathStatWaterVapourGaseousAttenuation(f_GHz, theta_deg, p, lat, lon, month, h_mamsl)
    return Ao+Aw


@jit(nopython=True)
def _WeibullApproxAttenuation(f_GHz: float, theta_deg: float, p: float,
                              mPs_hPa: float, mTs_K: float, mRhoWs_gm3: float,
                              lambdaVS: float, kVS: float) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.4
    Gets the Weibull approximation to the predicted slant path statistical gaseous attenuation
    attributable to water vapour (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        mPs_hPa (float): Mean total (barometric) surface pressure, in hPa, at the desired location.
        mTs_K (float): Mean surface temperature, in K, at the desired location.
        mRhoWs_gm3 (float): Mean surface water vapour density, in g/m3, at the desired location.
        lambdaVS (float): Surface Weibull water vapour scale parameter at the desired location.
        kVS (float): Surface Weibull water vapour shape parameter at the desired location.
    
    Returns:
        Aw (float): The Weibull approximation to the predicted slant path statistical gaseous
            attenuation attributable to water vapour (dB).
    """
    aV, bV, cV, dV = _Part2Coeffs(f_GHz)
    KV = aV + bV*mRhoWs_gm3 + cV*mTs_K + dV*mPs_hPa
    Aw = lambdaVS*KV*pow(-log(p/100), 1/kVS)/sin(radians(theta_deg))
    return Aw


def WeibullApproxAttenuation(f_GHz: float, theta_deg: float, p: float,
                             lat: float, lon: float, h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.676-13, Annex 2, Section 2.4
    Gets the Weibull approximation to the predicted slant path statistical gaseous attenuation
    attributable to water vapour (dB). This approximation is based on annual statistics.

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        Aw (float): The Weibull approximation to the predicted slant path statistical gaseous
            attenuation attributable to water vapour (dB).
    """
    mPs_hPa = itur_p2145.MeanSurfaceTotalPressure(lat, lon, None, h_mamsl)
    mTs_K = itur_p2145.MeanSurfaceTemperature(lat, lon, None, h_mamsl)
    mRhoWs_gm3 = itur_p2145.MeanSurfaceWaterVapourDensity(lat, lon, None, h_mamsl)
    kVS, lambdaVS = itur_p2145.WeibullParameters(lat, lon, h_mamsl)
    Aw = _WeibullApproxAttenuation(f_GHz, theta_deg, p, mPs_hPa, mTs_K, mRhoWs_gm3, lambdaVS, kVS)
    return Aw


@jit(nopython=True)
def _Part1Coeffs(f_GHz: float) -> tuple[float, float, float, float]:
    minFreq = 1
    freqInterval = 0.5
    numRows = 700
    row = int((f_GHz - minFreq) / freqInterval)
    if f_GHz >= 118.75: # there is an extra row at 118.75 GHz
        row += 1
    row = max(row, 0)
    row = min(row, numRows-2)
    f0, a0, b0, c0, d0 = _PART1[row]
    f1, a1, b1, c1, d1 = _PART1[row+1]
    a = (a0*(f1-f_GHz) + a1*(f_GHz-f0)) / (f1-f0)
    b = (b0*(f1-f_GHz) + b1*(f_GHz-f0)) / (f1-f0)
    c = (c0*(f1-f_GHz) + c1*(f_GHz-f0)) / (f1-f0)
    d = (d0*(f1-f_GHz) + d1*(f_GHz-f0)) / (f1-f0)
    return (a, b, c, d)


@jit(nopython=True)
def _Part2Coeffs(f_GHz: float) -> tuple[float, float, float, float]:
    minFreq = 1
    freqInterval = 0.5
    numRows = 699
    row = int((f_GHz - minFreq) / freqInterval)
    row = max(row, 0)
    row = min(row, numRows-2)
    f0, a0, b0, c0, d0 = _PART2[row]
    f1, a1, b1, c1, d1 = _PART2[row+1]
    a = (a0*(f1-f_GHz) + a1*(f_GHz-f0)) / (f1-f0)
    b = (b0*(f1-f_GHz) + b1*(f_GHz-f0)) / (f1-f0)
    c = (c0*(f1-f_GHz) + c1*(f_GHz-f0)) / (f1-f0)
    d = (d0*(f1-f_GHz) + d1*(f_GHz-f0)) / (f1-f0)
    return (a, b, c, d)


_TABLE1 = np.array([
    [50.474214,0.975,9.651,6.690,0.0,2.566,6.850],
    [50.987745,2.529,8.653,7.170,0.0,2.246,6.800],
    [51.503360,6.193,7.709,7.640,0.0,1.947,6.729],
    [52.021429,14.320,6.819,8.110,0.0,1.667,6.640],
    [52.542418,31.240,5.983,8.580,0.0,1.388,6.526],
    [53.066934,64.290,5.201,9.060,0.0,1.349,6.206],
    [53.595775,124.600,4.474,9.550,0.0,2.227,5.085],
    [54.130025,227.300,3.800,9.960,0.0,3.170,3.750],
    [54.671180,389.700,3.182,10.370,0.0,3.558,2.654],
    [55.221384,627.100,2.618,10.890,0.0,2.560,2.952],
    [55.783815,945.300,2.109,11.340,0.0,-1.172,6.135],
    [56.264774,543.400,0.014,17.030,0.0,3.525,-0.978],
    [56.363399,1331.800,1.654,11.890,0.0,-2.378,6.547],
    [56.968211,1746.600,1.255,12.230,0.0,-3.545,6.451],
    [57.612486,2120.100,0.910,12.620,0.0,-5.416,6.056],
    [58.323877,2363.700,0.621,12.950,0.0,-1.932,0.436],
    [58.446588,1442.100,0.083,14.910,0.0,6.768,-1.273],
    [59.164204,2379.900,0.387,13.530,0.0,-6.561,2.309],
    [59.590983,2090.700,0.207,14.080,0.0,6.957,-0.776],
    [60.306056,2103.400,0.207,14.150,0.0,-6.395,0.699],
    [60.434778,2438.000,0.386,13.390,0.0,6.342,-2.825],
    [61.150562,2479.500,0.621,12.920,0.0,1.014,-0.584],
    [61.800158,2275.900,0.910,12.630,0.0,5.014,-6.619],
    [62.411220,1915.400,1.255,12.170,0.0,3.029,-6.759],
    [62.486253,1503.000,0.083,15.130,0.0,-4.499,0.844],
    [62.997984,1490.200,1.654,11.740,0.0,1.856,-6.675],
    [63.568526,1078.000,2.108,11.340,0.0,0.658,-6.139],
    [64.127775,728.700,2.617,10.880,0.0,-3.036,-2.895],
    [64.678910,461.300,3.181,10.380,0.0,-3.968,-2.590],
    [65.224078,274.000,3.800,9.960,0.0,-3.528,-3.680],
    [65.764779,153.000,4.473,9.550,0.0,-2.548,-5.002],
    [66.302096,80.400,5.200,9.060,0.0,-1.660,-6.091],
    [66.836834,39.800,5.982,8.580,0.0,-1.680,-6.393],
    [67.369601,18.560,6.818,8.110,0.0,-1.956,-6.475],
    [67.900868,8.172,7.708,7.640,0.0,-2.216,-6.545],
    [68.431006,3.397,8.652,7.170,0.0,-2.492,-6.600],
    [68.960312,1.334,9.650,6.690,0.0,-2.773,-6.650],
    [118.750334,940.300,0.010,16.640,0.0,-0.439,0.079],
    [368.498246,67.400,0.048,16.400,0.0,0.000,0.000],
    [424.763020,637.700,0.044,16.400,0.0,0.000,0.000],
    [487.249273,237.400,0.049,16.000,0.0,0.000,0.000],
    [715.392902,98.100,0.145,16.000,0.0,0.000,0.000],
    [773.839490,572.300,0.141,16.200,0.0,0.000,0.000],
    [834.145546,183.100,0.145,14.700,0.0,0.000,0.000]
])


_TABLE2 = np.array([
    [22.235080,.1079,2.144,26.38,.76,5.087,1.00],
    [67.803960,.0011,8.732,28.58,.69,4.930,.82],
    [119.995940,.0007,8.353,29.48,.70,4.780,.79],
    [183.310087,2.273,.668,29.06,.77,5.022,.85],
    [321.225630,.0470,6.179,24.04,.67,4.398,.54],
    [325.152888,1.514,1.541,28.23,.64,4.893,.74],
    [336.227764,.0010,9.825,26.93,.69,4.740,.61],
    [380.197353,11.67,1.048,28.11,.54,5.063,.89],
    [390.134508,.0045,7.347,21.52,.63,4.810,.55],
    [437.346667,.0632,5.048,18.45,.60,4.230,.48],
    [439.150807,.9098,3.595,20.07,.63,4.483,.52],
    [443.018343,.1920,5.048,15.55,.60,5.083,.50],
    [448.001085,10.41,1.405,25.64,.66,5.028,.67],
    [470.888999,.3254,3.597,21.34,.66,4.506,.65],
    [474.689092,1.260,2.379,23.20,.65,4.804,.64],
    [488.490108,.2529,2.852,25.86,.69,5.201,.72],
    [503.568532,.0372,6.731,16.12,.61,3.980,.43],
    [504.482692,.0124,6.731,16.12,.61,4.010,.45],
    [547.676440,.9785,.158,26.00,.70,4.500,1.00],
    [552.020960,.1840,.158,26.00,.70,4.500,1.00],
    [556.935985,497.0,.159,30.86,.69,4.552,1.00],
    [620.700807,5.015,2.391,24.38,.71,4.856,.68],
    [645.766085,.0067,8.633,18.00,.60,4.000,.50],
    [658.005280,.2732,7.816,32.10,.69,4.140,1.00],
    [752.033113,243.4,.396,30.86,.68,4.352,.84],
    [841.051732,.0134,8.177,15.90,.33,5.760,.45],
    [859.965698,.1325,8.055,30.60,.68,4.090,.84],
    [899.303175,.0547,7.914,29.85,.68,4.530,.90],
    [902.611085,.0386,8.429,28.65,.70,5.100,.95],
    [906.205957,.1836,5.110,24.08,.70,4.700,.53],
    [916.171582,8.400,1.441,26.73,.70,5.150,.78],
    [923.112692,.0079,10.293,29.00,.70,5.000,.80],
    [970.315022,9.009,1.919,25.50,.64,4.940,.67],
    [987.926764,134.6,.257,29.85,.68,4.550,.90],
    [1780.000000,17506.,.952,196.3,2.00,24.15,5.00]
])


def _LoadAnnex2Coefficients() -> tuple[npt.ArrayLike, npt.ArrayLike]:
    curDir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    pathname = os.path.join(curDir, 'data/itu_proprietary/p676/R11010000020001TXTM.txt')
    part1 = np.loadtxt(fname=pathname)
    pathname = os.path.join(curDir, 'data/itu_proprietary/p676/R11010000020002TXTM.txt')
    part2 = np.loadtxt(fname=pathname)
    return (part1, part2)


_PART1, _PART2 = _LoadAnnex2Coefficients()


def FIGURE_1() -> None:
    """
    Calculates values and display FIGURE 1 from ITU-R P.676-13.
    """
    import matplotlib.pyplot as plt
    freqs = []
    atts = [[],[]]
    for f_GHz in range(1, 1001, 1):
        freqs.append(f_GHz)
        atts[0].append(GaseousAttenuation(f_GHz, 1013.25, 288.15, 7.5))
        atts[1].append(GaseousAttenuation(f_GHz, 1013.25, 288.15, 0))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_yscale('log')
    ax1.set_xlim([0,1000])
    ax1.set_xticks([*range(0, 1000+1, 100)])
    ax1.set_ylim([1E-3, 1E5])
    ax1.set_yticks([1E-3,1E-2,1E-1,1,1E1,1E2,1E3,1E4,1E5])
    ax1.plot(freqs, atts[0], color='#FF0000', label='standard')
    ax1.plot(freqs, atts[1], color='#0000FF', label='dry')
    ax1.set_title('FIGURE 1\nSpecific attenuation due to atmospheric gases, calculated at 1 GHz ' \
                   'intervals, including line centres')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Specific attenuation (dB/km)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_2() -> None:
    """
    Calculates values and display FIGURE 2 from ITU-R P.676-13.
    """
    import matplotlib.pyplot as plt
    freqs = []
    dists = [0,5,10,15,20]
    atts = [[],[],[],[],[]]
    for f_GHz in np.arange(50, 70+0.005, 0.01):
        freqs.append(f_GHz)
        for i in range(0, 5, 1):
            d_km = dists[i]
            atts[i].append(GaseousAttenuation(f_GHz, itur_p835.Pressure(d_km),
                                              itur_p835.Temperature(d_km),
                                              itur_p835.WaterVapourDensity(d_km)))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_yscale('log')
    ax1.set_xlim([50,70])
    ax1.set_xticks([*range(50, 70+1, 2)])
    ax1.set_ylim([1E-3, 1E2])
    ax1.set_yticks([1E-3,1E-2,1E-1,1,1E1,1E2])
    ax1.plot(freqs, atts[0], color='#FF0000', label='0 km')
    ax1.plot(freqs, atts[1], color='#0000FF', label='5 km')
    ax1.plot(freqs, atts[2], color='#22814C', label='10 km')
    ax1.plot(freqs, atts[3], color='#8000FF', label='15 km')
    ax1.plot(freqs, atts[4], color='#00A0E3', label='20 km')
    ax1.set_title('FIGURE 2')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Specific attenuation (dB/km)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_4() -> None:
    import matplotlib.pyplot as plt
    freqs = []
    atts = [[],[]]
    for f_GHz in range(1, 1001, 1):
        freqs.append(f_GHz)
        att_std_dB, _, _ = SlantPathGaseousAttenuation(f_GHz=f_GHz, h1_km=0, h2_km=100.01, phi1_deg=90,
                                                       refAtm=MAGRA, rho0_gm3=7.5)
        atts[0].append(att_std_dB)
        # Note: The "peaks" of the dry curve generated here do not perfectly match those from
        # the figure in the recommendation. Not sure what input parameters were used.
        att_dry_dB, _, _ = SlantPathGaseousAttenuation(f_GHz=f_GHz, h1_km=0, h2_km=100.01, phi1_deg=90,
                                                       refAtm=MAGRA, rho0_gm3=0)
        atts[1].append(att_dry_dB)
        print('{} GHz, {:.3f} dB, {:.3f} dB'.format(f_GHz, att_std_dB, att_dry_dB))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_yscale('log')
    ax1.set_xlim([0,1000])
    ax1.set_xticks([*range(0, 1000+1, 100)])
    ax1.set_ylim([1E-3, 1E5])
    ax1.set_yticks([1E-3,1E-2,1E-1,1,1E1,1E2,1E3,1E4,1E5])
    ax1.plot(freqs, atts[0], color='#FF0000', label='standard')
    ax1.plot(freqs, atts[1], color='#0000FF', label='dry')
    ax1.set_title('FIGURE 4\nSpecific attenuation due to atmospheric gases')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Zenith attenuation (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_6() -> None:
    """
    Calculates values and display FIGURE 6 from ITU-R P.676-13.
    """
    import matplotlib.pyplot as plt
    freqs = []
    disps = []
    for f_GHz in np.arange(0, 100.01, 0.5):
        freqs.append(f_GHz)
        disps.append(PhaseDispersion(f_GHz))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,100])
    ax1.set_xticks([*range(0, 100+1, 5)])
    ax1.set_ylim([-100, 100])
    ax1.set_yticks([*range(-100, 100+1, 10)])
    ax1.plot(freqs, disps, color='#000000')
    ax1.set_title('FIGURE 6\nThe frequency dependent specific phase dispersion for a standard atmosphere')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Frequency dependent specific phase dispersion (deg/km)')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_8() -> None:
    """
    Calculates values and display FIGURE 8 from ITU-R P.676-13.
    """
    import matplotlib.pyplot as plt
    freqs = []
    temperatures = []
    for f_GHz in range(1, 1000+1, 1):
        freqs.append(f_GHz)
        T = DownwellingMicrowaveBrightnessTemperature(f_GHz, 90)
        temperatures.append(T)
        print('{} GHz, {:.3f} K'.format(f_GHz, T))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,1000])
    ax1.set_xticks([*range(0, 1000+1, 100)])
    ax1.set_ylim([0, 300])
    ax1.set_yticks([*range(0, 300+1, 25)])
    ax1.plot(freqs, temperatures, color='#000000')
    ax1.set_title('FIGURE 8\nZenith downwelling microwave brightness temperature for a standard atmosphere')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Net downwelling microwave brightness temperature (K)')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_9() -> None:
    """
    Calculates values and display FIGURE 9 from ITU-R P.676-13.
    """
    import matplotlib.pyplot as plt
    freqs = []
    temperatures = []
    for f_GHz in range(1, 1000+1, 1):
        freqs.append(f_GHz)
        T = UpwellingMicrowaveBrightnessTemperature(f_GHz, -90)
        temperatures.append(T)
        print('{} GHz, {:.3f} K'.format(f_GHz, T))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,1000])
    ax1.set_xticks([*range(0, 1000+1, 100)])
    ax1.set_ylim([0, 300])
    ax1.set_yticks([*range(0, 300+1, 25)])
    ax1.plot(freqs, temperatures, color='#000000')
    ax1.set_title('FIGURE 9\nZenith upwelling microwave brightness temperature for a standard atmosphere')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Net upwelling microwave brightness temperature (K)')
    plt.grid(True, 'both','both')
    plt.show()
