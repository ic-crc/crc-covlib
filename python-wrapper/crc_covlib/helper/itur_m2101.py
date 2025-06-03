# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of ITU-R M.2021-0, Annex 1, Section 5.
"""

from math import pi, cos, sin, radians, sqrt, log10
import cmath
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['IMTAntennaElementGain',
           'IMTCompositeAntennaGain']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _A_EH(phi: float, phi_3dB: float, Am: float) -> float:
    """
    Horizontal radiation pattern gain (dB) of a single antenna element.
    See ITU-R M.2021-0, Annex 1, Section 5.1

    Args:
        phi (float): azimuth in degrees, from -180 to 180
        phi_3dB (float): horizontal 3dB bandwidth of single element, in degrees
        Am (float): front-to-back ratio, in dB

    Returns:
        float: horizontal radiation pattern gain (dB) of a single antenna element
    """
    aeh = -min(12*phi*phi/(phi_3dB*phi_3dB), Am)
    return aeh


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _A_EV(theta: float, theta_3dB: float, SLAv: float) -> float:
    """
    Vertical radiation pattern gain (dB) of a single antenna element.
    See ITU-R M.2021-0, Annex 1, Section 5.1

    Args:
        theta (float): elevation angle in degrees, from 0 (zenith) to 180 (nadir)
        theta_3dB (float): vertical 3dB bandwidth of single element, in degrees
        SLAv (float): vertical sidelobe attenuation, in dB

    Returns:
        float: vertical radiation pattern gain (dB) of a single antenna element
    """
    aev = -min(12*(theta-90)*(theta-90)/(theta_3dB*theta_3dB), SLAv)
    return aev


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def IMTAntennaElementGain(phi: float, theta: float, phi_3dB: float, theta_3dB: float, 
                          Am: float, SLAv: float, GEmax: float) -> float:
    """
    Gets the gain (dBi) of a single IMT (International Mobile Telecommunications) antenna element
    at the specified azimuth and elevation angle. See ITU-R M.2021-0, Annex 1, Section 5.1 for details.

    Args:
        phi (float): Azimuth in degrees, from -180 to 180.
        theta (float): Elevation angle in degrees, from 0 (zenith) to 180 (nadir).
        phi_3dB (float): Horizontal 3dB bandwidth of single element, in degrees.
        theta_3dB (float): Vertical 3dB bandwidth of single element, in degrees.
        Am (float): Front-to-back ratio, in dB.
        SLAv (float): Vertical sidelobe attenuation, in dB.
        GEmax (float): Maximum gain of single element, in dBi.

    Returns:
        float: gain of a single antenna element, in dBi
    """
    gemax = GEmax - min(-(_A_EH(phi, phi_3dB, Am)+_A_EV(theta, theta_3dB, SLAv)), Am)
    return gemax


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def IMTCompositeAntennaGain(phi: float, theta: float, phi_3dB: float, theta_3dB: float, 
                            Am: float, SLAv: float, GEmax: float, NH: int, NV: int,
                            dH_over_wl: float, dV_over_wl: float,
                            phi_i_escan: float, theta_i_etilt: float) -> float:
    """
    Gets the gain (dBi) of an IMT (International Mobile Telecommunications) composite antenna 
    (i.e. beamforming antenna) at the specified azimuth and elevation angle. See ITU-R M.2021-0,
    Annex 1, Section 5.2 for details.

    Args:
        phi (float): Azimuth at which to get the gain from, in degrees, from -180 to 180.
        theta (float): Elevation angle at which to get the gain from, in degrees, from 0 (zenith)
            to 180 (nadir).
        phi_3dB (float): Horizontal 3dB bandwidth of single element, in degrees.
        theta_3dB (float): Vertical 3dB bandwidth of single element, in degrees.
        Am (float): Front-to-back ratio, in dB.
        SLAv (float): Vertical sidelobe attenuation, in dB.
        GEmax (float): Maximum gain of single element, in dBi.
        NH (int): Number of columns in the array of elements.
        NV (int): Number of rows in the array of elements.
        dH_over_wl (float): Horizontal elements spacing over wavelength (dH/ʎ).
        dV_over_wl (float): Vertical elements spacing over wavelength (dV/ʎ).
        phi_i_escan (float): Bearing (h angle) of formed beam, in degrees.
        theta_i_etilt (float): Tilt (v angle) of formed beam, in degrees (positive value for
            downtilt, negative for uptilt).

    Returns:
        float: Composite antenna gain, in dBi.
    """
    cos_theta = cos(radians(theta))
    sin_theta = sin(radians(theta))
    sin_phi = sin(radians(phi))
    sin_theta_i = sin(radians(theta_i_etilt)) 
    cos_theta_i = cos(radians(theta_i_etilt))
    sin_phi_i =  sin(radians(phi_i_escan))
    
    j2pi = complex(0,1)*2*pi
    v1 = dV_over_wl*cos_theta
    v2 = dH_over_wl*sin_theta*sin_phi
    w0 = 1/sqrt(NH*NV)
    w1 = dV_over_wl*sin_theta_i
    w2 = dH_over_wl*cos_theta_i*sin_phi_i

    w_v_sum = 0
    for m in range(0, NH):
        for n in range(0, NV):
            v = cmath.exp(j2pi*( n*v1 + m*v2 ))
            w = w0*cmath.exp(j2pi*( n*w1 + m*w2 ))
            w_v_sum += v*w

    A_E = IMTAntennaElementGain(phi, theta, phi_3dB, theta_3dB, Am, SLAv, GEmax)

    w_v_sum = abs(w_v_sum)
    if w_v_sum == 0:
        A_A_Beami = -300
    else:
        A_A_Beami = A_E + 10*log10(w_v_sum**2)

    return A_A_Beami