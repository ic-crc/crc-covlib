# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of ITU-R P.681-11 (partial).
"""

from math import log as ln, exp, sqrt, sin, radians, cos, tan, atan, degrees, erf
from typing import Callable, Union
import inspect
from enum import Enum
import warnings
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['ShadowingLevel',
           'RoadsideTreesShadowingFade',
           'RoadsideTreesShadowingFadeEx',
           'NonGsoRoadsideTreesShadowingUnavail',
           'FadeDurationDistribution',
           'NonFadeDurationDistribution',
           'BuildingBlockageProbability',
           'StreetCanyonMaskingFunction',
           'SingleWallMaskingFunction',
           'StreetCrossingMaskingFunction',
           'TJunctionMaskingFunction',
           'MountainousMultipathFadingDistribution',
           'RoadsideTreesMultipathFadingDistribution',
           'ShadowingCrossCorrelationCoefficient',
           'AvailabilityImprobability',
           'FIGURE_1',
           'FIGURE_2',
           'FIGURE_4',
           'FIGURE_6a',
           'FIGURE_6b',
           'FIGURE_6c',
           'FIGURE_6d',
           'FIGURE_9',
           'FIGURE_10',
           'FIGURE_25']


class ShadowingLevel(Enum):
    MODERATE = 1
    EXTREME  = 2


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def RoadsideTreesShadowingFade(f_GHz: float, theta_deg: float, p: float) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.1.

    Estimates fade du to roadside tree-shadowing (dB).

    "The predicted fade distributions apply for highways and rural roads where the overall aspect
    of the propagation path is, for the most part, orthogonal to the lines of roadside trees and
    utility poles and it is assumed that the dominant cause of LMSS signal fading is tree canopy
    shadowing".

    Args:
        f_GHz (float): Frequency (GHz), with 0.8 <= f_GHz <= 20.
        theta_deg (float): Path elevation angle to the satellite (degrees),
            with 7 <= theta_deg <= 60.
        p (float): Percentage of distance travelled over which fade is exceeded (%), with
            1 <= p <= 80.

    Returns:
        A (float): Fade exceeded for the specified percentage of distance travelled (dB).
    """
    # Step 4
    theta_deg = max(theta_deg, 20)

    # Step 1
    def _AL_func(p: float, theta_deg: float) -> float:
        """Valid for f_GHz = 1.5, 1 <= p <= 20 and 20 <= theta_deg <= 60."""
        M = 3.44 + 0.0975*theta_deg - 0.002*theta_deg*theta_deg
        N = -0.443*theta_deg + 34.76
        AL = -M*ln(p) + N
        return AL

    # Step 2
    def _A20_func(p: float, theta_deg: float, f_GHz: float) -> float:
        """Valid for 0.8 <= f_GHz <= 20 and 1 <= p <= 20."""
        AL = _AL_func(p, theta_deg)
        A20 = AL*exp(1.5*((1/sqrt(1.5))-(1/sqrt(f_GHz))))
        return A20

    # Step 3
    if p > 20:
        A = _A20_func(20, theta_deg, f_GHz)*(1/ln(4))*ln(80/p)
    else:
        A = _A20_func(p, theta_deg, f_GHz)

    return A


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def RoadsideTreesShadowingFadeEx(f_GHz: float, theta_deg: float, p: int) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.1.1.

    Extension of the roadside trees model for elevation angles greater than 60 degrees at
    frequencies of 1.6 GHz and 2.6 GHz.

    Args:
        f_GHz (float): Frequency (GHz), with f_GHz = 1.6 or f_GHz = 2.6.
        theta_deg (float): Path elevation angle to the satellite (degrees),
            with 60 <= theta_deg <= 90.
        p (int): Percentage of distance travelled over which fade is exceeded (%). p must be set to
            one of: 1, 5, 10, 15, 20 or 30.

    Returns:
        A (float): Fade exceeded for the specified percentage of distance travelled (dB).
    """
    nominal_freqs_ghz = [1.6, 2.6]
    if f_GHz not in nominal_freqs_ghz:
        raise ValueError('f_GHz must be set to one of: 1.6, 2.6.')

    if theta_deg < 60 or theta_deg > 90:
        raise ValueError('theta_deg must be between 60 and 90.')

    nominal_percents = [1, 5, 10, 15, 20, 30]
    if p not in nominal_percents:
        raise ValueError('p_percent must be set to one of: 1, 5, 10, 15, 20, 30.')

    A60 = RoadsideTreesShadowingFade(f_GHz, 60, p)

    if f_GHz == 1.6:
        nominal_fades = {1: 4.1, 5: 2.0, 10: 1.5, 15: 1.4, 20: 1.3, 30: 1.2}
    else:
        nominal_fades = {1: 9.0, 5: 5.2, 10: 3.8, 15: 3.2, 20: 2.8, 30: 2.5}

    A80 = nominal_fades[p]
    if theta_deg <= 80:
        A = (A60*(80-theta_deg) + A80*(theta_deg-60))/(80-60)
    else:
        A = (A80*(90-theta_deg))/(90-80)

    return A


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def NonGsoRoadsideTreesShadowingUnavail(f_GHz: float, theta_list_deg: list[float],
                                        p_time_list: list[float], Gm_list_dBi: list[float],
                                        A_dB: float) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.1.2.

    Estimates the percentage of unavailability due to roadside trees shadowing for non-
    geostationary (non-GSO) and mobile-satellite systems.

    Args:
        f_GHz (float): Frequency (GHz), with 0.8 <= f_GHz <= 20.
        theta_list_deg (list[float]): List of path elevation angles under which the terminal will
            see the satellite (degrees), with 7 <= theta_list_deg[i] <= 60 for any i.
        p_time_list (list[float]): For each elevation angle in theta_list_deg, the percentage
            of time (%) for which the terminal will see the satellite at that angle, with
            0 <= p_time_list[i] <= 100 for any i.
        Gm_list_dBi (list[float]): For each elevation angle in theta_list_deg, the mobile terminal's
            antenna gain (dBi) at the corresponding elevation angle.
        A_dB (float): Fade margin (dB).

    Returns:
        p_unavail (float): The total system unavailability (%), from 0 to 100.
    """
    total_unavailability_percent = 0
    for theta_deg, time_percent, gm_dbi in zip(theta_list_deg, p_time_list, Gm_list_dBi):
        dist_travelled_percent = _RoadsideTreesShadowingFadeInv(f_GHz=f_GHz, theta_deg=theta_deg,
                                                                A_dB=A_dB+gm_dbi)
        total_unavailability_percent += time_percent * dist_travelled_percent / 100.0
    return total_unavailability_percent


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _RoadsideTreesShadowingFadeInv(f_GHz: float, theta_deg: float, A_dB: float) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.1.2.

    See roadside tree-shadowing from Section 4.1.1.

    Args:
        f_GHz (float): Frequency (GHz), with 0.8 <= f_GHz <= 20.
        theta_deg (float): Path elevation angle to the satellite (degrees),
            with 7 <= theta_deg <= 60.
        A_dB (float): Fade exceeded for the returned percentage of distance travelled (dB).

    Returns:
        p (float): Percentage of distance travelled over which the specified fade is exceeded (%).
            Returned value will be between 1 and 80.
    """
    min_p = 1
    max_p = 80
    mid_p = (max_p + min_p)/2.0
    while True:
        temp_a_db = RoadsideTreesShadowingFade(f_GHz=f_GHz, theta_deg=theta_deg, p=mid_p)
        if temp_a_db < A_dB:
            max_p = mid_p
        else:
            min_p = mid_p
        temp_mid_p = (max_p + min_p)/2.0
        if abs(mid_p-temp_mid_p) < 1E-4: # arbitrary value
            break
        mid_p = temp_mid_p
    if abs(A_dB-temp_a_db) > 0.01: # arbitrary value, accept max diff of 0.01 dB
        raise ValueError('Cannot calculate result. A_dB and/or other input parameters may be '\
                         'outside the validity range of the model.')
    return mid_p


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def FadeDurationDistribution(dd_m: float) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.2.

    Estimates the probability that the distance fade duration exceeds the distance dd (m), under
    the condition that the attenuation exceeds 5 dB.

    The model "is based on measurements at an elevation angle of 51 degrees and is applicable for
    [roads that exhibit] moderate to severe shadowing (percentage of optical shadowing between 55%
    and 90%)".

    Args:
        dd_m (float): Distance fade duration (meters), with 0.02 <= dd_m.

    Returns:
        p (float): Probability (%) that the distance fade duration exceeds the distance dd_m, under
            the condition that the attenuation exceeds 5 dB. Returned value will be between 0 and
            100.
    """
    alpha = 0.22
    sigma = 1.215
    P = 0.5*(1-erf((ln(dd_m)-ln(alpha))/(sqrt(2)*sigma)))
    return P*100


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def NonFadeDurationDistribution(dd_m: float, shadowingLevel: ShadowingLevel) -> float:
    """
    ITU-R P.681-5, Annex 1, Section 4.1.3.

    Estimates the probability that a continuous non-fade distance duration exceeds the distance dd
    (m), given that the fade is smaller than a 5 dB threshold.

    The model "is based on measurements at an elevation angle of 51 degrees and is applicable for
    [roads that exhibit] moderate to severe shadowing (percentage of optical shadowing between 55%
    and 90%)".

    Args:
        dd_m (float): Distance fade duration (meters).
        shadowingLevel (crc_covlib.helper.itur_p681.ShadowingLevel): One of MODERATE (percentage of
            optical shadowing between 55% and 75%) or EXTREME (percentage of optical shadowing
            between 75% and 90%).

    Returns:
        p (float): Probability that a continuous non-fade distance duration exceeds the distance dd
            (m), given that the fade is smaller than a 5 dB threshold. Returned value will be
            between 0 and 100.
    """
    if shadowingLevel == ShadowingLevel.MODERATE:
        beta = 20.54
        gamma = 0.58
    elif shadowingLevel == ShadowingLevel.EXTREME:
        beta = 11.71
        gamma = 0.8371
    else:
        raise ValueError('Unsupported value for parameter shadowingLevel (shadowing level).')
    p = beta*pow(dd_m, -gamma)
    p = min(p, 100) # NOTE: not in the rec., avoid exceeding 100%
    return p # already a percentage, do not multipy by 100


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def BuildingBlockageProbability(f_GHz: float, Cf: float, theta_deg: float, hm_m: float,
                                dm_m: float, hb_m: float, phi_deg: float) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 4.2.

    Estimates the percentage probability of blockage due to the buildings (%).

    Args:
        f_GHz (float): Frequency (GHz), with f_GHz from about 0.8 to 20.
        Cf (float): Required clearance as a fraction of the first Fresnel zone.
        theta_deg (float): Elevation angle of the ray to the satellite above horizontal (degrees),
            with 0 < theta_deg < 90.
        hm_m (float): Height of mobile above ground (meters).
        dm_m (float): Distance of the mobile from the front of the buildings (meters).
        hb_m (float): The most common (modal) building heigh (meters).
        phi_deg (float): Azimuth angle of the ray relative to street direction (degrees),
            with 0 < phi_deg < 180.

    Returns:
        p (float): Percentage probability of blockage due to the buildings (%).
    """
    sin_phi = sin(radians(phi_deg))
    cos_theta = cos(radians(theta_deg))
    dr_m = dm_m/(sin_phi/cos_theta)
    lambda_m = 1.0E-9*2.998E8/f_GHz
    h2_m = Cf*sqrt(lambda_m*dr_m)
    tan_theta = tan(radians(theta_deg))
    h1_m = hm_m + (dm_m*tan_theta/sin_phi)
    p = 100*exp(-(h1_m-h2_m)**2/(2*hb_m*hb_m))
    return p


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SA(h_m: float, w_m: float, phi_deg: float) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Equations (11a), (11b). See FIGURES 5 & 6.
    
    Args:
        h_m (float): Average building height (meters).
        w_m (float): Street width (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with 0 <= phi_deg <= 90.

    Returns:
        theta (float): Elevation angle (degrees) at which the frontier between link (non-shaded
            areas) and no-link (shaded areas) occurs.
    """
    if phi_deg == 90:
        theta_rad = atan(h_m/(w_m/2))
    elif phi_deg == 0:
        theta_rad = 0
    else:
        tan_phi = tan(radians(phi_deg))
        theta_rad = atan(h_m/sqrt((w_m*w_m/4)*((1/(tan_phi*tan_phi))+1)))
    theta_deg = degrees(theta_rad)
    return theta_deg


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SB(h_m: float, w1_m: float, w2_m: float, phi_deg: float) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Equations (11c), (11d), (11e). See FIGURES 5 & 6.
    
    Args:
        h_m (float): Average building height (meters).
        w1_m (float): Width of first street (meters).
        w2_m (float): Width of second street (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with 0 <= phi_deg <= 90.

    Returns:
        theta (float): Elevation angle (degrees) at which the frontier between link (non-shaded
            areas) and no-link (shaded areas) occurs.
    """
    if radians(phi_deg) > atan(w1_m/w2_m):
        phi_deg = 90-phi_deg
    return _SA(h_m, w1_m, phi_deg)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def StreetCanyonMaskingFunction(theta_deg: float, h_m: float, w_m: float, phi_deg: float) -> bool:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Estimates whether a link can be completed for the street canyon scenario.

    Args:
        theta_deg (float): Path elevation angle, with 0 <= theta_deg <= 90.
        h_m (float): Average building height (meters).
        w_m (float): Street width (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with -180 <= phi_deg <= 180.
    
    Returns:
        (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).
    """
    # have phi_deg in the -180 to 180 range
    while phi_deg > 180:
        phi_deg -= 360
    while phi_deg < -180:
        phi_deg += 360

    phi_deg = abs(phi_deg)
    if phi_deg > 90:
        phi_deg = 180 - phi_deg
    if theta_deg > _SA(h_m, w_m, phi_deg):
        return True
    return False


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SingleWallMaskingFunction(theta_deg: float, h_m: float, w_m: float, phi_deg: float) -> bool:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Estimates whether a link can be completed for the single wall scenario.

    Args:
        theta_deg (float): Path elevation angle, with 0 <= theta_deg <= 90.
        h_m (float): Average building height (meters).
        w_m (float): Street width (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with -180 <= phi_deg <= 180.
    
    Returns:
        (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).
    """
    # have phi_deg in the -180 to 180 range
    while phi_deg > 180:
        phi_deg -= 360
    while phi_deg < -180:
        phi_deg += 360

    if phi_deg >= 0:
        return True

    phi_deg = abs(phi_deg)
    if phi_deg > 90:
        phi_deg = 180 - phi_deg
    if theta_deg > _SA(h_m, w_m, phi_deg):
        return True
    return False


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def StreetCrossingMaskingFunction(theta_deg: float, h_m: float, w1_m: float, w2_m: float,
                                  phi_deg: float) -> bool:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Estimates whether a link can be completed for the street crossing scenario.

    Args:
        theta_deg (float): Path elevation angle, with 0 <= theta_deg <= 90.
        h_m (float): Average building height (meters).
        w1_m (float): Width of first street (meters).
        w2_m (float): Width of second street (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with -180 <= phi_deg <= 180.
    
    Returns:
        (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).
    """
    # have phi_deg in the -180 to 180 range
    while phi_deg > 180:
        phi_deg -= 360
    while phi_deg < -180:
        phi_deg += 360

    phi_deg = abs(phi_deg)
    if phi_deg > 90:
        phi_deg = 180 - phi_deg
    if theta_deg > _SB(h_m, w1_m, w2_m, phi_deg):
        return True
    return False


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def TJunctionMaskingFunction(theta_deg: float, h_m: float, w1_m: float, w2_m: float, phi_deg: float
                             ) -> bool:
    """
    ITU-R P.681-11, Annex 1, Section 4.4.

    Estimates whether a link can be completed for the T-junction scenario.

    Args:
        theta_deg (float): Path elevation angle, with 0 <= theta_deg <= 90.
        h_m (float): Average building height (meters).
        w1_m (float): Width of first street (meters).
        w2_m (float): Width of second street (meters).
        phi_deg (float): Street orientation with respect to the link (degrees),
            with -180 <= phi_deg <= 180.
    
    Returns:
        (bool): True when a link can be completed (non-shaded), False otherwise (shaded areas).
    """
    # have phi_deg in the -180 to 180 range
    while phi_deg > 180:
        phi_deg -= 360
    while phi_deg < -180:
        phi_deg += 360

    if phi_deg >= 0:
        if phi_deg > 90:
            phi_deg = 180 - phi_deg
        if theta_deg > _SB(h_m, w1_m, w2_m, phi_deg):
            return True
        return False
    else:
        phi_deg = abs(phi_deg)
        if phi_deg > 90:
            phi_deg = 180 - phi_deg
        if theta_deg > _SA(h_m, w1_m, phi_deg):
            return True
        return False


def MountainousMultipathFadingDistribution(f_GHz: float, theta_deg: float, A_dB: float,
                                           warningsOn: bool=True) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 5.1.

    Distribution of fade depths due to multipath in mountainous terrain. The model is valid when
    the effect of shadowing is negligible.

    Supported input values (from the recommendation's TABLE 3):

    | f_GHz  | theta_deg | A_dB  |
    |--------|-----------|-------|
    | 0.87   | 30        | 2-7   |
    | 1.5    | 30        | 2-8   |
    | 0.87   | 45        | 2-4   |
    | 1.5    | 45        | 2-5   |

    Args:
        f_GHz (float): Frequency (GHz), with f_GHz = 0.87 or f_GHz = 1.5.
        theta_deg (float): Path elevation angle to the satellite (degrees),
            with theta_deg = 30 or theta_deg = 45.
        A_dB (float): Fade exceeded (dB). See table above for ranges of valid values.
        warningsOn (bool): Indicates whether to raise a warning when A_dB is out of validity range.

    Returns:
        p (float): Percentage of distance over which the fade is exceeded (%), from 1 to 10.
    """
    if f_GHz == 0.87 and theta_deg == 30:
        a = 34.52
        b = 1.855
        if warningsOn and (A_dB < 2 or A_dB > 7):
            warnings.warn("a_db is outside its validity range of 2-7.", UserWarning)
    elif f_GHz == 1.5 and theta_deg == 30:
        a = 33.19
        b = 1.710
        if warningsOn and (A_dB < 2 or A_dB > 8):
            warnings.warn("a_db is outside its validity range of 2-8.", UserWarning)
    elif f_GHz == 0.87 and theta_deg == 45:
        a = 31.64
        b = 2.464
        if warningsOn and (A_dB < 2 or A_dB > 4):
            warnings.warn("a_db is outside its validity range of 2-4.", UserWarning)
    elif f_GHz == 1.5 and theta_deg == 45:
        a = 39.95
        b = 2.321
        if warningsOn and (A_dB < 2 or A_dB > 5):
            warnings.warn("a_db is outside its validity range of 2-5.", UserWarning)
    else:
        if not(f_GHz == 0.87 or f_GHz == 1.5):
            raise ValueError('f_GHz must be set to one of: 0.87, 1.5.')
        if not(theta_deg == 30 or theta_deg == 45):
            raise ValueError('theta_deg must be set to one of: 30, 45.')
    p = a*pow(A_dB, -b)
    return p


def RoadsideTreesMultipathFadingDistribution(f_GHz: float, A_dB: float, warningsOn: bool=True
                                             ) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 5.2.

    Distribution of fade depths due to multipath in a roadsite trees environment. The model assumes
    negligible shadowing.

    "Experiments conducted along tree-lined roads in the United States of America have shown that
    multipath fading is relatively insensitive to path elevation over the range of 30° to 60°".

    Supported input values (from the recommendation's TABLE 4):

    | f_GHz  | A_dB  |
    |--------|-------|
    | 0.87   | 1-4.5 |
    | 1.5    | 1-6   |

    Args:
        f_GHz (float): Frequency (GHz), with f_GHz = 0.87 or f_GHz = 1.5.
        A_dB (float): Fade exceeded (dB). See table above for ranges of valid values.
        warningsOn (bool): Indicates whether to raise a warning when A_dB is out of validity range.

    Returns:
        p (float): Percentage of distance over which the fade is exceeded (%), from 1 to 50.
    """
    if f_GHz == 0.87:
        u = 125.6
        v = 1.116
        if warningsOn and (A_dB < 1 or A_dB > 4.5):
            warnings.warn("A_dB is outside its validity range of 1-4.5.", UserWarning)
    elif f_GHz == 1.5:
        u = 127.7
        v = 0.8573
        if warningsOn and (A_dB < 1 or A_dB > 6):
            warnings.warn("A_dB is outside its validity range of 1-6.", UserWarning)
    else:
        raise ValueError('f_GHz must be set to one of: 0.87, 1.5.')
    p = u*exp(-v*A_dB)
    return p


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _ThreeSegmentModelPoints(theta1_deg: float, theta2_deg: float, h_m: float, w_m: float,
                             l_m: float
                             ) -> tuple[float, float, Union[float, None], Union[float, None]]:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1.

    Args:
        theta1_deg (float): Satellite 1 elevation angle (degrees), with 0 < theta1_deg < 90.
        theta2_deg (float): Satellite 2 elevation angle (degrees), with 0 < theta2_deg < 90 and
            with theta2_deg >= theta1_deg.
        h_m (float): Average building height (meters).
        w_m (float): Average street width (meters).
        l_m (float): Length of street under consideration (meters). A large value is advised for
            this parameter, i.e. l_m >= 200.

    Returns:
        rhoA (float): Cross-correlation coefficient at point A (no units), from -1 to +1.
        rhoD (float): Cross-correlation coefficient at point D (no units), from -1 to +1.
        delta_phiB (float|None): Azimuth spacing at point B (degrees), from 0 to 90 degrees.
        delta_phiC (float|None): Azimuth spacing at point C (degrees), from 0 to 90 degrees.

    """
    # Step 1
    tan_theta_1 = tan(radians(theta1_deg))
    tan_theta_2 = tan(radians(theta2_deg))
    x1_squared = ((h_m*h_m)/(tan_theta_1*tan_theta_1)) - (w_m*w_m/4)
    # NOTE: Using theta_2 instead of theta_1 for calculating x2 (likely an editing error in
    #       P.681-11).
    x2_squared = ((h_m*h_m)/(tan_theta_2*tan_theta_2)) - (w_m*w_m/4)

    if x1_squared < 0 and x2_squared < 0:
        # Step 6
        return (1, 1, None, None)

    if x1_squared >= 0:
        x1 = sqrt(x1_squared)
        x1 = min(x1, l_m/2)
        xi1_deg = round(degrees(atan((w_m/2)/x1)))
        M1 = (xi1_deg+0.5)/90

    if x2_squared >= 0:
        x2 = sqrt(x2_squared)
        x2 = min(x2, l_m/2)
        xi2_deg = round(degrees(atan((w_m/2)/x2)))
        M2 = (xi2_deg+0.5)/90

    if x1_squared >= 0 and x2_squared < 0:
        # Step 6
        N11 = 4*xi1_deg + 2
        rho = (N11/180)-1
        return (rho, rho, None, None)

    if x1_squared < 0 and x2_squared >= 0: # should not happend unless theta2_deg < theta1_deg
        if theta2_deg < theta1_deg:
            raise ValueError('theta2_deg must be greater than or equal to theta1_deg.')
        else:
            raise RuntimeError('Cannot compute three-segment model points.')

    # Steps 2 & 3
    rhoA = _rhoA(xi1_deg=xi1_deg, xi2_deg=xi2_deg, M1=M1, M2=M2)
    rhoD = _rhoD(xi1_deg=xi1_deg, xi2_deg=xi2_deg, M1=M1, M2=M2)

    # Step 4
    delta_phiB = xi2_deg - xi1_deg

    # Step 5
    if (xi1_deg+xi2_deg) <= 90:
        # NOTE: using plus (+) instead of minus (-) (likely an editing error in P.681-11)
        delta_phiC = xi1_deg + xi2_deg
    else:
        delta_phiC = 180 - xi1_deg - xi2_deg

    return (rhoA, rhoD, delta_phiB, delta_phiC)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _rhoA(xi1_deg, xi2_deg, M1, M2) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1, eq. (39), (41), (42a), (42b).
    """
    N11 = 4*xi1_deg + 2
    N00 = 360 - 4*xi2_deg - 2
    N01 = 4*(xi2_deg-xi1_deg)
    N10 = 0
    return _rho(xi1_deg=xi1_deg, xi2_deg=xi2_deg, M1=M1, M2=M2, N11=N11, N00=N00, N01=N01, N10=N10)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _rhoD(xi1_deg, xi2_deg, M1, M2) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1, eq. (40a), (40b), (41), (42a), (42b).
    """
    if (xi1_deg+xi2_deg) <= 90:
        N11 = 0
        N00 = 360 - 4*xi1_deg - 4*xi2_deg - 4
        N01 = 4*xi2_deg + 2
        N10 = 4*xi1_deg + 2
    else:
        N11 = 4*xi1_deg + 4*xi2_deg + 4 - 360
        N00 = 0
        N01 = 360 - 4*xi1_deg - 2
        N10 = 360 - 4*xi2_deg - 2
    return _rho(xi1_deg=xi1_deg, xi2_deg=xi2_deg, M1=M1, M2=M2, N11=N11, N00=N00, N01=N01, N10=N10)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _rho(xi1_deg, xi2_deg, M1, M2, N11, N00, N01, N10) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1, eq. (41), (42a), (42b).
    """
    num = N11*(1-M1)*(1-M2) + N00*(0-M1)*(0-M2) + N10*(1-M1)*(0-M2) + N01*(0-M1)*(1-M2)
    sigma_of_theta1 = sqrt(_sigma_squared(xi1_deg, M1))
    sigma_of_theta2 = sqrt(_sigma_squared(xi2_deg, M2))
    rho = num/(359*sigma_of_theta1*sigma_of_theta2)
    return rho


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _sigma_squared(xi_deg, M) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1, eq. (42a), (42b).
    """
    return ((4*xi_deg+2)*(1-M)*(1-M) + (360-4*xi_deg-2)*(0-M)*(0-M))/359


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def ShadowingCrossCorrelationCoefficient(delta_phi_deg: float, theta1_deg: float,
                                         theta2_deg: float, h_m: float, w_m: float, l_m: float
                                         ) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1.

    Quantifies the cross-correlation coefficient between shadowing events in urban areas (using a
    "street canyon" area geometry).

    Args:
        delta_phi_deg (float): Azimuth spacing between two separate satellite-to-mobile links in
            street canyons (degrees), with -180 <= delta_phi_deg <= 180.
        theta1_deg (float): Satellite 1 elevation angle (degrees), with 0 < theta1_deg < 90.
        theta2_deg (float): Satellite 2 elevation angle (degrees), with 0 < theta2_deg < 90 and
            with theta2_deg >= theta1_deg.
        h_m (float): Average building height (meters).
        w_m (float): Average street width (meters).
        l_m (float): Length of street under consideration (meters). A large value is advised for
            this parameter, i.e. l_m >= 200.

    Returns:
        rho (float): The cross-correlation coefficient (no units).
    """
    rhoA, rhoD, delta_phiB, delta_phiC = _ThreeSegmentModelPoints(theta1_deg, theta2_deg, h_m, w_m,
                                                                  l_m)

    # have delta_phi_deg in the -180 to 180 range
    while delta_phi_deg > 180:
        delta_phi_deg -= 360
    while delta_phi_deg < -180:
        delta_phi_deg += 360

    # apply symmetry, as "correlation values are symmetric for all four Δφ quadrants"
    delta_phi_deg = abs(delta_phi_deg)
    if delta_phi_deg > 90:
        delta_phi_deg = 180 - delta_phi_deg

    # See FIGURE 25 regarding following code

    if rhoA == rhoD:
        return rhoA

    if delta_phi_deg <= delta_phiB:
        return rhoA
    if delta_phi_deg >= delta_phiC:
        return rhoD

    # linear interpolation
    rho = (rhoA*(delta_phiC-delta_phi_deg) + rhoD*(delta_phi_deg-delta_phiB))/ \
          (delta_phiC-delta_phiB)

    return rho


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def AvailabilityImprobability(rho: float, p1: float, p2: float) -> float:
    """
    ITU-R P.681-11, Annex 1, Section 9.2.1.

    Computes the overall availability improbability after satellite diversity.

    Args:
        rho (float): Cross-correlation coefficient. May be obtained from the
            shadowing_cross_correlation_coefficient() function.
        p1 (float): Unavailability probability for the first link (%), with 0 <= p1 <= 100.
        p2 (float): Unavailability probability for the second link (%), with 0 <= p2 <= 100. For
            urban areas, p1 and p2 values may be computed using the building_blockage_probability()
            function.

    Returns:
        p0 (float): The overall availability improbability after satellite diversity (%). The
            probability of availability will be 100-p0.
    """
    p1 = p1/100
    p2 = p2/100
    p0 = rho*sqrt(p1*(1-p1))*sqrt(p1*(1-p2)) + p1*p2
    if p0 < 0:
        raise ValueError('Input values are yielding a negative probability value.')
    return p0*100


def FIGURE_1() -> None:
    import matplotlib.pyplot as plt
    f_GHz = 1.5
    percents = [1,2,5,10,20,30,50]
    fades = [[],[],[],[],[],[],[]]
    thetas = [*range(10, 60+1, 1)]
    for index, p in enumerate(percents):
        for theta_deg in thetas:
            fade = RoadsideTreesShadowingFade(f_GHz, theta_deg, p)
            fades[index].append(fade)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6.75)
    ax1.set_xlim([10, 60])
    ax1.set_xticks([*range(10, 60+1, 5)])
    ax1.set_ylim([0, 30])
    ax1.set_yticks([*range(0, 30+1, 2)])
    for i, p in enumerate(percents):
        if i%2 == 0:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'
        ax1.plot(thetas, fades[i], color='#000000', linestyle=linestyle)
        pt_index = int(len(thetas)*0.16)
        plt.text(thetas[pt_index], fades[i][pt_index], f'{p}%')
    ax1.set_title('FIGURE 1\nFading at 1.5 GHz due to roadside shadowing versus path elevation '\
                  'angle')
    ax1.set_xlabel('Path elevation angle (degrees)')
    ax1.set_ylabel('Fade exceeded (dB)')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_2() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    dists = [*np.arange(0.01, 0.1, 0.01), *np.arange(0.1, 1, 0.1), *np.arange(1, 10+1, 1)]
    probs = []
    for dd_m in dists:
        p = FadeDurationDistribution(dd_m)
        probs.append(p)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6.75)
    ax1.set_xlim([0.02, 10])
    ax1.set_xscale('log')
    ax1.set_xticks(dists)
    ax1.set_ylim([1, 100])
    ax1.set_yscale('log')
    ax1.set_yticks([*range(1, 10, 1), *range(10, 100+1, 10)])
    ax1.plot(dists, probs, color='#000000', linestyle='solid')
    ax1.set_title('FIGURE 2\nFBest fit cumulative fade distribution for roadside tree\n'\
                  'shadowing with a 5 dB threshold')
    ax1.set_xlabel('Fade duration (m)')
    ax1.set_ylabel('Percentage of fade duration > abscissa')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_4() -> None:
    import matplotlib.pyplot as plt
    f_GHz = 1.6
    hb = 15
    hm = 1.5
    dm = 17.5
    phis = [45, 90]
    Cfs = [0, 0.7]
    probabilities = [[],[],[],[]]
    thetas = [*range(0, 80+1, 1)]
    index = 0
    for phi in phis:
        for Cf in Cfs:
            for theta in thetas:
                p = BuildingBlockageProbability(f_GHz, Cf, theta, hm, dm, hb, phi)
                probabilities[index].append(p)
            index +=1
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6.75)
    ax1.set_xlim([0, 80])
    ax1.set_xticks([*range(0, 80+1, 10)])
    ax1.set_ylim([0, 100])
    ax1.set_yticks([*range(0, 100+1, 20)])
    ax1.plot(thetas, probabilities[0], color='#000000', linestyle='solid', label='phi=45, Cf=0')
    ax1.plot(thetas, probabilities[1], color='#000000', linestyle='dashed', label='phi=45, Cf=0.7')
    ax1.plot(thetas, probabilities[2], color='#0000AA', linestyle='solid', label='phi=90, Cf=0')
    ax1.plot(thetas, probabilities[3], color='#0000AA', linestyle='dashed', label='phi=90, Cf=0.7')
    ax1.set_title('FIGURE 4\nExamples of roadside building shadowing\n'\
                  '(f=1.6GHz, hb=15m, hm=1.5m, dm=17.5m')
    ax1.set_xlabel('Path elevation angle (degrees)')
    ax1.legend()
    ax1.set_ylabel('Fade exceeded (dB)')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_6a() -> None:
    _FIGURE_6('MKF of a street canyon', StreetCanyonMaskingFunction)


def FIGURE_6b() -> None:
    _FIGURE_6('MKF of a single wall', SingleWallMaskingFunction)


def FIGURE_6c() -> None:
    _FIGURE_6('MKF of a street crossing', StreetCrossingMaskingFunction)


def FIGURE_6d() -> None:
    _FIGURE_6('MKF of a T-junction', TJunctionMaskingFunction)


def _FIGURE_6(subtitle: str, masking_function: Callable) -> None:
    import matplotlib.pyplot as plt

    def masking_function_calls(phi_min, phi_max, mask_func):
        h_m = 20
        w1_m = 20
        w2_m = 20
        phi_list = []
        theta_list = []
        arg_count = len(inspect.signature(masking_function).parameters)
        for phi in range(phi_min, phi_max+1, 1):
            for theta in range(0, 90+1, 1):
                if arg_count == 4:
                    link_avail = mask_func(theta, h_m, w1_m, phi)
                else:
                    link_avail = mask_func(theta, h_m, w1_m, w2_m, phi)
                if link_avail is False:
                    phi_list.append(phi)
                    theta_list.append(theta)
        return (phi_list, theta_list)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(8, 6.75)

    ax1.set_xlim([0, 180])
    ax1.set_xticks([*range(0, 180+1, 20)])
    ax1.set_ylim([0, 90])
    ax1.set_yticks([*range(0, 80+1, 20)])
    ax1.set_title(f'FIGURE 6\n{subtitle}')
    ax1.set_ylabel('Elevation angle (degrees)')
    ax1.grid(True, 'both','both')

    phis, thetas = masking_function_calls(0, 180, masking_function)
    ax1.scatter(phis, thetas, s=10, c='#000000')

    ax2.set_xlim([0, -180])
    ax2.set_xticks([*range(0, -180-1, -20)])
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_ylim([90, 0])
    ax2.set_yticks([*range(80, 0-1, -20)])
    ax2.set_xlabel('Azimuth (degrees)')
    ax2.set_ylabel('Elevation angle (degrees)')
    ax2.grid(True, 'both','both')

    phis, thetas = masking_function_calls(-180, 0, masking_function)
    ax2.scatter(phis, thetas, s=10, c='#000000')

    plt.show()


def FIGURE_9() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [0.87, 1.5]
    thetas = [45, 30]
    fade_depths = [*np.arange(2, 8+0.01, 0.1)]
    probs = [[],[],[],[]]
    i=0
    for theta_deg in thetas:
        for f_ghz in freqs:
            for fade_depth in fade_depths:
                p = MountainousMultipathFadingDistribution(f_ghz, theta_deg, fade_depth, False)
                probs[i].append(p)
            i += 1
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6.75, 8)
    ax1.set_xlim([0, 10])
    ax1.set_xticks([*range(0, 10+1, 1)])
    ax1.set_ylim([1, 20])
    ax1.set_yscale('log')
    ax1.set_yticks([*range(1, 10+1, 1), 20])
    ax1.set_yticklabels([*range(1, 10+1, 1), 20])
    ax1.plot(fade_depths, probs[0], color='#000000', linestyle='solid', label='A: 870 MHz, 45°')
    ax1.plot(fade_depths, probs[1], color='#FF0000', linestyle='solid', label='B: 1.5 GHz, 45°')
    ax1.plot(fade_depths, probs[2], color='#00FF00', linestyle='solid', label='C: 870 MHz, 30°')
    ax1.plot(fade_depths, probs[3], color='#0000FF', linestyle='solid', label='D: 1.5 GHz, 30°')
    ax1.set_title('FIGURE 9\nBest fit cumulative fade distributions for multipath fading in\n'\
                  'mountainous terrain')
    ax1.set_xlabel('Fade depth (dB)')
    ax1.set_ylabel('Percentage of distance fade > abscissa')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_10() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [0.87, 1.5]
    fade_depths = [*np.arange(1, 6+0.01, 0.1)]
    probs = [[],[]]
    i=0
    for f_ghz in freqs:
        for fade_depth in fade_depths:
            p = RoadsideTreesMultipathFadingDistribution(f_ghz, fade_depth, False)
            probs[i].append(p)
        i += 1
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6.75, 8)
    ax1.set_xlim([0, 10])
    ax1.set_xticks([*range(0, 10+1, 1)])
    ax1.set_ylim([1, 100])
    ax1.set_yscale('log')
    ax1.set_yticks([*range(1, 10, 1), *range(10, 100+1, 10)])
    ax1.set_yticklabels([*range(1, 10, 1), *range(10, 100+1, 10)])
    ax1.plot(fade_depths, probs[0], color='#000000', linestyle='solid', label='A: 870 MHz')
    ax1.plot(fade_depths, probs[1], color='#FF0000', linestyle='solid', label='B: 1.5 GHz')
    ax1.set_title('FIGURE 10\nBest fit cumulative fade distributions for multipath fading on\n'\
                  'tree-lined roads')
    ax1.set_xlabel('Fade depth (dB)')
    ax1.set_ylabel('Percentage of distance fade > abscissa')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_25() -> None:
    import matplotlib.pyplot as plt
    from dataclasses import dataclass

    h_m = 20
    w_m = 20
    l_m = 200

    delta_phi_min = 0
    delta_phi_max = 90
    delta_phi_list = [*range(delta_phi_min, delta_phi_max+1, 1)]

    MKA = degrees(atan(h_m/(w_m/2))) # eq.(9)

    @dataclass
    class _TestCase:
        index: int
        name: str
        color: str
        theta1_deg: float
        theta2_deg: float
        rho_list: list[float]
    case_list = []

    # Special case 1: both satellites are above the MKA for any azimuth spacing
    case_1 = _TestCase(index=0,
                       name='Special case 1',
                       color='#FF0000',
                       theta1_deg = MKA + (90-MKA)*0.1,
                       theta2_deg = MKA + (90-MKA)*0.2,
                       rho_list=[])
    case_list.append(case_1)

    # Special case 2: one satellite is always above MKA and the other is always below
    case_2 = _TestCase(index=1,
                       name='Special case 2',
                       color='#00AA00',
                       theta1_deg = MKA*0.33,
                       theta2_deg = MKA + (90-MKA)*0.5,
                       rho_list=[])
    case_list.append(case_2)

    # Special case 3: the two satellites are at the same elevation
    case_3 = _TestCase(index=2,
                       name='Special case 3',
                       color='#0000FF',
                       theta1_deg = MKA*0.375,
                       theta2_deg = MKA*0.375,
                       rho_list=[])
    case_list.append(case_3)

    # Special case 4: satellites with very different elevations
    case_4 = _TestCase(index=3,
                       name='Special case 4',
                       color='#FF7700',
                       theta1_deg = MKA * 0.001,
                       theta2_deg = MKA * 0.999,
                       rho_list=[])
    case_list.append(case_4)

    # General model
    case_5 = _TestCase(index=4,
                       name='General model',
                       color='#000000',
                       theta1_deg = MKA * 0.35,
                       theta2_deg = MKA * 0.65,
                       rho_list=[])
    case_list.append(case_5)

    # Compute rho values
    for case_i in case_list:
        for delta_phi_deg in delta_phi_list:
            rho = ShadowingCrossCorrelationCoefficient(delta_phi_deg, case_i.theta1_deg,
                                                       case_i.theta2_deg, h_m, w_m, l_m)
            case_i.rho_list.append(rho)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6.75)
    ax1.set_xlim([delta_phi_min, delta_phi_max])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    for case_i in case_list:
        ax1.plot(delta_phi_list, case_i.rho_list, case_i.color, linestyle='solid',
                 label=case_i.name)
    ax1.set_title('FIGURE 25\nThree segment cross correlation coefficient model')
    ax1.set_xlabel('Azimuth spacing (degrees)')
    ax1.legend()
    ax1.set_ylabel('Correlation coefficient')
    plt.grid(True, 'both','both')
    plt.show()
