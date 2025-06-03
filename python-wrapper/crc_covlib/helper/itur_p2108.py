# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of ITU-R P.2108-1.
"""

from math import log10, sqrt, pi, log, tan, degrees, atan
from typing import Union
import enum
from . import itur_p1057
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['ClutterType', # enum
           'GetDefaultRepresentativeHeight',
           'HeightGainModelClutterLoss',
           'TerrestrialPathClutterLoss',
           'EarthSpaceClutterLoss',
           'FIGURE_1',
           'FIGURE_2']


class ClutterType(enum.Enum):
    WATER_SEA          = 1
    OPEN_RURAL         = 2
    SUBURBAN           = 3
    URBAN_TREES_FOREST = 4
    DENSE_URBAN        = 5


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def GetDefaultRepresentativeHeight(clut: ClutterType) -> float:
    """
    ITU-R P.2108-1, Annex 1, Section 3.1
    Gets the default representative clutter height, as defined in the recommendation's Table 3.

    Args:
        clut (crc_covlib.helper.itur_p2108.ClutterType): One of the clutter types from the
            recommendation's Table 3.

    Returns:
        (float): Representative clutter height (m).
    """
    if clut == ClutterType.URBAN_TREES_FOREST:
        return 15.0
    elif clut == ClutterType.DENSE_URBAN:
        return 20.0
    else:
        return 10.0


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def HeightGainModelClutterLoss(f_GHz: float, h_m: float, clut: ClutterType,
                               R_m: Union[float, None]=None, ws_m: float=27.0) -> float:
    """
    ITU-R P.2108-1, Annex 1, Section 3.1
    "An additional loss, Ah, is calculated which can be added to the basic transmission loss of a
    path calculated above the clutter, therefore basic transmission loss should be calculated
    to/from the height of the representative clutter height used. This model can be applied to both
    transmitting and receiving ends of the path."

    Args:
        f_GHz (float): Frequency (GHz), with 0.03 <= f_GHz <= 3.
        h_m (float): Antenna height (m), with 0 <= h_m.
        clut (crc_covlib.helper.itur_p2108.ClutterType): One of the clutter types from the 
            recommendation's Table 3.
        R_m (float|None): Representative clutter height (m), with 0 < R_m. When R_m is set to None,
            the default representative clutter height for the specified clutter type is used.
        ws_m (float): Street width (m).
    
    Returns:
        Ah (float): Clutter loss (dB).
    """
    if R_m is None:
        R_m = GetDefaultRepresentativeHeight(clut)
    if h_m >= R_m:
        Ah = 0.0
    else:
        if clut==ClutterType.SUBURBAN or clut==ClutterType.URBAN_TREES_FOREST or \
           clut==ClutterType.DENSE_URBAN:
            Knu = 0.342*sqrt(f_GHz)
            hdif = R_m - h_m
            theta_clut_deg = degrees(atan(hdif/ws_m))
            v = Knu*sqrt(hdif*theta_clut_deg)
            if v <= -0.78:
                Jv = 0.0
            else:
                Jv = 6.9 + 20.0*log10(sqrt((v-0.1)*(v-0.1)+1.0)+v-0.1)
            Ah = Jv - 6.03
        else:
            Kh2 = 21.8 + 6.2*log10(f_GHz)
            Ah = -Kh2*log10(h_m/R_m)
    return Ah


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _TerrestrialPathClutterLoss(f_GHz: float, d_km: float, loc_percent: float) -> float:
    """
    See TerrestrialPathClutterLoss().
    """
    d = d_km
    log10_f = log10(f_GHz)
    Ls = 32.98 + 23.9*log10(d) + 3.0*log10_f
    Ll = -2.0*log10(pow(10.0, -5.0*log10_f-12.5)+pow(10.0, -16.5))
    pow_Ll = pow(10.0, -0.2*Ll)
    pow_Ls = pow(10.0, -0.2*Ls)
    sigma_cb = sqrt((16.0*pow_Ll+36.0*pow_Ls) / (pow_Ll+pow_Ls))
    Lctt = -5.0*log10(pow_Ll+pow_Ls) - sigma_cb*itur_p1057.Qinv(loc_percent/100.0)
    return Lctt


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def TerrestrialPathClutterLoss(f_GHz: float, d_km: float, loc_percent: float) -> float:
    """
    ITU-R P.2108-1, Annex 1, Section 3.2
    Statistical clutter loss model for terrestrial paths that can be applied for urban and suburban
    clutter loss modelling provided terminal heights are well below the clutter height.

    Args:
        f_GHz (float): Frequency (GHz), with 0.5 <= f_GHz <= 67.
        d_km: Distance (km), with 0.25 <= d_km.
        loc_percent: Percentage of locations (%), with 0 < loc_percent < 100.
    
    Returns:
        (float): Clutter loss (dB).
    """
    Lctt = _TerrestrialPathClutterLoss(f_GHz, d_km, loc_percent)
    Lctt2km = _TerrestrialPathClutterLoss(f_GHz, 2.0, loc_percent)
    return min(Lctt, Lctt2km)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def EarthSpaceClutterLoss(f_GHz: float, elevAngle_deg: float, loc_percent: float) -> float:
    """
    ITU-R P.2108-1, Annex 1, Section 3.3
    Statistical distribution of clutter loss where one end of the interference path is within
    man-made clutter, and the other is a satellite, aeroplane, or other platform above the surface
    of the Earth. This model is applicable to urban and suburban environments.

    Args:
        f_GHz (float): Frequency (GHz), with 0.5 <= f_GHz <= 67.
        elevAngle_deg (float): Elevation angle (deg). The angle of the airborne platform or
            satellite as seen from the terminal, with 0 <= elevAngle_deg <= 90.
        loc_percent (float): Percentage of locations (%), with 0 < loc_percent < 100.
    
    Returns:
        (float): Clutter loss (dB).
    """
    th = elevAngle_deg
    A1 = 0.05
    K1 = 93*pow(f_GHz, 0.175)
    a = -K1*(log(1-(loc_percent/100)))
    b = 1/tan((A1*(1-(th/90)))+(pi*th/180))
    Lces = pow(a*b, 0.5*(90-th)/90)-1-(0.6*itur_p1057.Qinv(loc_percent/100))
    return Lces


def FIGURE_1() -> None:
    """
    Calculates values and display FIGURE 1 from ITU-R P.2108-1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    dists = [*np.arange(0.25, 2+0.01, 0.05), *range(2, 10+1, 1), *range(10, 100+1, 1)]
    freqs = [1,2,4,8,16,32,67]
    losses = [[],[],[],[],[],[],[]]
    for d_km in dists:
        for i, f_GHz in enumerate(freqs):
            losses[i].append(TerrestrialPathClutterLoss(f_GHz, d_km, 50))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xscale('log')
    ax1.set_xlim([0,100])
    ax1.set_xticks([0.1,1,10,100])
    ax1.set_ylim([15, 35])
    ax1.set_yticks([15,20,25,30,35])
    ax1.plot(dists, losses[0], color='#2020EC', label='1 GHz')
    ax1.plot(dists, losses[1], color='#FF8000', label='2 GHz')
    ax1.plot(dists, losses[2], color='#FFDE00', label='4 GHz')
    ax1.plot(dists, losses[3], color='#8000FF', label='8 GHz')
    ax1.plot(dists, losses[4], color='#427010', label='16 GHz')
    ax1.plot(dists, losses[5], color='#00FFFF', label='32 GHz')
    ax1.plot(dists, losses[6], color='#A00000', label='67 GHz')
    ax1.set_title('FIGURE 1\nMedian clutter loss for terrestrial paths')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Median clutter loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_2() -> None:
    """
    Calculates values and display FIGURE 2 from ITU-R P.2108-1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    thetas = [0,5,10,15,20,30,40,50,60,70,80,90]
    loc_percents = [0.1, *range(1, 100, 1), 99.9]
    losses = [[] for _ in range(len(thetas))]
    for i, theta in enumerate(thetas):
        for p in loc_percents:
            losses[i].append(EarthSpaceClutterLoss(30, theta, p))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([-10,70])
    ax1.set_xticks([*range(-10,70+1,10)])
    ax1.set_ylim([0, 100])
    ax1.set_yticks([*range(0,100+1,10)])
    for i, theta in enumerate(thetas):
        ax1.plot(losses[i], loc_percents, label='{}°'.format(theta))
    ax1.set_title('FIGURE 2\nCumulative distribution of clutter loss not exceeded for 30 GHz')
    ax1.set_xlabel('Clutter loss (dB)')
    ax1.set_ylabel('Percent of locations')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()
