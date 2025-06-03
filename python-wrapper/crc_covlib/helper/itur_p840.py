# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of ITU-R P.840-9

This module supports using both annual and monthly statistics. Statistics data files can be
obtained at https://www.itu.int/rec/R-REC-P.840/en or they can be installed by running the
install_ITU_data.py script (use the Custom mode for the option to install monthly statistics,
the default mode will only install annual statistics). Another option is to use functions from the
itur_data.py module.

When installing files manually (i.e. when not using the install_ITU_data.py script):
Annual statistics files must be placed in the helper/data/itu_proprietary/p840/annual/ directory.
Statistics for the month of Jan must be placed in the helper/data/itu_proprietary/p840/monthly/01/
directory, for the month of Feb they must be placed in the helper/data/p840/monthly/02/ directory,
etc.
"""

from . import itur_p1144
from . import itur_p1057
import numpy.typing as npt
from typing import Union
from math import log10, exp, sin, radians
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['CloudLiquidWaterAttenuationCoefficient',
           'InstantaneousCloudAttenuation',
           'StatisticalCloudAttenuation',
           'LogNormalApproxCloudAttenuation',
           'IntegratedCloudLiquidWaterContent',
           'IntegratedCloudLiquidWaterContentMean',
           'IntegratedCloudLiquidWaterContentStdDev']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def CloudLiquidWaterAttenuationCoefficient(f_GHz, T_K) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 2
    Gets the cloud liquid water specific attenuation coefficient (dB/km)/(g/m3).

    Args:
        f_GHz (float): Frequency (GHz), f_GHz <= 200.
        T_K (float): Liquid water temperature (K).
    
    Returns:
        Kl (float): The cloud liquid water specific attenuation coefficient (dB/km)/(g/m3).
    """
    a = (300/T_K)-1
    fp = 20.20 - 146*a + 316*a*a
    fs = 39.8*fp
    eps0 = 77.66 + 103.3*a
    eps1 = 0.0671*eps0
    eps2 = 3.52
    xp = 1 + (f_GHz*f_GHz)/(fp*fp)
    xs = 1 + (f_GHz*f_GHz)/(fs*fs)
    eps_pp = f_GHz*(eps0-eps1)/(fp*xp) + f_GHz*(eps1-eps2)/(fs*xs)
    eps_p = (eps0-eps1)/xp + (eps1-eps2)/xs + eps2
    n = (2+eps_p)/eps_pp
    Kl = 0.819*f_GHz/(eps_pp*(1+(n*n)))
    return Kl


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _KL(f_GHz) -> float:
    """
    Gets the cloud liquid mass absorption coefficient in dB/(kg/m2) or dB/mm.

    Args:
        f_GHz (float): Frequency (GHz).
    
    Returns:
        KL (float): The cloud liquid mass absorption coefficient in dB/(kg/m2) or dB/mm.
    """
    A1 = 0.1522
    A2 = 11.51
    A3 = -10.4912
    f1 = -23.9589
    f2 = 219.2096
    s1 = 3.2991E3
    s2 = 2.7595E6
    e1 = exp(-((f_GHz-f1)**2)/s1)
    e2 = exp(-((f_GHz-f2)**2)/s2)
    KL = CloudLiquidWaterAttenuationCoefficient(f_GHz, 273.75)* ((A1*e1) + (A2*e2) + A3)
    return KL


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def InstantaneousCloudAttenuation(f_GHz: float, theta_deg: float, L_kgm2: float) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 3.1
    Gets the predicted slant path instantaneous cloud attenuation (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
        theta_deg (float): Elevation angle (deg).
        L_kgm2 (float): Integrated cloud liquid water content, in kg/m2 or mm, from the surface
            of the Earth at the desired location.

    Returns:
        Ac (float): The predicted slant path instantaneous cloud attenuation (dB).
    """
    Ac = _KL(f_GHz)*L_kgm2/sin(radians(theta_deg))
    return Ac


def StatisticalCloudAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                                month: Union[int, None]=None) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 3.2
    Gets the predicted slant path statistical cloud attenuation (dB).

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
        theta_deg (float): Elevation angle (deg).
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual
            statistics and with 0.1 <= p <= 100 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual
            statistics.

    Returns:
        Ac (float): The predicted slant path statistical cloud attenuation (dB).
    """
    L = IntegratedCloudLiquidWaterContent(p, lat, lon, month)
    return InstantaneousCloudAttenuation(f_GHz, theta_deg, L)


def LogNormalApproxCloudAttenuation(f_GHz: float, theta_deg: float, p: float,
                                    lat: float, lon: float) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 3.3
    Gets the log-normal approximation to the predicted slant path statistical cloud attenuation (dB).
    Uses annual statistics.
    
    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
        theta_deg (float): Elevation angle (deg).
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual
            statistics and with 0.1 <= p <= 100 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.

    Returns:
        Ac (float): The log-normal approximation to the predicted slant path statistical cloud
            attenuation (dB).
    """
    PL = _GetDigitalMapValue(None, 'PL', lat, lon)
    if PL <= 0.02:
        return 0
    if p >= PL:
        return 0
    mL = _GetDigitalMapValue(None, 'mL', lat, lon)
    sL = _GetDigitalMapValue(None, 'sL', lat, lon)
    KL = _KL(f_GHz)
    Ac = KL*exp(mL+(sL*itur_p1057.Qinv(p/PL)))/sin(radians(theta_deg))
    return Ac


def IntegratedCloudLiquidWaterContent(p: float, lat: float, lon: float,
                                      month: Union[int, None]=None) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 4
    Gets the integrated cloud liquid water content from digital maps, in kg/m2, or, equivalently, mm.

    Args:
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 100 for annual
            statistics and with 0.1 <= p <= 100 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual
            statistics.

    Returns:
        L (float): The integrated cloud liquid water content from digital maps, in kg/m2, or,
            equivalently, mm.
    """
    if month is None:
        nominalPercentages = _ANNUAL_P
        filenames = _ANNUAL_FILENAMES
    else:
        nominalPercentages = _MONTHLY_P
        filenames = _MONTHLY_FILENAMES       
    for i in range(0, len(nominalPercentages)-1):
        if p >= nominalPercentages[i] and p <= nominalPercentages[i+1]:
            L0 = _GetDigitalMapValue(month, filenames[i], lat, lon)
            L1 = _GetDigitalMapValue(month, filenames[i+1], lat, lon)
            p0 = nominalPercentages[i] # p below
            p1 = nominalPercentages[i+1] # p above
            L = (L0*log10(p1/p) + L1*log10(p/p0)) / log10(p1/p0)
            return L
    return float('nan')


def IntegratedCloudLiquidWaterContentMean(lat: float, lon: float, month: Union[int, None]=None
                                          ) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 4.2.2
    Gets the mean of the integrated cloud liquid water content.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual
            statistics.

    Returns:
        (float): The mean of the integrated cloud liquid water content.
    """
    return _GetDigitalMapValue(month, 'L_mean', lat, lon)


def IntegratedCloudLiquidWaterContentStdDev(lat: float, lon: float, month: Union[int, None]=None
                                            ) -> float:
    """
    ITU-R P.840-9, Annex 1, Section 4.2.2
    Gets the standard deviation of the integrated cloud liquid water content.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). Use None for annual
            statistics.

    Returns:
        (float): The standard deviation of the integrated cloud liquid water content.
    """
    return _GetDigitalMapValue(month, 'L_std', lat, lon)


def _GetGrid(month: Union[int, None], filename: str) -> npt.ArrayLike:
    if (month, filename) not in _digital_maps:
        if month is None:
            pathname = 'data/itu_proprietary/p840/annual/{}.TXT'.format(filename)
        else:
            pathname = 'data/itu_proprietary/p840/monthly/{:02d}/{}.TXT'.format(month, filename)
        _digital_maps[(month, filename)] = itur_p1144.LoadITUDigitalMapFile(pathname)
    return _digital_maps[(month, filename)]


def _GetDigitalMapValue(month: Union[int, None], filename: str, lat: float, lon: float) -> float:
    # lat from -90 to +90, lon from -180 to +180 in L files
    latInterval = 0.25
    lonInterval = 0.25
    numRows = 721 # number of points from [-90, 90] latitude deg range, at 0.25 deg intervals: (180/0.25)+1
    rowSize = 1441 # number of points from [-180, 180] longitude deg range, at 0.75 deg intervals: (360/0.25)+1
    r = (90.0 + lat) / latInterval
    c = (180.0 + lon) / lonInterval
    grid = _GetGrid(month, filename)
    return itur_p1144.SquareGridBilinearInterpolation(grid, numRows, rowSize, r, c)


_ANNUAL_P = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 60, 70, 80,
             90, 95, 99, 100]

_ANNUAL_FILENAMES = ['L_001', 'L_002', 'L_003', 'L_005', 'L_01', 'L_02', 'L_03', 'L_05', 'L_1',
                     'L_2', 'L_3', 'L_5', 'L_10', 'L_20', 'L_30', 'L_50', 'L_60', 'L_70', 'L_80',
                     'L_90', 'L_95', 'L_99', 'L_100']

_MONTHLY_P = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 60, 70, 80, 90, 95, 99, 100]

_MONTHLY_FILENAMES = ['L_01', 'L_02', 'L_03', 'L_05', 'L_1', 'L_2', 'L_3', 'L_5', 'L_10', 'L_20',
                      'L_30', 'L_50', 'L_60', 'L_70', 'L_80', 'L_90', 'L_95', 'L_99', 'L_100']

_digital_maps = {}
