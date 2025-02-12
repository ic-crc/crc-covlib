"""Implementation of ITU-R P.839-4.
"""

from . import itur_p1144
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['MeanAnnualZeroCelsiusIsothermHeight',
           'MeanAnnualRainHeight']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def MeanAnnualZeroCelsiusIsothermHeight(lat: float, lon: float) -> float:
    """
    ITU-R P.839-4
    Gets the mean annual 0°C isotherm height (km above mean sea level).

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The mean annual 0°C isotherm height (km above mean sea level).
    """
    # lat from +90 to -90, lon from 0 to 360 in h0.txt
    latInterval = 1.5
    lonInterval = 1.5
    numRows = 121 # number of points from [-90, 90] latitude deg range, at 1.5 deg intervals: (180/1.5)+1
    rowSize = 241 # number of points from [0, 360] longitude deg range, at 1.5 deg intervals: (360/1.5)+1

    if lon < 0:
        lon += 360
    r = (90.0 - lat) / latInterval
    c = lon / lonInterval
    return itur_p1144.SquareGridBilinearInterpolation(_H0, numRows, rowSize, r, c)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def MeanAnnualRainHeight(lat: float, lon: float) -> float:
    """
    ITU-R P.839-4
    Gets the mean annual rain height (km above mean sea level).

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The mean annual rain height (km above mean sea level).
    """
    return MeanAnnualZeroCelsiusIsothermHeight(lat, lon) + 0.36


# Data originally from ITU file h0.txt within 'R-REC-P.839-4-201309-I!!ZIP-E.zip'
_H0 = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p839/h0.txt')
