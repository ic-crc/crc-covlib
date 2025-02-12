"""Implementation of ITU-R P.453-14 (partial).
"""

from . import itur_p1144
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['MedianAnnualNwet']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def MedianAnnualNwet(lat: float, lon: float) -> float:
    """
    ITU-R P.453-14
    Gets the median value of the wet term of the surface refractivity exceeded for the average
    year (ppm).

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The median value of the wet term of the surface refractivity exceeded for the
            average year (ppm).
    """
    # lat from -90 to +90, lon from -180 to +180 in NWET_Annual_50.TXT
    latInterval = 0.75
    lonInterval = 0.75
    numRows = 241 # number of points from [-90, 90] latitude deg range, at 0.75 deg intervals: (180/0.75)+1
    rowSize = 481 # number of points from [-180, 180] longitude deg range, at 0.75 deg intervals: (360/0.75)+1

    r = (90.0 + lat) / latInterval
    c = (180.0 + lon) / lonInterval
    return itur_p1144.SquareGridBilinearInterpolation(_NWET50, numRows, rowSize, r, c)


# Data originally from ITU file NWET_Annual_50.TXT within 'R-REC-P.453-14-201908-I!!ZIP-E.zip'
_NWET50 = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p453/NWET_Annual_50.TXT')