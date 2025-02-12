"""Implementation of ITU-R P.837-7 (partial).
"""

from . import itur_p1144
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['RainfallRate001']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def RainfallRate001(lat: float, lon: float) -> float:
    """
    ITU-R P.837-7
    Gets the annual rainfall rate exceeded for 0.01% of an average year (mm/hr).
    This function uses bilinear interpolation on the precalculated R001.txt file.

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The annual rainfall rate exceeded for 0.01% of an average year (mm/hr).
    """
    # lat from -90 to +90, lon from -180 to +180 in R001.txt
    latInterval = 0.125
    lonInterval = 0.125
    numRows = 1441 # number of points from [-90, 90] latitude deg range, at 0.125 deg intervals: (180/0.125)+1
    rowSize = 2881 # number of points from [-180, 180] longitude deg range, at 0.125 deg intervals: (360/0.125)+1

    r = (90.0 + lat) / latInterval
    c = (180.0 + lon) / lonInterval
    return itur_p1144.SquareGridBilinearInterpolation(_R001, numRows, rowSize, r, c)


# Data originally from ITU file R001.TXT within 'R-REC-P.837-7-201706-I!!ZIP-E.zip'
_R001 = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p837/R001.TXT')
