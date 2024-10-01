
"""Implementation of ITU-R P.1511-3 (partial).
"""


from . import itur_p1144
from numba import jit


__all__ = ['TopographicHeightAMSL']


@jit(nopython=True)
def TopographicHeightAMSL(lat: float, lon: float) -> float:
    """
    ITU-R P.1511-3, Annex 1, Section 1.1
    Gets the topographic height of the surface of the Earth above mean sea level (m).

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The topographic height of the surface of the Earth above mean sea level (m).
    """
    # lat from +90.125 to -90.125, lon from -180.125 to +180.125
    latInterval = 1/12
    lonInterval = 1/12
    numRows = 2164 # (180.250/(1/12))+1
    rowSize = 4324 # (360.250/(1/12))+1

    r = (90.125 - lat) / latInterval
    c = (180.125 + lon) / lonInterval
    return itur_p1144.BicubicInterpolation(_TOPO, numRows, rowSize, r, c)


# Data originally from ITU file TOPO.dat within 'R-REC-P.1511-3-202408-I!!ZIP-E.zip'
_TOPO = itur_p1144._LoadITUDigitalMapFile('data/itu_proprietary/p1511/TOPO.dat')