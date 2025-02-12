"""Implementation of ITU-R P.1144-12 (partial).
"""

import os
from typing import Union
import numpy as np
import numpy.typing as npt
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['SquareGridBilinearInterpolation',
           'BicubicInterpolation',
           'CompressITUDigitalMapFile',
           'CompressAllITUDigitalMapFiles',
           'LoadCompressedITUDigitalMapFile',
           'LoadITUDigitalMapFile']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SquareGridBilinearInterpolation(grid: npt.ArrayLike, numRows: int, rowSize: int,
                                    r: float, c: float) -> float:
    """
    Bi-linear interpolation as documented in ITU-R P.1144-11, Annex 1, section 1b.

    Args:
        grid (numpy.typing.ArrayLike): The square grid to interpolate from. Assumed to be
            a one-dimensional array of numRows*rowSize elements.
        numRows (int): Number of rows in grid.
        rowSize (int): Number of elements in a row of grid.
        r (float): Fractional row number (zero-based index).
        c (float): Fractional column number (zero-based index).

    Returns:
        (float): Bi-linear interpolated value from grid.
    """
    R = int(r)
    R = max(R, 0)
    R = min(R, numRows-2)
    C = int(c)
    C = max(C, 0)
    C = min(C, rowSize-2)
    I00 = grid[ R    * rowSize +  C   ]
    I10 = grid[(R+1) * rowSize +  C   ]
    I01 = grid[ R    * rowSize + (C+1)]
    I11 = grid[(R+1) * rowSize + (C+1)]
    irc = I00*((R+1-r)*(C+1-c)) + I10*((r-R)*(C+1-c)) + I01*((R+1-r)*(c-C)) + I11*((r-R)*(c-C))
    return irc


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def BicubicInterpolation(grid: npt.ArrayLike, numRows: int, rowSize: int,
                         r: float, c: float) -> float:
    """
    Bi-cubic interpolation as documented in ITU-R P.1144-11, Annex 1, section 2.

    Args:
        grid (numpy.typing.ArrayLike): The square grid to interpolate from. Assumed to be
            a one-dimensional array of numRows*rowSize elements.
        numRows (int): Number of rows in grid.
        rowSize (int): Number of elements in a row of grid.
        r (float): Fractional row number (zero-based index).
        c (float): Fractional column number (zero-based index).

    Returns:
        (float): Bi-cubic interpolated value from grid.
    """
    R = int(r)-1
    R = max(R, 0)
    R = min(R, numRows-4)
    C = int(c)-1
    C = max(C, 0)
    C = min(C, rowSize-4)
    RI = [0.0]*4
    I = 0
    for ri_index, X in enumerate(range(R, R+4, 1)):
        for j in range(C, C+4, 1):
            RI[ri_index] += grid[(X*rowSize)+j]*_K(c-j)
    for ri_index, i in enumerate(range(R, R+4, 1)):
        I += RI[ri_index]*_K(r-i)
    return I


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _K(d: float) -> float:
    a = -0.5
    abs_d = abs(d)
    if abs_d >= 2:
        return 0
    elif abs_d >= 1:
        return a*abs_d*abs_d*abs_d - 5*a*abs_d*abs_d + 8*a*abs_d - 4*a
    else:
        return (a+2)*abs_d*abs_d*abs_d - (a+3)*abs_d*abs_d + 1


_scriptDir = os.path.dirname(os.path.abspath(__file__))


def CompressITUDigitalMapFile(filename: str, deleteOriginal: bool=False) -> None:
    """
    filename may be the full path (i.e. absolute path) or it may be relative to this file's
    directory.
    """
    if os.path.isabs(filename) == False:
        pathname = os.path.join(_scriptDir, filename)
    else:
        pathname = filename
    mapArray = np.loadtxt(fname=pathname).flatten()
    newPathname, _ = os.path.splitext(pathname)
    np.savez_compressed(newPathname, mapArray)
    if deleteOriginal == True:
        os.remove(pathname)


def CompressAllITUDigitalMapFiles(directory: str, fileExt: str='.txt', deleteOriginal: bool=False) -> None:
    """
    directory may be a full path (i.e. absolute path) or it may be relative to this file's
    directory. Goes through all sub-directories.
    """
    if os.path.isabs(directory) == False:
        fullPathDir = os.path.join(_scriptDir, directory)
    else:
        fullPathDir = directory
    for (dirpath, subdirs, filenames) in os.walk(fullPathDir):
        for filename in filenames:
            if filename.endswith(fileExt.lower()) or filename.endswith(fileExt.upper()):
                pathname = os.path.join(dirpath, filename)
                CompressITUDigitalMapFile(pathname, deleteOriginal)


def LoadCompressedITUDigitalMapFile(filename: str) -> npt.ArrayLike:
    """
    filename is assumed to be relative to this file's directory
    """
    pathname = os.path.join(_scriptDir, filename)
    return np.load(pathname)['arr_0']


def LoadITUDigitalMapFile(filename: str, delimiter: Union[str, None]=None) -> npt.ArrayLike:
    """
    filename is assumed to be relative to this file's directory
    """
    pathname = os.path.join(_scriptDir, filename)
    return np.loadtxt(fname=pathname, delimiter=delimiter).flatten()