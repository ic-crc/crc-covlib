# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Wrapper around NTIA's C++ implementation of ITU-R P.528-5.
See https://github.com/NTIA/p528
"""

import ctypes as ct
import sys
import os
import enum
from math import radians, asin, cos, pi


__all__ = ['Polarization', # enum
           'PropagationMode', # enum
           'ReturnCode', # enum
           'WarningFlag', # int enum
           'Results', # data class
           'FreeSpaceElevAngleToGreatCircleDistance',
           'BasicTransmissionLoss',
           'BasicTransmissionLossEx']


def _load_p528lib():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if sys.maxsize > 2**32:
        bin_dir = 'bin_64bit'
    else:
        bin_dir = 'bin_32bit'
        raise ImportError('\nNo 32-bit version of the P.528 library is currently available, ' \
            'you may want to use a 64-bit python interpreter instead.\n')
    if sys.platform.startswith('win32'):
        p452lib_pathname = os.path.join(cur_dir, bin_dir, 'iturp528.dll')
    else:
        p452lib_pathname = os.path.join(cur_dir, bin_dir, 'iturp528.so')
    try:
        p452lib = ct.CDLL(p452lib_pathname)
        #print("iturp528 library successfully loaded ", p452lib)
        _set_args_and_return_ctypes(p452lib)
        return p452lib
    except Exception as e:
        print(e)
        return None


class Polarization(enum.Enum):
    HORIZONTAL = 0
    VERTICAL   = 1


class PropagationMode(enum.Enum):
    NOT_SET       = 0
    LINE_OF_SIGHT = 1
    DIFFRATION    = 2
    TROPOSCATTER  = 3


class ReturnCode(enum.Enum):
    SUCCESS                        = 0
    ERROR_VALIDATION__D_KM         = 1
    ERROR_VALIDATION__H_1          = 2
    ERROR_VALIDATION__H_2          = 3
    ERROR_VALIDATION__TERM_GEO     = 4
    ERROR_VALIDATION__F_MHZ_LOW    = 5
    ERROR_VALIDATION__F_MHZ_HIGH   = 6
    ERROR_VALIDATION__PERCENT_LOW  = 7
    ERROR_VALIDATION__PERCENT_HIGH = 8
    ERROR_VALIDATION__POLARIZATION = 9
    ERROR_HEIGHT_AND_DISTANCE      = 10
    SUCCESS_WITH_WARNINGS          = 11


class WarningFlag(enum.IntEnum):
    WARNING__NO_WARNINGS        = 0x00
    WARNING__DFRAC_TROPO_REGION = 0x01
    WARNING__HEIGHT_LIMIT_H_1   = 0x02
    WARNING__HEIGHT_LIMIT_H_2   = 0x04


class Results:
    """
    Results from the BasicTransmissionLossEx() function.

    Attributes:
        A_dB (float): Basic transmission loss (dB).
        retCode (crc_covlib.helper.itur_p528.ReturnCode): Return code.
        warnings (int): Warning flags.
        d_km (float): Great circle path distance (km). Could be slightly different than specified
            in input variable if within LOS region.
        Afs_dB (float): Free space basic transmission loss (dB).
        Aa_dB (float): Median atmospheric absorption loss (dB).
        thetah1_rad (float): Elevation angle of the ray at the low terminal (rad).
        propagMode (crc_covlib.helper.itur_p528.PropagationMode): Mode of propagation.
    """
    def __init__(self):
        self.A_dB:float = 0
        self.retCode:ReturnCode = ReturnCode.SUCCESS
        self.warnings:int = 0
        self.d_km:float = 0
        self.Afs_dB:float = 0
        self.Aa_dB:float = 0
        self.thetah1_rad:float = 0
        self.propagMode:PropagationMode = PropagationMode.NOT_SET


def FreeSpaceElevAngleToGreatCircleDistance(fsElevAngle_deg: float, hr1_m: float, hr2_m: float
                                           ) -> float:
    """
    ITU-R P.528-5, Annex 2, Section 1
    Computes the great-circle distance between two terminals, in km.

    Args:
        fsElevAngle_deg (float): Free space elevation angle of the low terminal to the high
            terminal (deg).
        hr1_m (float): Height of the low terminal above mean sea level (m).
        hr2_m (float): Height of the high terminal above mean sea level (m).

    Returns:
        (float): Great-circle distance between the terminals (km).
    """
    theta_elev_rad = radians(fsElevAngle_deg)
    hr1_km = hr1_m/1000.0
    hr2_km = hr2_m/1000.0
    phi_rad = asin(((6371.0+hr1_km)/(6371.0+hr2_km))*cos(theta_elev_rad)) # Earth central angle
    theta_ca_rad = (pi/2.0) - theta_elev_rad - phi_rad
    d_km = 6371.0*theta_ca_rad
    return d_km


def BasicTransmissionLoss(f_MHz: float, p: float, d_km: float, hr1_m: float, hr2_m: float,
                          Tpol: Polarization) -> float:
    """
    ITU-R P.528-5, Annex 2, Section 3
    Computes the basic transmission loss, in dB.

    Args:
        f_MHz (float): Frequency (MHz), with 100 <= f_MHz <= 30000.
        p (float): Time percentage (%), with 1 <= p <= 99.
        d_km (float): Great-circle path distance between terminals (km), with 0 <= d_km.
        hr1_m (float): Height of the low terminal above mean sea level (m),
            with 1.5 <= hr1_m <= 20000.
        hr2_m (float): Height of the high terminal above mean sea level (km),
            with 1.5 <= hr2_m <= 20000, and with hr1_m <= hr2_m.
        Tpol (crc_covlib.helper.itur_p528.Polarization): Parameter indicating either horizontal or
            vertical linear polarization.

    Returns:
        A (float): Basic transmission loss (dB). Returns 0 on error.
    """
    propagation_mode = ct.c_int()
    warnings = ct.c_int()
    d__km = ct.c_double()
    A__db = ct.c_double()
    A_fs__db = ct.c_double()
    A_a__db = ct.c_double()
    theta_h1__rad = ct.c_double()

    _p452_cdll.P528(d_km, hr1_m, hr2_m, f_MHz, Tpol.value, p, ct.byref(propagation_mode),
                    ct.byref(warnings), ct.byref(d__km), ct.byref(A__db), ct.byref(A_fs__db),
                    ct.byref(A_a__db), ct.byref(theta_h1__rad))
    return A__db.value


def BasicTransmissionLossEx(f_MHz: float, p: float, d_km: float, hr1_m: float, hr2_m: float,
                            Tpol: Polarization) -> Results:
    """
    ITU-R P.528-5, Annex 2, Section 3
    Computes the basic transmission loss (extended results version), in dB.

    Args:
        f_MHz (float): Frequency (MHz), with 100 <= f_MHz <= 30000.
        p (float): Time percentage (%), with 1 <= p <= 99.
        d_km (float): Great-circle path distance between terminals (km), with 0 <= d_km.
        hr1_m (float): Height of the low terminal above mean sea level (m),
            with 1.5 <= hr1_m <= 20000.
        hr2_m (float): Height of the high terminal above mean sea level (km),
            with 1.5 <= hr2_m <= 20000, and with hr1_m <= hr2_m.
        Tpol (crc_covlib.helper.itur_p528.Polarization): Parameter indicating either horizontal or
            vertical linear polarization.

    Returns:
        (crc_covlib.helper.itur_p528.Results): Object of type Results containing the basic
            transmission loss value (dB) and other intermediate results. See the Results class
            definition for more details.
    """
    propagation_mode = ct.c_int()
    warnings = ct.c_int()
    d__km = ct.c_double()
    A__db = ct.c_double()
    A_fs__db = ct.c_double()
    A_a__db = ct.c_double()
    theta_h1__rad = ct.c_double()

    ret = _p452_cdll.P528(d_km, hr1_m, hr2_m, f_MHz, Tpol.value, p, ct.byref(propagation_mode),
                          ct.byref(warnings), ct.byref(d__km), ct.byref(A__db), ct.byref(A_fs__db),
                          ct.byref(A_a__db), ct.byref(theta_h1__rad))
    results = Results()
    results.A_dB = A__db.value
    results.retCode = ReturnCode(ret)
    results.warnings = warnings.value
    results.d_km = d__km.value
    results.Afs_dB = A_fs__db.value
    results.Aa_dB = A_a__db.value
    results.thetah1_rad = theta_h1__rad.value
    results.propagMode = PropagationMode(propagation_mode.value)

    return results


def _set_args_and_return_ctypes(lib):
    lib.P528.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int, ct.c_double,
                         ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_double),
                         ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                         ct.POINTER(ct.c_double)]
    lib.P528.restype = ct.c_int


_p452_cdll = _load_p528lib()
