# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of Annex 1 from ITU-R P.835-6.
"""

import enum
from math import exp, sqrt
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['ReferenceAtmosphere',
           'Temperature',
           'Pressure',
           'WaterVapourDensity',
           'DryPressure',
           'WaterVapourPressure']


class ReferenceAtmosphere(enum.Enum):
    """
    Enumerates available reference atmospheres.
    """
    MEAN_ANNUAL_GLOBAL   = 1
    LOW_LATITUDE         = 2  # smaller than 22°
    MID_LATITUDE_SUMMER  = 3  # between 22° and 45°
    MID_LATITUDE_WINTER  = 4  # between 22° and 45°
    HIGH_LATITUDE_SUMMER = 5  # higher than 45°
    HIGH_LATITUDE_WINTER = 6  # higher than 45°


# MAGRA: Mean Annual Global Reference Atmosphere
MAGRA = ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Temperature(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float:
    """
    ITU-R P.835-6, Annex 1
    Gets the atmospheric temperature (K) at geometric height h_km (km).

    Args:
        h_km (float): Geometric height (km), with 0 <= h_km <= 100.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.

    Returns:
        (float): The atmospheric temperature (K) at geometric height h_km.
    """
    if h_km < 0.0:
        h_km = 0.0
    if h_km > 100.0:
        h_km = 100.0

    if refAtm == ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL:
            T = _MagraTemperature(h_km)
    elif refAtm == ReferenceAtmosphere.LOW_LATITUDE:
            T = _LowLatTemperature(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_SUMMER:
            T = _SummerMidLatTemperature(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_WINTER:
            T = _WinterMidLatTemperature(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_SUMMER:
            T = _SummerHighLatTemperature(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_WINTER:
            T= _WinterHighLatTemperature(h_km)

    return T


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Pressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float:
    """
    ITU-R P.835-6, Annex 1
    Gets the total atmospheric pressure (hPa) at geometric height h_km (km).

    Args:
        h_km (float): Geometric height (km), with 0 <= h_km <= 100.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.

    Returns:
        (float): The total atmospheric pressure (hPa) at geometric height h_km.
    """
    if h_km < 0.0:
        h_km = 0.0
    if h_km > 100.0:
        h_km = 100.0

    if refAtm == ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL:
            P = _MagraPressure(h_km)
    elif refAtm == ReferenceAtmosphere.LOW_LATITUDE:
            P = _LowLatPressure(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_SUMMER:
            P = _SummerMidLatPressure(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_WINTER:
            P = _WinterMidLatPressure(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_SUMMER:
            P = _SummerHighLatPressure(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_WINTER:
            P = _WinterHighLatPressure(h_km)

    return P


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def WaterVapourDensity(h_km: float, refAtm: ReferenceAtmosphere=MAGRA,
                       rho0_gm3: float=7.5, h0_km: float=2) -> float:
    """
    ITU-R P.835-6, Annex 1
    Gets the water-vapour density (g/m3) at geometric height h_km (km).

    Args:
        h_km (float): Geometric height (km), with 0 <= h_km <= 100.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).
        h0_km (float): Scale height (km). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL
            (MAGRA).

    Returns:
        (float): The water-vapour density (g/m3) at geometric height h_km.
    """
    if h_km < 0.0:
        h_km = 0.0
    if h_km > 100.0:
        h_km = 100.0

    if refAtm == ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL:
        rho = _MagraWaterVapourDensity(h_km, rho0_gm3, h0_km)

        # See Section 1.2 of Annex 1
        P = Pressure(h_km, refAtm)
        if rho/P < 2E-6:
            T = Temperature(h_km, refAtm)
            rho = 2E-6 * 216.7 * P / T
            # make sure rho is not higher than the specified ground-level rho
            # i.e. to always return 0 if rho0_gm3 is 0
            rho = min(rho, rho0_gm3) 
    elif refAtm == ReferenceAtmosphere.LOW_LATITUDE:
        rho = _LowLatWaterVapourDensity(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_SUMMER:
        rho = _SummerMidLatWaterVapourDensity(h_km)
    elif refAtm == ReferenceAtmosphere.MID_LATITUDE_WINTER:
        rho = _WinterMidLatWaterVapourDensity(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_SUMMER:
        rho = _SummerHighLatWaterVapourDensity(h_km)
    elif refAtm == ReferenceAtmosphere.HIGH_LATITUDE_WINTER:
        rho = _WinterHighLatWaterVapourDensity(h_km)

    return rho


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def DryPressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA) -> float:
    """
    ITU-R P.835-6, Annex 1
    Gets the dry atmospheric pressure (hPa) at geometric height h_km (km).

    Args:
        h_km (float): Geometric height (km), with 0 <= h_km <= 100.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.

    Returns:
        (float): The dry atmospheric pressure (hPa) at geometric height h_km.
    """
    rho = WaterVapourDensity(h_km, refAtm, 7.5, 2) # in g/m3
    T = Temperature(h_km, refAtm)
    P = Pressure(h_km, refAtm)
    e = rho*T/216.7 # water vapour pressure, in hPa
    Pdry = P - e # eq. (5) of P.453-14
    return Pdry


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def WaterVapourPressure(h_km: float, refAtm: ReferenceAtmosphere=MAGRA,
                        rho0_gm3: float=7.5, h0_km: float=2) -> float:
    """
    ITU-R P.835-6, Annex 1
    Gets the water vapour pressure (hPa) at geometric height h_km (km).

    Args:
        h_km (float): Geometric height (km), with 0 <= h_km <= 100.
        ref_atm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).
        h0_km (float): Scale height (km). Only applies when refAtm is set to MEAN_ANNUAL_GLOBAL
            (MAGRA).

    Returns:
        (float): The water vapour pressure (hPa) at geometric height h_km.
    """
    rho = WaterVapourDensity(h_km, refAtm, rho0_gm3, h0_km)
    T = Temperature(h_km, refAtm)
    e = rho*T/216.7
    return e


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _ToGeopotentialHeight(geometricHeight_km: float) -> float:
	return (6356.766*geometricHeight_km)/(6356.766+geometricHeight_km)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _MagraPressure(h: float) -> float:
    if h < 86:
        hp = _ToGeopotentialHeight(h)
        if hp <= 11:
            return 1013.25*pow(288.15/(288.15-6.5*hp), -34.1632/6.5)
        elif hp <= 20:
            return 226.3226*exp(-34.1632*(hp-11)/216.65)
        elif hp <= 32:
            return 54.74980*pow(216.65/(216.65+(hp-20)), 34.1632)
        elif hp <= 47:
            return 8.680422*pow(228.65/(228.65+2.8*(hp-32)), 34.1632/2.8)
        elif hp <= 51:
            return 1.109106*exp(-34.1632*(hp-47)/270.65)
        elif hp <= 71:
            return 0.6694167*pow(270.65/(270.65-2.8*(hp-51)), -34.1632/2.8)
        else:
            return 0.03956649*pow(214.65/(214.65-2.0*(hp-71)), -34.1632/2.0)
    else:
        return exp(95.571899-(4.011801*h)+(6.424731E-2*h*h)-(4.789660E-4*h*h*h)+(1.340543E-6*h*h*h*h))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _MagraTemperature(h: float) -> float:
    if h < 86:
        hp = _ToGeopotentialHeight(h)
        if hp <= 11:
            return 288.15-(6.5*hp)
        elif hp <= 20:
            return 216.65
        elif hp <= 32:
            return 216.65+(hp-20)
        elif hp <= 47:
            return 228.65+(2.8*(hp-32))
        elif hp <= 51:
            return 270.65
        elif hp <= 71:
            return 270.65-(2.8*(hp-51))
        else:
            return 214.65-(2.0*(hp-71))
    else:
        if h <= 91:
            return 186.8673
        else:
            x = (h-91)/19.9429
            return 263.1905-76.3232*sqrt(1-(x*x))
        

@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _MagraWaterVapourDensity(h: float, rho0: float=7.5, h0: float=2) -> float:
    return rho0*exp(-h/h0)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _LowLatTemperature(h: float) -> float:
    if h < 17:
        return 300.4222-(6.3533*h)+(0.005886*h*h)
    elif h < 47:
        return 194+((h-17)*2.533)
    elif h < 52:
        return 270
    elif h < 80:
        return 270-((h-52)*3.0714)
    else:
        return 184


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE) # Note: Numba caching to file does not support recursivity
def _LowLatPressure(h: float) -> float:
    if h <= 10:
        return 1012.0306-(109.0338*h)+(3.6316*h*h)
    elif h <= 72:
        #return _LowLatPressure(10)*exp(-0.147*(h-10))
        press_10 = 1012.0306-(109.0338*10)+(3.6316*10*10)
        return press_10*exp(-0.147*(h-10))
    else:
        #return _LowLatPressure(72)*exp(-0.165*(h-72))
        press_10 = 1012.0306-(109.0338*10)+(3.6316*10*10)
        press_72 = press_10*exp(-0.147*(72-10))
        return press_72*exp(-0.165*(h-72))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _LowLatWaterVapourDensity(h: float) -> float:
    if h <= 15:
        return 19.6542*exp(-(0.2313*h)-(0.1122*h*h)+(0.01351*h*h*h)-0.0005923*h*h*h*h)
    else:
        return 0


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SummerMidLatTemperature(h: float) -> float:
    if h < 13:
        return 294.9838-(5.2159*h)-(0.07109*h*h)
    elif h < 17:
        return 215.15
    elif h < 47:
        return 215.15*exp((h-17)*0.008128)
    elif h < 53:
        return 275
    elif h < 80:
        return 275+((1-exp((h-53)*0.06))*20)
    else:
        return 175


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE) # Note: Numba caching to file does not support recursivity
def _SummerMidLatPressure(h: float) -> float:
    if h <= 10:
        return 1012.8186-(111.5569*h)+(3.8646*h*h)
    elif h <= 72:
        #return _SummerMidLatPressure(10)*exp(-0.147*(h-10))
        press_10 = 1012.8186-(111.5569*10)+(3.8646*10*10)
        return press_10*exp(-0.147*(h-10))
    else:
        #return _SummerMidLatPressure(72)*exp(-0.165*(h-72))
        press_10 = 1012.8186-(111.5569*10)+(3.8646*10*10)
        press_72 = press_10*exp(-0.147*(72-10))
        return press_72*exp(-0.165*(h-72))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SummerMidLatWaterVapourDensity(h: float) -> float:
    if h <= 15:
        return 14.3542*exp(-(0.4174*h)-(0.02290*h*h)+(0.001007*h*h*h))
    else:
        return 0


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _WinterMidLatTemperature(h: float) -> float:
    if h < 10:
        return 272.7241-(3.6217*h)-(0.1759*h*h)
    elif h < 33:
        return 218
    elif h < 47:
        return 218+((h-33)*3.3571)
    elif h < 53:
        return 265
    elif h < 80:
        return 265-((h-53)*2.0370)
    else:
        return 210


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE) # Note: Numba caching to file does not support recursivity
def _WinterMidLatPressure(h: float) -> float:
    if h <= 10:
        return 1018.8627-(124.2954*h)+(4.8307*h*h)
    elif h <= 72:
        #return _WinterMidLatPressure(10)*exp(-0.147*(h-10))
        press_10 = 1018.8627-(124.2954*10)+(4.8307*10*10)
        return press_10*exp(-0.147*(h-10))
    else:
        #return _WinterMidLatPressure(72)*exp(-0.155*(h-72))
        press_10 = 1018.8627-(124.2954*10)+(4.8307*10*10)
        press_72 = press_10*exp(-0.147*(72-10))
        return press_72*exp(-0.155*(h-72))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _WinterMidLatWaterVapourDensity(h: float) -> float:
    if h <= 10:
        return 3.4742*exp(-(0.2697*h)-(0.03604*h*h)+(0.0004489*h*h*h))
    else:
        return 0


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SummerHighLatTemperature(h: float) -> float:
    if h < 10:
        return 286.8374-(4.7805*h)-(0.1402*h*h)
    elif h < 23:
        return 225
    elif h < 48:
        return 225*exp((h-23)*0.008317)
    elif h < 53:
        return 277
    elif h < 79:
        return 277-((h-53)*4.0769)
    else:
        return 171


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE) # Note: Numba caching to file does not support recursivity
def _SummerHighLatPressure(h: float) -> float:
    if h <= 10:
        return 1008.0278-(113.2494*h)+(3.9408*h*h)
    elif h <= 72:
        #return _SummerHighLatPressure(10)*exp(-0.140*(h-10))
        press_10 = 1008.0278-(113.2494*10)+(3.9408*10*10)
        return press_10*exp(-0.140*(h-10))
    else:
        #return _SummerHighLatPressure(72)*exp(-0.165*(h-72))
        press_10 = 1008.0278-(113.2494*10)+(3.9408*10*10)
        press_72 = press_10*exp(-0.140*(72-10))
        return press_72*exp(-0.165*(h-72))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SummerHighLatWaterVapourDensity(h: float) -> float:
    if h <= 15:
        return 8.988*exp(-(0.3614*h)-(0.005402*h*h)-(0.001955*h*h*h))
    else:
        return 0


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _WinterHighLatTemperature(h: float) -> float:
    if h < 8.5:
        return 257.4345+(2.3474*h)-(1.5479*h*h)+(0.08473*h*h*h)
    elif h < 30:
        return 217.5
    elif h < 50:
        return 217.5+((h-30)*2.125)
    elif h < 54:
        return 260
    else:
        return 260-((h-54)*1.667)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE) # Note: Numba caching to file does not support recursivity
def _WinterHighLatPressure(h: float) -> float:
    if h <= 10:
        return 1010.8828-(122.2411*h)+(4.554*h*h)
    elif h <= 72:
        #return _WinterHighLatPressure(10)*exp(-0.147*(h-10))
        press_10 = 1010.8828-(122.2411*10)+(4.554*10*10)
        return press_10*exp(-0.147*(h-10))
    else:
        #return _WinterHighLatPressure(72)*exp(-0.150*(h-72))
        press_10 = 1010.8828-(122.2411*10)+(4.554*10*10)
        press_72 = press_10*exp(-0.147*(72-10))
        return press_72*exp(-0.150*(h-72))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _WinterHighLatWaterVapourDensity(h: float) -> float:
    if h <= 10:
        return 1.2319*exp((0.07481*h)-(0.0981*h*h)+(0.00281*h*h*h))
    else:
        return 0
