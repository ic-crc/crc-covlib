# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Implementation of ITU-R P.530-18 (partial).
"""

from math import sqrt, exp, tanh, log10, log as ln, pow, radians, cos
import enum
from typing import Callable
from . import itur_p676
from . import itur_p837
from . import itur_p838
from . import itur_p1144
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['GeoclimaticFactorK',
           'DN75',
           'AtmosphericAttenuation',
           'FirstFresnelEllipsoidRadius',
           'ApproximatedDiffractionLoss',
           'TimePeriod', # enum
           'SingleFrequencyFadingDistribution',
           'FadingDistribution',
           'EnhancementDistribution',
           'InverseDistribution',
           'PathType', # enum
           'AvgWorstMonthToShorterWorstPeriod',
           'RainAttenuationLongTermStatistics',
           'FIGURE_3',
           'FIGURE_4']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def GeoclimaticFactorK(lat: float, lon: float) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 1.1
    Gets the geoclimatic factor K.

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        K (float): The geoclimatic factor K.
    """
    # lat from +90 to -90, lon from -180 to +180 in LogK.csv
    latInterval = 0.25
    lonInterval = 0.25
    numRows = 721 # number of points from [-90, 90] latitude deg range, at 0.25 deg intervals: (180/0.25)+1
    rowSize = 1441 # number of points from [-180, 180] longitude deg range, at 0.25 deg intervals: (360/0.25)+1
    r = (90.0 - lat) / latInterval
    c = (180.0 + lon) / lonInterval
    log10_K = itur_p1144.SquareGridBilinearInterpolation(_LOGK, numRows, rowSize, r, c)
    return pow(10, log10_K)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def DN75(lat: float, lon: float) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 1.1
    Gets parameter dN75, an empirical prediction of 0.1% of the average worst month refractivity
    increase with height over the lowest 75 m of the atmosphere from surface dewpoint data
    (N-units).

    Args:
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        dN75 (float): dN75, an empirical prediction of 0.1% of the average worst month refractivity
            increase with height over the lowest 75 m of the atmosphere from surface dewpoint data
            (N-units).
    """
    # lat from +90 to -90, lon from -180 to +180 in dN75.csv
    latInterval = 0.25
    lonInterval = 0.25
    numRows = 721 # number of points from [-90, 90] latitude deg range, at 0.25 deg intervals: (180/0.25)+1
    rowSize = 1441 # number of points from [-180, 180] longitude deg range, at 0.25 deg intervals: (360/0.25)+1
    r = (90.0 - lat) / latInterval
    c = (180.0 + lon) / lonInterval
    return itur_p1144.SquareGridBilinearInterpolation(_DN75, numRows, rowSize, r, c)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def AtmosphericAttenuation(d_km: float, f_GHz: float, P_hPa: float=1013.25,
                           T_K: float=288.15, rho_gm3: float=7.5) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 2.1
    Terrestrial path attenuation due to absorption by oxygen and water vapour (dB).

    Args:
        d_km (float): Path length (km), with 0 <= d_km.
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
        P_hPa (float): Atmospheric pressure (hPa).
        T_K (float): temperature (K).
        rho_gm3 (float): Water vapour density (g/m3).

    Returns:
        (float): Terrestrial path attenuation due to absorption by oxygen and water vapour (dB).
    """
    return itur_p676.TerrestrialPathGaseousAttenuation(d_km, f_GHz, P_hPa, T_K, rho_gm3)


def FirstFresnelEllipsoidRadius(d_km: float, f_GHz: float, d1_km: float, d2_km: float) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 2.2.1
    Radius of the first Fresnel ellipsoid (m).

    Args:
        d_km (float): Path length (km), with 0 < d_km.
        f_GHz (float): Frequency (GHz).
        d1_km (float): Distance from the first terminal to the path obstruction (km).
        d2_km (float): Distance from the second terminal to the path obstruction (km).

    Returns:
        F1 (float): Radius of the first Fresnel ellipsoid (m).
    """
    F1 = 17.3*sqrt((d1_km*d2_km)/(f_GHz*d_km))
    return F1


def ApproximatedDiffractionLoss(d_km: float, f_GHz: float, d1_km: float, d2_km: float, h_m: float
                                ) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 2.2.1
    Approximation of the diffraction loss (dB) over average terrain (strictly valid for losses
    greater than about 15 dB).

    Args:
        d_km (float): Path length (km), with 0 < d_km.
        f_GHz (float): Frequency (GHz).
        d1_km (float): Distance from the first terminal to the path obstruction (km).
        d2_km (float): Distance from the second terminal to the path obstruction (km).
        h_m (float): Height difference (m) between most significant path blockage and the path
            trajectory (h_m is negative if the top of the obstruction of interest is above the
            virtual line-of-sight).

    Returns:
        Ad (float): Approximation of the diffraction loss (dB) over average terrain (strictly valid
            for losses greater than about 15 dB).
    """
    F1 = FirstFresnelEllipsoidRadius(d_km, f_GHz, d1_km, d2_km)
    Ad = -20.0*h_m/F1 + 10
    return Ad


class TimePeriod(enum.Enum):
    """
    Enumerates possible time periods for fading/enhancement distributions.
    """
    AVG_WORST_MONTH = 1
    AVG_YEAR        = 2


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SingleFrequencyFadingDistribution(A_dB: float, d_km: float, f_GHz: float, he_masl: float,
                                      hr_masl: float, ht_masl: float, lat: float, lon: float,
                                      timePeriod: TimePeriod) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.1 and 2.3.4
    Calculates the percentage of time that fade depth A_dB is exceeded in the specified time
    period (%).
    
    This implements a method for predicting the single-frequency (or narrow-band) fading distribution
    at large fade depths in the average worst month or average year in any part of the world. This
    method does not make use of the path profile and can be used for initial planning, licensing,
    or design purposes.

    Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km,
    and can be set to zero for shorter paths.

    Args:
        A_dB (float): Fade depth (dB), with 0 <= A_dB.
        d_km (float): Path length (km), with 0 < d_km.
        f_GHz (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
        he_masl (float): Emitter antenna height (meters above sea level).
        hr_masl (float): Receiver antenna height (meters above sea level).
        ht_masl (float): Mean terrain elevation along the path, excluding trees (meters above sea
            level).
        lat (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
        lon (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
        timePeriod (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or
            average year).

    Returns:
        p (float): The percentage of time that fade depth A_dB is exceeded in the specified time
            period (%).
    """
    f = f_GHz
    d = d_km
    he = he_masl
    hr = hr_masl
    ht = ht_masl
    epsp = abs(hr-he)/d
    hc = ((hr+he)/2.0) - (d*d/102.0) - ht
    K = GeoclimaticFactorK(lat, lon)
    dN75 = DN75(lat, lon)
    vsrlimit = dN75*pow(d, 1.5)*sqrt(f)/24730.0
    vsr = pow(dN75/50.0, 1.8)*exp(-hc/(2.5*sqrt(d)))
    vsr = min(vsr, vsrlimit)
    hL = min(he, hr)
    xponent = -0.376*tanh((hc-147.0)/125.0) - 0.334*pow(epsp, 0.39) - 0.00027*hL + 17.85*vsr - A_dB/10.0
    pw = K*pow(d, 3.51)*pow((f*f)+13.0, 0.447)*pow(10.0, xponent)
    if timePeriod == TimePeriod.AVG_WORST_MONTH:
        return pw
    else: # AVG_YEAR
        delta_G = _DeltaG(d_km, he_masl, hr_masl, lat)
        p = pow(10.0, -delta_G/10.0)*pw
        return p


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def FadingDistribution(A_dB: float, d_km: float, f_GHz: float, he_masl: float, hr_masl: float,
                       ht_masl: float, lat: float, lon: float, timePeriod: TimePeriod) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.2 and 2.3.4
    Calculates the percentage of time that fade depth A_dB is exceeded in the specified time
    period (%).

    This implementation is suitable for all fade depths and employs the method for large fade depths
    and an interpolation procedure for small fade depths. It combines the deep fading distribution
    given in section 2.3.1 and an empirical interpolation procedure for shallow fading down to 0 dB.

    Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km,
    and can be set to zero for shorter paths.

    Args:
        A_dB (float): Fade depth (dB), with 0 <= A_dB.
        d_km (float): Path length (km), with 0 < d_km.
        f_GHz (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
        he_masl (float): Emitter antenna height (meters above sea level).
        hr_masl (float): Receiver antenna height (meters above sea level).
        ht_masl (float): Mean terrain elevation along the path, excluding trees (meters above sea
            level).
        lat (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
        lon (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
        timePeriod (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or
            average year).
        
    Returns:
        p (float): The percentage of time that fade depth A_dB is exceeded in the specified time
            period (%).
    """
    p0 = SingleFrequencyFadingDistribution(0.0, d_km, f_GHz, he_masl, hr_masl, ht_masl, lat, lon,
             TimePeriod.AVG_WORST_MONTH) # always use worst month here, as of step 1 of 2.3.4
    return _FadeDepthDistribution(A_dB, p0, d_km, he_masl, hr_masl, lat, timePeriod)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _FadeDepthDistribution(A_dB: float, p0: float, d_km: float, he_masl: float, hr_masl: float,
                           lat: float, timePeriod: TimePeriod) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.2 and 2.3.4
    See FadingDistribution() for details.
    """
    At = 25.0 + 1.2*log10(p0)
    if A_dB >= At and timePeriod == TimePeriod.AVG_WORST_MONTH:
        p = p0*pow(10.0, -A_dB/10.0)
    else:
        pt = p0*pow(10.0, -At/10.0)
        if timePeriod == TimePeriod.AVG_YEAR:
            delta_G = _DeltaG(d_km, he_masl, hr_masl, lat)
            pt = pow(10.0, -delta_G/10.0)*pt
        qpa = -20.0*log10(-ln((100.0-pt)/100.0))/At
        qt = (qpa-2.0)/((1.0+0.3*pow(10.0, -At/20.0))*pow(10.0, -0.016*At)) - 4.3*(pow(10.0, -At/20.0)+At/800.0)
        qa = 2.0 + (1.0+0.3*pow(10.0, -A_dB/20.0))*pow(10.0, -0.016*A_dB)*(qt+4.3*(pow(10.0, -A_dB/20.0)+A_dB/800.0))
        p = 100.0*(1.0-exp(-pow(10.0, -qa*A_dB/20.0)))
    return p


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def EnhancementDistribution(E_dB: float, d_km: float, f_GHz: float, he_masl: float, hr_masl: float,
                            ht_masl: float, lat: float, lon: float, timePeriod: TimePeriod) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.3 and 2.3.4
    Calculates the percentage of time that enhancement E_dB is not exceeded in the specified time
    period (%).

    Multipath fading and enhancement only need to be calculated for path lengths longer than 5 km,
    and can be set to zero for shorter paths.

    Args:
        E_dB (float): Enhancement (dB), with 0 <= E_dB.
        d_km (float): Path length (km), with 0 < d_km.
        f_GHz (float): Frequency (GHz), with 15/d_km <= f_GHz <= 45.
        he_masl (float): Emitter antenna height (meters above sea level).
        hr_masl (float): Receiver antenna height (meters above sea level).
        ht_masl (float): Mean terrain elevation along the path, excluding trees (meters above sea
            level).
        lat (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
        lon (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
        timePeriod (crc_covlib.helper.itur_p530.TimePeriod): Time period (average worst month or
            average year).

    Returns:
        p (float): The percentage of time that enhancement E_dB is not exceeded in the specified
            time period (%).
    """
    p0 = SingleFrequencyFadingDistribution(0.0, d_km, f_GHz, he_masl, hr_masl, ht_masl, lat, lon,
             TimePeriod.AVG_WORST_MONTH) # always use worst month here, as of step 1 of 2.3.4
    return _EnhancementDistribution(E_dB, p0, d_km, he_masl, hr_masl, lat, timePeriod)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _EnhancementDistribution(E_dB: float, p0: float, d_km: float, he_masl: float, hr_masl: float,
                             lat: float, timePeriod: TimePeriod) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.3 and 2.3.4
    See EnhancementDistribution() for details.
    """
    if timePeriod == TimePeriod.AVG_WORST_MONTH:
        A001 = 10.0*log10(p0/0.01) # <-- this is what the official matlab code does
        #A001 = InverseDistribution(_FadeDepthDistribution, 0.01, p0, d_km, he_masl, hr_masl, lat,
        #                           timePeriod)
    else: # AVG_YEAR
        delta_G = _DeltaG(d_km, he_masl, hr_masl, lat)
        pw = 0.01/(pow(10.0, -delta_G/10.0))
        A001 = 10.0*log10(p0/pw)
        #A001 = InverseDistribution(_FadeDepthDistribution, pw, p0, d_km, he_masl, hr_masl, lat,
        #                           timePeriod)

    if E_dB > 10.0:
        p = 100.0-pow(10.0, (-1.7+0.2*A001-E_dB)/3.5)
    else:
        Ep = 10.0
        pwp = 100.0-pow(10.0, (-1.7+0.2*A001-Ep)/3.5)
        qep = -(20.0/Ep)*(log10(-ln(1.0-(100.0-pwp)/58.21)))
        qs = 2.05*qep - 20.3
        qe = 8.0 + (1.0+0.3*pow(10.0, -E_dB/20.0)) * pow(10.0, -0.7*E_dB/20.0) * \
             (qs+12.0*(pow(10.0, -E_dB/20.0)+E_dB/800.0))
        p = 100.0-58.21*(1.0-exp(-pow(10.0, -qe*E_dB/20.0)))
    return p


def InverseDistribution(distribFunc: Callable[..., float], p: float, *distribFuncArgs) -> float:
    """
    Related to ITU-R P.530-18, Annex 1, Sections 2.3.1, 2.3.2, 2.3.3 and 2.3.4
    Calculates the fading (dB) exceeded for the specified time percentage, or the enhancement (dB)
    not exceeded for the specified percentage.
    
    Note: Implemented algorithm is not part of the ITU recommendation, use with caution.

    Args:
        distribFunc (Callable[..., float]): A function, one of SingleFrequencyFadingDistribution,
            FadingDistribution or EnhancementDistribution.
        p (float): Percentage of time (%).
        distribFuncArgs: Arguments for distribFunc, with the exception of A_dB or E_dB that should
            not be specified. 

    Returns:
        (float): The fading (dB) exceeded for the specified time percentage, or the enhancement (dB)
            not exceeded for the specified percentage.
    """
    if distribFunc.__name__ == "EnhancementDistribution":
        # the time percentage is for the dB value **not** being exceeded
        return _InverseDistribution(distribFunc, p, True, *distribFuncArgs)
    else:
        # the time percentage is for the dB value being exceeded
        return _InverseDistribution(distribFunc, p, False, *distribFuncArgs)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _InverseDistribution(distribFunc, p: float, notExceeded: bool, *distribFuncArgs) -> float:
    """
    See InverseDistribution() for details.
    """
    min_dB = 0
    max_dB = 300
    mid_dB = (max_dB + min_dB) / 2.0
    while(True):
        temp_p = distribFunc(mid_dB, *distribFuncArgs)
        if (temp_p < p and notExceeded == False) or (temp_p > p and notExceeded == True):
            max_dB = mid_dB
        else:
            min_dB = mid_dB
        temp_mid_dB = (max_dB + min_dB) / 2.0
        if abs(mid_dB-temp_mid_dB) < 1E-4: # arbitrary value
            break
        mid_dB = temp_mid_dB
    return mid_dB


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _DeltaG(d_km: float, he_masl: float, hr_masl: float, lat: float) -> float:
    """
    ITU-R P.530-18, Annex 1, Section 2.3.4 (Step 2)
    Calculates the logarithmic geoclimatic conversion factor (dB).

    Args:
        d_km (float): Path length (km), with 0 < d_km.
        he_masl (float): Emitter antenna height (meters above sea level).
        hr_masl (float): Receiver antenna height (meters above sea level).
        lat (float): Latitude of the path location (degrees), with -90 <= lat <= 90.

    Returns:
        delta_G (float): The logarithmic geoclimatic conversion factor (dB)
    """
    lat = abs(lat)
    if lat > 45:
        sign = -1
    else:
        sign = 1
    lat_rad = radians(lat)
    epsp = (hr_masl-he_masl)/d_km
    delta_G = 10.5 - 5.6*log10(1.1+(sign*pow(abs(cos(2*lat_rad)), 0.7))) - 2.7*log10(d_km) + \
              1.7*log10(1.0+abs(epsp))
    delta_G = max(10.8, delta_G)
    return delta_G


class PathType(enum.Enum):
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.5
    Enumerates path types.
    """
    RELATIVELY_FLAT = 1
    HILLY           = 2
    HILLY_LAND      = 3


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def AvgWorstMonthToShorterWorstPeriod(pw: float, T_hours: float, pathType: PathType) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.3.5
    Converts the percentage of time pw of exceeding a deep fade A in the average worst month to a
    percentage of time pws of exceeding the same deep fade during a shorter worst period of time T.

    Args:
        pw (float): The percentage of time of exceeding a deep fade A in the average worst month.
        T_hours (float): Shorter than a month worst period of time (hours), with 1 <= T_hours < 720.
        pathType (crc_covlib.helper.itur_p530.PathType): Type of path.

    Returns:
        psw (float): Percentage of time of exceeding the deep fade A during the worst T hours.
    """
    psw = pw
    if pathType == PathType.RELATIVELY_FLAT:
        psw = pw*(89.34*pow(T_hours, -0.854)+0.676)
    elif pathType == PathType.HILLY:
        psw = pw*(119.0*pow(T_hours, -0.78)+0.295)
    elif pathType == PathType.HILLY_LAND:
        psw = pw*(199.85*pow(T_hours, -0.834)+0.175)
    return psw


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def RainAttenuationLongTermStatistics(p: float, d_km: float, f_GHz: float,
                                      pathElevAngle_deg: float, polTiltAngle_deg: float,
                                      lat: float, lon: float) -> float:
    """
    ITU-R P.530-18, Annex 1, Sections 2.4.1
    Calculates the rain attenuation over the specified path lengths, exceeded for p percent of
    time, based on yearly statistics (dB).

    Args:
        p (float): Time percentage (%), with 0.001 <= p <= 1.
        d_km (float): Path length (km), with 0 < d_km <= 60.
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 100.
        pathElevAngle_deg (float): Path elevation angle (deg).
        polTiltAngle_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
            for circular polarization.
        lat (float): Latitude of the path location (degrees), with -90 <= lat <= 90.
        lon (float): Longitude of the path location(degrees), with -180 <= lon <= 180.
    
    Returns:
        Ap (float): Rain attenuation over the specified path lengths, exceeded for p percent of
            time, based on yearly statistics (dB).
    """
    # Step 1
    R001 = itur_p837.RainfallRate001(lat, lon)

    # Step 2
    k, alpha = itur_p838.Coefficients(f_GHz, pathElevAngle_deg, polTiltAngle_deg)
    gamma_R = k*pow(R001, alpha)

    # Step 3
    r_inv = 0.477*pow(d_km, 0.633)*pow(R001, 0.073*alpha)*pow(f_GHz, 0.123) - \
            10.579*(1.0-exp(-0.024*d_km))
    r = 1.0/r_inv
    d_eff = d_km*r

    # Step 4
    A001 = gamma_R*d_eff

    # Step 5
    if f_GHz >= 10:
        C0 = 0.12 + 0.4*log10(pow(f_GHz/10.0, 0.8))
    else:
        C0 = 0.12
    C1 = pow(0.07, C0)*pow(0.12, 1.0-C0)
    C2 = 0.855*C0 + 0.546*(1.0-C0)
    C3 = 0.139*C0 + 0.043*(1.0-C0)
    Ap = A001*C1*pow(p, -(C2+C3*log10(p)))

    return Ap


# Data originally from ITU file R-REC-P.530-18-202109-I!!ZIP-E.zip
_DN75 = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p530/dN75.csv', ',')
_LOGK = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p530/LogK.csv', ',')


def FIGURE_3() -> None:
    """
    Calculates values and display FIGURE 3 from ITU-R P.530-18.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    p0_list = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100, 316, 1000]
    pw_list_per_p0 = []
    A_list = np.arange(0, 50.01, 0.5)
    for i, p0 in enumerate(p0_list):
        pw_list_per_p0.append([])
        for A in A_list:
            pw = _FadeDepthDistribution(A, p0, 0, 0, 0, 0, TimePeriod.AVG_WORST_MONTH)
            pw_list_per_p0[i].append(pw)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,50])
    ax1.set_xticks([*range(0, 50+1, 5)])
    ax1.set_yscale('log')
    ax1.set_ylim([1E-5, 1E2])
    ax1.set_yticks([1E-5,1E-4,1E-3,1E-2,1E-1,1,1E1,1E2])
    ax1.minorticks_off()
    for i, p0 in enumerate(p0_list):
        ax1.plot(A_list, pw_list_per_p0[i], color='#000000')
        midPtIndex = int(len(A_list)/2)
        plt.text(A_list[midPtIndex], pw_list_per_p0[i][midPtIndex], '{}'.format(p0))
    ax1.set_title('FIGURE 3\nPercentage of time, pw, fade depth, A, exceeded in average worst ' \
                  'month,\nwith p0 (in equation (11))\nranging from 0.01 to 1 000')
    ax1.set_xlabel('Fade depth, A (dB)')
    ax1.set_ylabel('Percentage of time abscissa is exceeded')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_4() -> None:
    """
    Calculates values and display FIGURE 4 from ITU-R P.530-18.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    p0_list = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100, 316, 1000]
    pw_list_per_p0 = []
    E_list = np.arange(0, 20.01, 0.2)
    for i, p0 in enumerate(p0_list):
        pw_list_per_p0.append([])
        for E in E_list:
            pw = _EnhancementDistribution(E, p0, 0, 0, 0, 0, TimePeriod.AVG_WORST_MONTH)
            pw_list_per_p0[i].append(100.0-pw)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,20])
    ax1.set_xticks([*range(0, 20+1, 2)])
    ax1.set_yscale('log')
    ax1.set_ylim([1E-4, 1E2])
    ax1.set_yticks([1E-4,1E-3,1E-2,1E-1,1,1E1,1E2])
    ax1.minorticks_off()
    for i, p0 in enumerate(p0_list):
        ax1.plot(E_list, pw_list_per_p0[i], color='#000000')
        midPtIndex = int(len(E_list)/2)
        plt.text(E_list[midPtIndex], pw_list_per_p0[i][midPtIndex], '{}'.format(p0))
    ax1.set_title('FIGURE 4\nPercentage of time, (100-pw), enhancement, E, exceeded in the average worst ' \
                  'month,\nwith p0 (in equation (11)) ranging from 0.01 to 1 000')
    ax1.set_xlabel('Enhancement (dB)')
    ax1.set_ylabel('Percentage of time abscissa is exceeded')
    plt.grid(True, 'both','both')
    plt.show()
