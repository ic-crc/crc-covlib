"""Implementation of ITU-R P.1411-12 (partial).
"""

import enum
from math import log10, sqrt, log as ln, pi, atan, tanh, tan, sin, radians, exp, expm1
from numpy import random
from .itur_p1057 import Finv
from .itur_p676 import TerrestrialPathGaseousAttenuation
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['EnvironmentA', # enum
           'EnvironmentB', # enum
           'EnvironmentC',  # enum
           'EnvironmentD', # enum
           'BuildingsLayout', # enum
           'PathType', # enum
           'WarningFlag', # enum
           'SiteGeneralWithinStreetCanyons',
           'SiteGeneralOverRoofTops',
           'SiteGeneralNearStreetLevel',
           'SiteSpecificWithinStreetCanyonsUHFLoS',
           'SiteSpecificWithinStreetCanyonsSHFLoS',
           'SiteSpecificWithinStreetCanyonsEHFLoS',
           'SiteSpecificWithinStreetCanyonsUHFNonLoS',
           'SiteSpecificWithinStreetCanyonsSHFNonLoS',
           'SiteSpecificOverRoofTopsUrban',
           'SiteSpecificOverRoofTopsSuburban',
           'SiteSpecificNearStreetLevelUrban',
           'SiteSpecificNearStreetLevelResidential',
           'FIGURE_4',
           'FIGURE_6'
          ]


class EnvironmentA(enum.Enum):
    URBAN_HIGH_RISE            = 10
    URBAN_LOW_RISE_OR_SUBURBAN = 11
    RESIDENTIAL                = 12


class EnvironmentB(enum.Enum):
    SUBURBAN              = 20
    URBAN                 = 21
    DENSE_URBAN_HIGH_RISE = 22


class EnvironmentC(enum.Enum):
    URBAN       = 30
    RESIDENTIAL = 31


class EnvironmentD(enum.Enum):
    MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE = 40
    METROPOLITAN_CENTRE                 = 41


class BuildingsLayout(enum.Enum):
    WEDGE_SHAPED    = 1
    CHAMFERED_SHAPE = 2


class PathType(enum.Enum):
    LOS  = 1
    NLOS = 2


class WarningFlag(enum.IntEnum):
    NO_WARNINGS       = 0x00
    FREQ_OUT_OF_RANGE = 0x01
    DIST_OUT_OF_RANGE = 0x02


def SiteGeneralWithinStreetCanyons(f_GHz: float, d_m: float, env: EnvironmentA, path: PathType,
                                   addGaussianRandomVar: bool) -> tuple[float, int]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.1
    "This site-general model is applicable to situations where both the transmitting and receiving
    stations are located below-rooftop, regardless of their antenna heights."

    Supported configurations (from recommendation's TABLE 4):
    --------------------------------------------------------------
    f_GHz       d_m         env                           path
    --------------------------------------------------------------
    0.8-82      5-660       URBAN_HIGH_RISE,              LOS
                            URBAN_LOW_RISE_OR_SUBURBAN
    0.8-82      30-715      URBAN_HIGH_RISE               NLOS
    10-73       30-250      URBAN_LOW_RISE_OR_SUBURBAN    NLOS
    0.8-73      30-170      RESIDENTIAL                   NLOS
    --------------------------------------------------------------

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 82 generally, but see
            above table for additional constraints.
        d_m (float): 3D direct distance between the transmitting and receiving stations (m), with
            5 <= d_m <= 715 generally, but see above table for additional constraints.
        env (crc_covlib.helper.itur_p1411.EnvironmentA): One of URBAN_HIGH_RISE,
            URBAN_LOW_RISE_OR_SUBURBAN or RESIDENTIAL.
        path (crc_covlib.helper.itur_p1411.PathType): One of LOS or NLOS. Indicates whether the
            path is line-of-sight or non-line-of-sight.
        addGaussianRandomVar (bool): When set to True, a gaussian random variable is added to the
            median basic transmission loss. Use this option for Monte Carlo simulations for
            example.

    Returns:
        Lb (float): Basic transmission loss (dB). When addGaussianRandomVar is set to False, Lb is
            the median basic transmission loss.
        warnings (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.
    """
    warnings = WarningFlag.NO_WARNINGS.value
    if path == PathType.LOS:
        if env == EnvironmentA.URBAN_HIGH_RISE or env == EnvironmentA.URBAN_LOW_RISE_OR_SUBURBAN:
            alpha, beta, gamma, sigma = 2.12, 29.2, 2.11, 5.06
            min_f, max_f, min_d, max_d = 0.8, 82.0, 5.0, 660.0
        else:
            raise RuntimeError('Unsupported combination of env and path parameter values.')
    else:
        if env == EnvironmentA.URBAN_HIGH_RISE:
            alpha, beta, gamma, sigma = 4.0, 10.2, 2.36, 7.60
            min_f, max_f, min_d, max_d = 0.8, 82.0, 30.0, 715.0
        elif env == EnvironmentA.URBAN_LOW_RISE_OR_SUBURBAN:
            alpha, beta, gamma, sigma = 5.06, -4.68, 2.02, 9.33
            min_f, max_f, min_d, max_d = 10.0, 73.0, 30.0, 250.0
        elif env == EnvironmentA.RESIDENTIAL:
            alpha, beta, gamma, sigma = 3.01, 18.8, 2.07, 3.07
            min_f, max_f, min_d, max_d = 0.8, 73.0, 30.0, 170.0
        else:
            raise RuntimeError('Unsupported combination of env and path parameter values.')

    if f_GHz < min_f or f_GHz > max_f:
        warnings += WarningFlag.FREQ_OUT_OF_RANGE.value

    if d_m < min_d or d_m > max_d:
        warnings += WarningFlag.DIST_OUT_OF_RANGE.value

    Lb = 10.0*alpha*log10(d_m) + beta + 10.0*gamma*log10(f_GHz)
    if addGaussianRandomVar == True:
        if path == PathType.NLOS and (env == EnvironmentA.URBAN_HIGH_RISE or \
                                      env == EnvironmentA.URBAN_LOW_RISE_OR_SUBURBAN):
            c = 2.998E8 # in m/s, as specified in ITU R. P.2001-5
            Lfs = 20.0*log10(4.0E9*pi*d_m*f_GHz/c)
            mu = Lb - Lfs
            A = _GaussianRandom(mu, sigma)
            Lb = Lfs + 10.0*log10(pow(10.0, 0.1*A)+1.0)
        else:
            Lb += _GaussianRandom(0, sigma)

    return (Lb, warnings)


def SiteGeneralOverRoofTops(f_GHz: float, d_m: float, env: EnvironmentA, path: PathType,
                            addGaussianRandomVar: bool) -> tuple[float, int]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.1
    "This site-general model is applicable to situations where one of the stations is located above
    rooftop and the other station is located below-rooftop, regardless of their antenna heights."

    Supported configurations (from from recommendation's TABLE 8):
    --------------------------------------------------------------
    f_GHz       d_m         env                           path
    --------------------------------------------------------------
    2.2-73      55-1200     URBAN_HIGH_RISE,              LOS
                            URBAN_LOW_RISE_OR_SUBURBAN
    2.2-66.5    260-1200    URBAN_HIGH_RISE               NLOS
    --------------------------------------------------------------

    Args:
        f_GHz (float): Operating frequency (GHz), with 2.2 <= f_GHz <= 73 generally, but see
            above table for additional constraints.
        d_m (float): 3D direct distance between the transmitting and receiving stations (m), with
            55 <= d_m <= 1200 generally, but see above table for additional constraints.
        env (crc_covlib.helper.itur_p1411.EnvironmentA): One of URBAN_HIGH_RISE or
            URBAN_LOW_RISE_OR_SUBURBAN.
        path (crc_covlib.helper.itur_p1411.PathType): One of LOS or NLOS. Indicates whether the
            path is line-of-sight or non-line-of-sight.
        addGaussianRandomVar (bool): When set to True, a gaussian random variable is added to the
            median basic transmission loss. Use this option for Monte Carlo simulations for
            example.

    Returns:
        Lb (float): Basic transmission loss (dB). When addGaussianRandomVar is set to False, Lb is
            the median basic transmission loss.
        warnings (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.
    """
    warnings = WarningFlag.NO_WARNINGS.value
    if path == PathType.LOS:
        if env == EnvironmentA.URBAN_HIGH_RISE or env == EnvironmentA.URBAN_LOW_RISE_OR_SUBURBAN:
            alpha, beta, gamma, sigma = 2.29, 28.6, 1.96, 3.48
            min_f, max_f, min_d, max_d = 2.2, 73.0, 55.0, 1200.0
        else:
            raise RuntimeError('Unsupported combination of env and path parameter values.')
    else:
        if env == EnvironmentA.URBAN_HIGH_RISE:
            alpha, beta, gamma, sigma = 4.39, -6.27, 2.30, 6.89
            min_f, max_f, min_d, max_d = 2.2, 66.5, 260.0, 1200.0
        else:
            raise RuntimeError('Unsupported combination of env and path parameter values.')

    if f_GHz < min_f or f_GHz > max_f:
        warnings += WarningFlag.FREQ_OUT_OF_RANGE.value

    if d_m < min_d or d_m > max_d:
        warnings += WarningFlag.DIST_OUT_OF_RANGE.value

    Lb = 10.0*alpha*log10(d_m) + beta + 10.0*gamma*log10(f_GHz)
    if addGaussianRandomVar == True:
        # Note: Should the capping mentioned at page 10 of the rec. be applied here too?
        Lb += _GaussianRandom(0, sigma)

    return (Lb, warnings)


def _GaussianRandom(mean: float, stdDev: float):
    rng = random.default_rng()
    y = mean + (rng.standard_normal()*stdDev)
    return y


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteGeneralNearStreetLevel(f_GHz: float, d_m: float, p: float, env: EnvironmentB,
                               w_m: float=20.0) -> tuple[float, int]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.1
    "[This site-general model is] recommended for propagation between low-height terminals where
    both terminal antenna heights are near street level well below roof-top height, but are
    otherwise unspecified."

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.3 <= f_GHz <= 3.
        d_m (float): Distance between the transmitting and receiving stations (m),
            with 0 < d_m <= 3000.
        p (float): Location percentage (%), with 0 < p < 100.
        env (crc_covlib.helper.itur_p1411.EnvironmentB): One of SUBURBAN, URBAN or
            DENSE_URBAN_HIGH_RISE.
        w_m (float): Transition region width between line-of-sight and non-line-of-sight (m).

    Returns:
        L (float): Basic transmission loss (dB).
        warnings (int): 0 if no warnings. Otherwise contains one or more WarningFlag values.
    """
    warnings = WarningFlag.NO_WARNINGS.value
    f_MHz = f_GHz*1000.0

    # Step 7
    if p < 45.0:
        logp = log10(p/100.0)
        d_LoS = 212.0*logp*logp - 64.0*logp
    else:
        d_LoS = 79.2 - 70.0*(p/100.0)

    # Step 8
    d = d_m
    w = w_m
    if d < d_LoS:
        L = _L_LoS(f_MHz, d, p)
    elif d > (d_LoS+w):
        L = _L_NLoS(f_MHz, d, p, env)
    else:
        L_LoS = _L_LoS(f_MHz, d_LoS, p)
        L_NLoS = _L_NLoS(f_MHz, d_LoS+w, p, env)
        L = L_LoS + (L_NLoS-L_LoS)*(d-d_LoS)/w

    if f_GHz < 0.3 or f_GHz > 3.0:
        warnings += WarningFlag.FREQ_OUT_OF_RANGE.value

    if d_m <= 0 or d_m > 3000.0:
        warnings += WarningFlag.DIST_OUT_OF_RANGE.value

    return (L, warnings)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _L_LoS(f_MHz: float, d_m: float, p: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.1

    Args:
        f_MHz (float): Operating frequency (MHz), with 300 <= f_MHz <= 3000.
        d_m (float): Distance between the transmitting and receiving stations (m),
            with 0 < d_m <= 3000.
        p (float): Location percentage (%), with 0 < p < 100.

    Returns:
        L_LoS (float): Line-of-sight loss (dB).
    """
    # Step 1
    L_median_LoS = 32.45 + 20.0*log10(f_MHz) + 20.0*log10(d_m/1000.0)

    # Step 2
    sigma = 7.0
    delta_L_LoS = 1.5624*sigma*(sqrt(-2.0*ln(1.0-(p/100.0)))-1.1774)

    # Step 3
    L_LoS = L_median_LoS + delta_L_LoS

    return L_LoS


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _L_NLoS(f_MHz: float, d_m: float, p: float, env: EnvironmentB) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.1

    Args:
        f_MHz (float): Operating frequency (MHz), with 300 <= f_MHz <= 3000.
        d_m (float): Distance between the transmitting and receiving stations (m),
            with 0 < d_m <= 3000.
        p (float): Location percentage (%), with 0 < p < 100.
        env (crc_covlib.helper.itur_p1411.EnvironmentB): One of SUBURBAN, URBAN or
            DENSE_URBAN_HIGH_RISE.

    Returns:
        L_NLoS (float): Non-line-of-sight loss (dB).
    """
    # Step 4
    L_urban = 0.0 # SUBURBAN
    if env == EnvironmentB.URBAN:
        L_urban = 6.8
    elif env == EnvironmentB.DENSE_URBAN_HIGH_RISE:
        L_urban = 2.3
    L_median_NLoS = 9.5 + 45.0*log10(f_MHz) + 40.0*log10(d_m/1000.0) + L_urban

    # Step 5
    delta_L_NLoS = 7.0*Finv(p/100.0)

    # Step 6
    L_NLoS = L_median_NLoS + delta_L_NLoS

    return L_NLoS


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz: float, d_m: float, h1_m: float, h2_m: float
                                          ) -> tuple[float, float, float]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.2, UHF propagation
    Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban
    environment (street canyons context) in the UHF frequency range.
    
    Args:
        f_GHz (float): Operating frequency (GHz), with 0.3 <= f_GHz <= 3.
        d_m (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
        h1_m (float): Station 1 antenna height (m), with 0 < h1_m.
        h2_m (float): Station 2 antenna height (m), with 0 < h2_m.

    Returns:
        L_LoS_m (float): Median basic transmission loss (dB).
        L_LoS_l (float): Lower bound of the basic transmission loss (dB).
        L_LoS_u (float): Upper bound of the basic transmission loss (dB).
    """
    d = d_m
    h1 = h1_m
    h2 = h2_m
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    R_bp = 4.0*h1*h2/wavelength # in meters
    L_bp = abs(20.0*log10(wavelength*wavelength/(8.0*pi*h1*h2)))
    logdR = log10(d/R_bp)
    if d <= R_bp:
        L_LoS_l = L_bp + 20.0*logdR
        L_LoS_u = L_bp + 20.0 + 25.0*logdR
        L_LoS_m = L_bp + 6.0 + 20.0*logdR
    else:
        L_LoS_l = L_bp + 40.0*logdR
        L_LoS_u = L_bp + 20.0 + 40.0*logdR
        L_LoS_m = L_bp + 6.0 + 40.0*logdR
    return (L_LoS_m, L_LoS_l, L_LoS_u)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificWithinStreetCanyonsSHFLoS(f_GHz: float, d_m: float, h1_m: float, h2_m: float,
                                          hs_m: float) -> tuple[float, float, float]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.2, SHF propagation up to 15 GHz
    Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban
    environment (street canyons context) in the SHF frequency range.
    
    Args:
        f_GHz (float): Operating frequency (GHz), with 3 <= f_GHz <= 15.
        d_m (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
        h1_m (float): Station 1 antenna height (m), with 0 < h1_m.
        h2_m (float): Station 2 antenna height (m), with 0 < h2_m.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
           traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
           to 1.6 meters.

    Returns:
        L_LoS_m (float): Median basic transmission loss (dB).
        L_LoS_l (float): Lower bound of the basic transmission loss (dB).
        L_LoS_u (float): Upper bound of the basic transmission loss (dB).
    """
    d = d_m
    h1 = h1_m
    h2 = h2_m
    hs = hs_m
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    if h1 > hs and h2 > hs:
        L_bp = abs(20.0*log10(wavelength*wavelength/(8.0*pi*(h1-hs)*(h2-hs))))
        R_bp = 4.0*(h1-hs)*(h2-hs)/wavelength
        logdR = log10(d/R_bp)
        if d <= R_bp:
            L_LoS_l = L_bp + 20.0*logdR
            L_LoS_u = L_bp + 20.0 + 25.0*logdR
            L_LoS_m = L_bp + 6.0 + 20.0*logdR
        else:
            L_LoS_l = L_bp + 40.0*logdR
            L_LoS_u = L_bp + 20.0 + 40.0*logdR
            L_LoS_m = L_bp + 6.0 + 40.0*logdR
    else:
        Rs = 20.0
        if d < Rs:
            L_LoS_m, L_LoS_l, L_LoS_u = SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz=f_GHz, d_m=d,
                                                                              h1_m=h1, h2_m=h2)
        else:
            Ls = abs(20.0*log10(wavelength/(2.0*pi*Rs)))
            logdRs = log10(d/Rs)
            L_LoS_l = Ls + 30.0*logdRs
            L_LoS_u = Ls + 20.0 + 30.0*logdRs
            L_LoS_m = Ls + 6.0 + 30.0*logdRs
    return (L_LoS_m, L_LoS_l, L_LoS_u)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificWithinStreetCanyonsEHFLoS(f_GHz: float, d_m: float, n: float, P_hPa: float=1013.25,
                                          T_K: float=288.15, rho_gm3: float=7.5) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.2, Millimetre-wave propagation
    Site-specific model that estimates the basic transmission loss (dB) for a line-of-sight urban
    environment (street canyons context) in the EHF frequency range (millimetre-wave).
    
    The computed loss includes attenuation by atmospheric gases but does not include attenuation
    due to rain. The RainAttenuationLongTermStatistics() function from itur_p530 may be used to
    calculate this value when required.

    Args:
        f_GHz (float): Operating frequency (GHz), with 10 <= f_GHz <= 100.
        d_m (float): Distance from station 1 to station 2 (m), with 0 < d_m <= 1000.
        n (float): Basic transmission loss exponent. The recommendations's TABLES 7 gives values
            ranging from 1.9 to 2.21.
        P_hPa (float): Atmospheric pressure (hPa), for estimating the attenuation by atmospheric
            gases.
        T_K (float): Temperature (K), for estimating the attenuation by atmospheric gases.
        rho_gm3 (float): Water vapour density (g/m3), for estimating the attenuation by atmospheric
            gases.

    Returns:
        L_LoS (float): Basic transmission loss (dB). Does not include rain attenuation.
    """
    f_MHz = f_GHz*1000.0
    L_gas = TerrestrialPathGaseousAttenuation(pathLength_km=d_m/1000.0, f_GHz=f_GHz, P_hPa=P_hPa,
                                              T_K=T_K, rho_gm3=rho_gm3)
    L0 = 20.0*log10(f_MHz) - 28.0
    L_LoS = L0 + 10.0*n*log10(d_m) + L_gas
    return L_LoS


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificWithinStreetCanyonsUHFNonLoS(f_GHz: float, x1_m: float, x2_m: float, w1_m: float,
                                             w2_m: float, alpha_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.3.1
    Site-specific model that estimates the basic transmission loss (dB) for a non-line-of-sight
    urban environment (street canyons context) in the UHF frequency range. The model considers a
    non-line-of-sight situation "where the diffracted and reflected waves at the corners of the
    street crossings have to be considered."
    
    Args:
        f_GHz (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 2.
        x1_m (float): Distance from station 1 to street crossing (m), with 0 < x1_m.
        x2_m (float): Distance from station 2 to street crossing (m), with 0 < x2_m.
        w1_m (float): Street width at the position of the station 1 (m), with 0 < w1_m.
        w2_m (float): Street width at the position of the station 2 (m), with 0 < w2_m.
        alpha_rad (float): Corner angle (rad), with 0.6 < alpha_rad < pi.
    
    Returns:
        L_NLoS2 (float): Basic transmission loss (dB).
    """
    x1 = x1_m
    x2 = x2_m
    w1 = w1_m
    w2 = w2_m
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    fa = 3.86/pow(alpha_rad, 3.5)
    log4pw = log10(4.0*pi/wavelength)
    Lr = 20.0*log10(x1+x2) + x1*x2*fa/(w1*w2) + 20.0*log4pw
    Da = (20.0/pi)*(atan(x2/w2)+atan(x1/w1)-(pi/2.0))
    Ld = 10.0*log10(x1*x2*(x1+x2)) + 2.0*Da - 0.1*(90.0-alpha_rad*180.0/pi) + 20.0*log4pw
    L_NLoS2 = -10.0*log10(pow(10.0,-Lr/10.0)+pow(10.0,-Ld/10.0)) # the rec. is missing the front minus sign
    return L_NLoS2


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificWithinStreetCanyonsSHFNonLoS(f_GHz: float, x1_m: float, x2_m: float, w1_m: float,
                                             h1_m: float, h2_m: float, hs_m: float, n: float,
                                             env: EnvironmentC,
                                             bldgLayout: BuildingsLayout=BuildingsLayout.WEDGE_SHAPED,
                                             P_hPa: float=1013.25, T_K: float=288.15,
                                             rho_gm3: float=7.5) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.1.3.2
    Site-specific model that estimates the basic transmission loss (dB) for a non-line-of-sight
    urban or residential environment (street canyons context) in the SHF frequency range. The model
    considers a non-line-of-sight situation "where the diffracted and reflected waves at the
    corners of the street crossings have to be considered." It assumes street corner angles of pi/2
    radians (90 degrees).

    For frequencies from 10 GHz and up, the losses in the line-of-sight region do not include the
    attenuation due to rain (see eq.(13)). The RainAttenuationLongTermStatistics() function from
    itur_p530 may be used to calculate this value when required.

    Args:
        f_GHz (float): Operating frequency (GHz), with 2 <= f_GHz <= 38.
        x1_m (float): Distance from station 1 to street crossing (m), with 20 < x1_m.
        x2_m (float): Distance from station 2 to street crossing (m), with 0 <= x2_m.
        w1_m (float): Street width at the position of the station 1 (m), with 0 < w1_m.
        h1_m (float): Station 1 antenna height (m), with 0 < h1_m.
        h2_m (float): Station 2 antenna height (m), with 0 < h2_m.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
           traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
           to 1.6 meters.
        n (float): Basic transmission loss exponent. The recommendations's TABLES 7 gives values
            ranging from 1.9 to 2.21.
        env (crc_covlib.helper.itur_p1411.EnvironmentC): URBAN or RESIDENTIAL.
        bldgLayout (crc_covlib.helper.itur_p1411.BuildingsLayout): WEDGE_SHAPED or CHAMFERED_SHAPE.
            See recommendatinon's FIGURE 5. Only applies when env is set to URBAN.
        P_hPa (float): Atmospheric pressure (hPa), for estimating the attenuation by atmospheric
            gases.
        T_K (float): Temperature (K), for estimating the attenuation by atmospheric gases.
        rho_gm3 (float): Water vapour density (g/m3), for estimating the attenuation by atmospheric
            gases.

    Returns:
        L_NLoS2 (float): Basic transmission loss (dB).
    """
    x1 = x1_m
    x2 = x2_m
    w1 = w1_m

    beta = 6.0
    d_corner = 30.0 # meters
    if env == EnvironmentC.RESIDENTIAL:
        L_corner = 30.0 # dB
    else: # URBAN
        L_corner = 20.0 # dB
        if bldgLayout == BuildingsLayout.CHAMFERED_SHAPE:
            f_MHz = f_GHz*1000.0
            beta = 4.2 + (1.4*log10(f_MHz)-7.8)*(0.8*log10(x1)-1.0)

    if x2 > (w1/2.0)+1.0+d_corner:
        # NLOS region
        Lc = L_corner
        L_att = 10.0*beta*log10((x1+x2)/(x1+(w1/2.0)+d_corner))
    else:
        if (w1/2.0)+1.0 < x2:
            # Corner region
            Lc = (L_corner/log10(1.0+d_corner))*log10(x2-(w1/2.0))
        else:
            # LOS region
            Lc = 0.0
        L_att = 0.0

    if f_GHz < 3.0:
        L_LoS, _, _ = SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz=f_GHz, d_m=x1, h1_m=h1_m,
                                                            h2_m=h2_m)
    elif f_GHz < 10.0:
        L_LoS, _, _ = SiteSpecificWithinStreetCanyonsSHFLoS(f_GHz=f_GHz, d_m=x1, h1_m=h1_m,
                                                            h2_m=h2_m, hs_m=hs_m)
    else:
        L_LoS = SiteSpecificWithinStreetCanyonsEHFLoS(f_GHz=f_GHz, d_m=x1, n=n, P_hPa=P_hPa,
                                                      T_K=T_K, rho_gm3=rho_gm3)

    # Note: There seems to be an error in eq.(19), Ls should be Lc, as it was in previous versions
    #       of the recommendation.
    L_NLoS2 = L_LoS + Lc + L_att

    return L_NLoS2


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificOverRoofTopsUrban(f_GHz: float, d_m: float, h1_m: float, h2_m: float, hr_m: float,
                                  l_m: float, b_m: float, w2_m: float, phi_deg: float,
                                  env: EnvironmentD) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.1 & FIGURE 2
    Site-specific model that uses a multi-screen diffraction model to estimate the basic
    transmission loss (dB) in an urban environment. "The multi-screen diffraction model ...is valid
    if the roof-tops are all about the same height". It assumes "the roof-top heights differ only
    by an amount less than the first Fresnel-zone radius over [the path length]". The model
    includes "free-space basic transmission loss, ...the diffraction loss from roof-top to street
    ...and the reduction due to multiple screen diffraction past rows of buildings".

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 26 in general, but with
            2 <= f_GHz <= 16 for h1_m < hr_m and w2_m < 10.
        d_m (float): Path length (m), with 20 <= d_m <= 5000.
        h1_m (float): Station 1 antenna height (m), with 4 <= h1_m <= 55.
        h2_m (float): Station 2 antenna height (m), with 1 <= h2_m <= 3.
        hr_m (float): Average height of buildings (m), with h2_m < hr_m. 
        l_m (float): Length of the path covered by buildings (m), with 0 <= l_m. When l_m is set to
            zero, only the free-space basic transmission loss component is computed and returned
            (using f_GHz and d_m while other parameters are ignored).
        b_m (float): Average building separation (m), with 0 < b_m.
        w2_m (float): Street width (m) at station 2's location, with 0 < w2_m.
        phi_deg (float): Street orientation with respect to the direct path (deg),
            with 0 <= phi_deg <= 90 (i.e. for a street that is perpendicular to the direct path,
            phi_deg is 90 deg).
        env (crc_covlib.helper.itur_p1411.EnvironmentD): MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE or
            METROPOLITAN_CENTRE. Only used when f_GHz <= 2.
    
    Returns:
        L_NLoS1 (float): Basic transmission loss (dB).
    """
    f_MHz = f_GHz*1000.0
    c = 2.998E8 # in m/s
    wavelength = 1.0E-6*c/f_MHz # in meters
    L_bf = 32.4 + 20.0*log10(d_m/1000.0) + 20.0*log10(f_MHz)

    if l_m <= 0:
        return L_bf

    if phi_deg < 35:
        L_ori = -10.0 + 0.354*phi_deg
    elif phi_deg < 55:
        L_ori = 2.5 + 0.075*(phi_deg-35.0)
    else:
        L_ori = 4.0 - 0.114*(phi_deg-55.0)
    delta_h2 = hr_m - h2_m
    L_rts = -8.2 - 10.0*log10(w2_m) + 10.0*log10(f_MHz) + 20.0*log10(delta_h2) + L_ori

    delta_h1 = h1_m - hr_m
    ds = wavelength*d_m*d_m/(delta_h1*delta_h1)
    d_bp = abs(delta_h1)*sqrt(l_m/wavelength) # be careful to use l (not 1)
    L_upp = _L1msd(f_MHz=f_MHz, d_m=d_bp, h1_m=h1_m, hr_m=hr_m, b_m=b_m, env=env)
    L_low = _L2msd(f_MHz=f_MHz, d_m=d_bp, h1_m=h1_m, hr_m=hr_m, b_m=b_m)
    L_mid = (L_upp+L_low)/2.0
    zeta = (L_upp-L_low)*0.0417
    dh_bp = L_upp-L_low

    L1_msd = _L1msd(f_MHz=f_MHz, d_m=d_m, h1_m=h1_m, hr_m=hr_m, b_m=b_m, env=env)
    L2_msd = _L2msd(f_MHz=f_MHz, d_m=d_m, h1_m=h1_m, hr_m=hr_m, b_m=b_m)

    if l_m > ds and dh_bp > 0.0:
        L_msd = -tanh((log10(d_m)-log10(d_bp))/0.1)*(L1_msd-L_mid)+L_mid
    elif l_m <= ds and dh_bp > 0.0:
        L_msd = tanh((log10(d_m)-log10(d_bp))/0.1)*(L2_msd-L_mid)+L_mid
    elif dh_bp == 0.0:
        L_msd = L2_msd
    elif l_m > ds and dh_bp < 0.0:
        L_msd = L1_msd-tanh((log10(d_m)-log10(d_bp))/zeta)*(L_upp-L_mid)-L_upp+L_mid
    elif l_m <= ds and dh_bp < 0.0:
        L_msd = L2_msd+tanh((log10(d_m)-log10(d_bp))/zeta)*(L_mid-L_low)+L_mid-L_low
    else:
        raise RuntimeError('Cannot calculate L_msd.')

    L_NLoS1 = max(L_bf+L_rts+L_msd, L_bf)

    return L_NLoS1


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _L1msd(f_MHz: float, d_m: float, h1_m: float, hr_m: float, b_m: float, env: EnvironmentD
          ) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.1

    Args:
        f_MHz (float): Operating frequency (GHz), with 800 <= f_MHz <= 26000.
        d_m (float): Path length (m), with 20 <= d_m <= 5000.
        h1_m (float): Station 1 antenna height (m), with 4 <= h1_m <= 55.
        hr_m (float): Average height of buildings (m), with h2_m < hr_m. 
        b_m (float): Average building separation (m), with 0 < b_m. See FIGURE 2.
        env (crc_covlib.helper.itur_p1411.EnvironmentD): MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE or
            METROPOLITAN_CENTRE. Only used when f_GHz <= 2.
        
    Returns:
        L1_msd (float): Intermediate loss value in the calculation of L_msd (dB).
    """
    delta_h1 = h1_m - hr_m

    if h1_m > hr_m:
        L_bsh = -18.0*log10(1.0+delta_h1)
        kd = 18.0
        if f_MHz > 2000.0:
            ka = 71.4
        else:
            ka = 54.0
    else:
        L_bsh = 0.0
        kd = 18.0 - 15.0*delta_h1/hr_m
        if f_MHz > 2000.0:
            if d_m < 500.0:
                ka = 73.0 - 1.6*delta_h1*d_m/1000.0
            else:
                ka = 73.0 - 0.8*delta_h1
        else:
            if d_m < 500.0:
                ka = 54.0 - 1.6*delta_h1*d_m/1000.0
            else:
                ka = 54.0 - 0.8*delta_h1

    if f_MHz > 2000.0:
        kf = -8.0
    else:
        if env == EnvironmentD.MEDIUM_SIZED_CITY_OR_SUBURAN_CENTRE:
            kf = -4.0 + 0.7*((f_MHz/925.0)-1.0)
        else: # METROPOLITAN_CENTRE
            kf = -4.0 + 1.5*((f_MHz/925.0)-1.0)

    L1_msd = L_bsh + ka + kd*log10(d_m/1000.0) + kf*log10(f_MHz) - 9.0*log10(b_m)

    return L1_msd


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _L2msd(f_MHz: float, d_m: float, h1_m: float, hr_m: float, b_m: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.1

    Args:
        f_MHz (float): Operating frequency (GHz), with 800 <= f_MHz <= 26000.
        d_m (float): Path length (m), with 20 <= d_m <= 5000.
        h1_m (float): Station 1 antenna height (m), with 4 <= h1_m <= 55.
        hr_m (float): Average height of buildings (m), with h2_m < hr_m. 
        b_m (float): Average building separation (m), with 0 < b_m. See FIGURE 2.

    Returns:
        L2_msd (float): Intermediate loss value in the calculation of L_msd (dB).
    """
    c = 2.998E8 # in m/s
    wavelength = 1.0E-6*c/f_MHz # in meters

    dhu_exp = -log10(sqrt(b_m/wavelength)) - log10(d_m)/9.0 + (10.0/9.0)*log10(b_m/2.35)
    dhu = pow(10.0, dhu_exp)

    dhl = (0.00023*b_m*b_m - 0.1827*b_m - 9.4978)/(pow(log10(f_MHz),2.938)) + 0.000781*b_m + 0.06923

    delta_h1 = h1_m - hr_m

    if h1_m > (hr_m + dhu):
        QM = 2.35*pow((delta_h1/d_m)*sqrt(b_m/wavelength), 0.9)
    elif h1_m <= (hr_m + dhu) and h1_m >= (hr_m + dhl):
        QM = b_m/d_m
    elif h1_m < (hr_m + dhl):
        theta_rad = atan(delta_h1/b_m)
        rho = sqrt(delta_h1*delta_h1 + b_m*b_m)
        QM = (b_m/(2.0*pi*d_m))*sqrt(wavelength/rho)*((1.0/theta_rad)-(1.0/(2*pi+theta_rad)))
    else:
        raise RuntimeError('Cannot calculate QM.')

    L2_msd = -10.0*log10(QM*QM)

    return L2_msd


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificOverRoofTopsSuburban(f_GHz: float, d_m: float, h1_m: float, h2_m: float,
                                     hr_m: float, w2_m: float, phi_deg: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2
    Site-specific model that estimates the basic transmission loss (dB) for a suburban environment.
    The estimated loss "can be divided into three regions in terms of the dominant arrival waves at
    station 2. These are the direct wave, reflected wave, and diffracted wave dominant regions."

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.8 <= f_GHz <= 38.
        d_m (float): Path length (m), with 10 <= d_m <= 5000.
        h1_m (float): Station 1 antenna height (m), with hr_m+1 <= h1_m <= hr_m+100.
        h2_m (float): Station 2 antenna height (m), with hr_m-10 <= h2_m <= hr_m-4.
        hr_m (float): Average height of buildings (m).
        w2_m (float): Street width (m) at station 2's location, with 10 <= w2_m <= 25.
        phi_deg (float): Street orientation with respect to the direct path (deg),
            with 0 < phi_deg <= 90 (i.e. for a street that is perpendicular to the direct path,
            phi_deg is 90 deg).

    Returns:
        L_NLoS1 (float): Basic transmission loss (dB).
    """
    phi_rad = radians(phi_deg)
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    d0 = _dk(k=0, h1=h1_m, h2=h2_m, hr=hr_m, w=w2_m, phi_rad=phi_rad)
    if d_m < d0:
        L_NLoS1 = 20.0*log10(4.0*pi*d_m/wavelength)
    else:
        d_RD = _dRD(f_GHz=f_GHz, h1=h1_m, h2=h2_m, hr=hr_m, w=w2_m, phi_rad=phi_rad)
        if d_m < d_RD:
            L_NLoS1 = _L0n(f_GHz=f_GHz, d=d_m, h1=h1_m, h2=h2_m, hr=hr_m, w=w2_m, phi_rad=phi_rad)
        else:
            L_d_RD = _LdRD(f_GHz=f_GHz, h1=h1_m, h2=h2_m, hr=hr_m, w=w2_m, phi_rad=phi_rad)
            L_NLoS1 = 32.1*log10(d_m/d_RD) + L_d_RD

    return L_NLoS1


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _L0n(f_GHz: float, d: float, h1: float, h2: float, hr: float, w: float, phi_rad: float
        ) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(49)
    """
    max_k = 10
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    dRD = _dRD(f_GHz=f_GHz, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    result_calculated = False
    dk = _dk(k=0, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    for k in range(0, max_k):
        dkp1 = _dk(k=k+1, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
        if dk <= d and d < dkp1 and dkp1 < dRD:
            Ldk = _Ldk(k=k, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad, wavelength=wavelength)
            Ldkp1 = _Ldk(k=k+1, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad, wavelength=wavelength)
            L0n = Ldk + ((Ldkp1-Ldk)/(dkp1-dk))*(d-dk)
            result_calculated = True
            break
        elif dk <= d and d < dRD and dRD < dkp1:
            Ldk = _Ldk(k=k, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad, wavelength=wavelength)
            LdRD = _LdRD(f_GHz=f_GHz, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
            L0n = Ldk + ((LdRD-Ldk)/(dRD-dk))*(d-dk)
            result_calculated = True
            break
        dk = dkp1
    if result_calculated == False:
        raise RuntimeError('Cannot calculate L0n.')
    return L0n


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _dk(k: int, h1: float, h2: float, hr: float, w: float, phi_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(50)
    """
    a = _Bk(k=k, h1=h1, h2=h2, hr=hr, w=w)/sin(phi_rad)
    b = h1-h2
    return sqrt(a*a+b*b)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _Ldk(k: int, h1: float, h2: float, hr: float, w: float, phi_rad: float, wavelength: float
        ) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(51)
    """
    a = 4.0*pi*_dkp(k=k, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    b = pow(0.4, k)*wavelength
    return 20.0*log10(a/b)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _dRD(f_GHz: float, h1: float, h2: float, hr: float, w: float, phi_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(52)
    """
    d1 = _dk(k=1, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    d2 = _dk(k=2, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    d3 = _dk(k=3, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    d4 = _dk(k=4, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    return (0.25*d3 + 0.25*d4 - 0.16*d1 - 0.35*d2)*log10(f_GHz) + 0.25*d1 + 0.56*d2 + 0.10*d3 + 0.10*d4


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _LdRD(f_GHz: float, h1: float, h2: float, hr: float, w: float, phi_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2,  eq.(53)
    """
    max_k = 10
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    dRD = _dRD(f_GHz=f_GHz, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    result_calculated = False
    dk = _dk(k=0, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
    for k in range(0, max_k):
        dkp1 = _dk(k=k+1, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad)
        if dk <= dRD and dRD <= dkp1:
            Ldk = _Ldk(k=k, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad, wavelength=wavelength)
            Ldkp1 = _Ldk(k=k+1, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad, wavelength=wavelength)
            LdRD = Ldk + ((Ldkp1-Ldk)/(dkp1-dk))*(dRD-dk)
            result_calculated = True
            break
        dk = dkp1
    if result_calculated == False:
        raise RuntimeError('Cannot calculate LdRD.')
    return LdRD


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _dkp(k: int, h1: float, h2: float, hr: float, w: float, phi_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(54)
    """
    a = _Ak(k=k, h1=h1, h2=h2, hr=hr, w=w)/sin(_phik(k=k, h1=h1, h2=h2, hr=hr, w=w, phi_rad=phi_rad))
    b = h1-h2
    return sqrt(a*a+b*b)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _Ak(k: int, h1: float, h2: float, hr: float, w: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(55)
    """
    return w*(h1-h2)*(2.0*k+1.0)/(2.0*(hr-h2))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _Bk(k: int, h1: float, h2: float, hr: float, w: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(56)
    """
    return w*(h1-h2)*(2.0*k+1.0)/(2.0*(hr-h2)) - k*w


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _phik(k: int, h1: float, h2: float, hr: float, w: float, phi_rad: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.2.2.2, eq.(57)
    """
    Ak = _Ak(k=k, h1=h1, h2=h2, hr=hr, w=w)
    Bk = _Bk(k=k, h1=h1, h2=h2, hr=hr, w=w)
    return atan((Ak/Bk)*tan(phi_rad))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificNearStreetLevelUrban(f_GHz: float, x1_m: float, x2_m: float, x3_m: float,
                                     h1_m: float, h2_m: float, hs_m: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.2
    Site-specific model that estimates the basic transmission loss (dB) in a rectilinear street
    grid urban environment.

    When both x2_m and x3_m are set to zero, the estimate is based on the algorithm for line-of-
    sight situations (section 4.3.2.1).
    When x3_m only is set to zero, the estimate is based on the "1-Turn NLoS propagation" algorithm
    (section 4.3.2.2).
    When x1_m, x2_m and x3_m values are all greater than zero, the estimate is based on the "2-Turn
    NLoS propagation" algorithm (section 4.3.2.2). For a 2-Turn NLoS link, "it is possible to
    establish multiple travel route paths". Use equation (68) from the recommendation to consider
    all 2-Turn route paths in the overall path loss calculation.
    
    Args:
        f_GHz (float): Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
        x1_m (float): Distance between station 1 and the first street corner (m), with 0 < x1_m.
        x2_m (float): Distance between the first street corner and the second street corner (m),
            with 0 <= x2_m.
        x3_m (float): Distance between the second street corner and station 2 (m), with 0 <= x3_m.
        h1_m (float): Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
        h2_m (float): Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
            traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
            to 1.6 meters. Only used when f_GHz >= 3.

    Return:
        L (float): Basic transmission loss (dB).
    """
    if x2_m > 0 and x3_m > 0:
        L = _SiteSpecNearStreetLevelUrban2TurnNLoS(f_GHz=f_GHz, x1_m=x1_m, x2_m=x2_m, x3_m=x3_m,
                                                   h1_m=h1_m, h2_m=h2_m, hs_m=hs_m)
    elif x2_m > 0:
        L = _SiteSpecNearStreetLevelUrban1TurnNLoS(f_GHz=f_GHz, x1_m=x1_m, x2_m=x2_m, h1_m=h1_m,
                                                   h2_m=h2_m, hs_m=hs_m)
    else:
        L = _SiteSpecNearStreetLevelUrbanLoS(f_GHz=f_GHz, d_m=x1_m, h1_m=h1_m, h2_m=h2_m,
                                             hs_m=hs_m)
    return L


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SiteSpecNearStreetLevelUrbanLoS(f_GHz: float, d_m: float, h1_m: float, h2_m: float,
                                     hs_m: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.2.1
    Calculates the basic transmission loss (dB) for the LoS situation in rectilinear street grid
    environments (urban area).

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
        d_m (float): Distance from Station 1 to Station 2 (m), with 0 < d_m <= 1000.
        h1_m (float): Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
        h2_m (float): Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
           traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
           to 1.6 meters. Only used when f_GHz >= 3.

    Returns:
        L_LoS (float): Median basic transmission loss (dB).
    """
    if f_GHz < 3:
        L_LoS, _, _ = SiteSpecificWithinStreetCanyonsUHFLoS(f_GHz=f_GHz, d_m=d_m, h1_m=h1_m,
                                                            h2_m=h2_m)
    else:
        L_LoS, _, _ = SiteSpecificWithinStreetCanyonsSHFLoS(f_GHz=f_GHz, d_m=d_m, h1_m=h1_m,
                                                            h2_m=h2_m, hs_m=hs_m)
    return L_LoS


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SiteSpecNearStreetLevelUrban1TurnNLoS(f_GHz: float, x1_m: float, x2_m: float, h1_m: float,
                                           h2_m: float, hs_m: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.2.2, 1-Turn NLoS propagation
    Calculates the basic transmission loss (dB) for the 1-Turn NLoS situation in rectilinear street
    grid environments (urban area). "The maximum distance between terminals is up to 1000 m".

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
        x1_m (float): Distance from Station 1 to street crossing (m), with 0 < x1_m.
        x2_m (float): Distance from Station 2 to street crossing (m), with 0 <= x2_m.
        h1_m (float): Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
        h2_m (float): Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
           traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
           to 1.6 meters. Only used when f_GHz >= 3.

    Return:
        L_1Turn (float): Basic transmission loss (dB).
    """
    # Note: rec. P.1411-12 refers to section 4.1.1 here but this is likely a mistake, it should be
    #       referring to section 4.1.2.
    L_LoS = _SiteSpecNearStreetLevelUrbanLoS(f_GHz=f_GHz, d_m=x1_m+x2_m, h1_m=h1_m, h2_m=h2_m,
                                             hs_m=hs_m)

    f_Hz = f_GHz*1.0E9
    S1 = 3.45E4*pow(f_Hz, -0.46)
    d_corner = 30
    if x2_m > max(S1*S1, d_corner):
        L_1Turn = L_LoS + 10.0*log10((x1_m*x2_m)/(x1_m+x2_m)) - 20.0*log10(S1)
    else:
        d0 = 0
        L0_dB = L_LoS
        L0 = pow(10.0, L0_dB/10.0)
        d1 = max(S1*S1, d_corner)
        L1_dB = L_LoS + 10.0*log10((x1_m*d1)/(x1_m+d1)) - 20.0*log10(S1)
        L1 = pow(10.0, L1_dB/10.0)
        d = x2_m
        L = (L0*(d1-d) + L1*(d-d0))/(d1-d0)
        L_1Turn = 10.0*log10(L)

    return L_1Turn


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _SiteSpecNearStreetLevelUrban2TurnNLoS(f_GHz: float, x1_m: float, x2_m: float, x3_m: float,
                                           h1_m: float, h2_m: float, hs_m: float) -> float:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.2.2, 2-Turn NLoS propagation
    Calculates the basic transmission loss (dB) for the 2-Turn NLoS situation in rectilinear street
    grid environments (urban area). "The maximum distance between terminals is up to 1000 m".
    
    For a 2-Turn NLoS link, "it is possible to establish multiple travel route paths". See equation
    (68) from the recommendation for considering all 2-Turn route paths in the overall path loss
    calculation.

    Args:
        f_GHz (float): Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
        x1_m (float): Distance between Station 1 and the first street corner (m), with 0 < x1_m.
        x2_m (float): Distance between the first street corner and the second street corner (m),
            with 0 < x2_m.
        x3_m (float): Distance between the second street corner and Station 2 (m), with 0 <= x3_m.
        h1_m (float): Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
        h2_m (float): Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
        hs_m (float): Effective road height (m), with 0 <= hs_m. hs_m varies depending on the
           traffic on the road. The recommendations's TABLES 5 and 6 give values ranging from 0.23
           to 1.6 meters. Only used when f_GHz >= 3.

    Return:
        L_2Turn_n (float): Basic transmission loss (dB).
    """    
    L_LoS = _SiteSpecNearStreetLevelUrbanLoS(f_GHz=f_GHz, d_m=x1_m+x2_m, h1_m=h1_m, h2_m=h2_m,
                                             hs_m=hs_m)
    f_Hz = f_GHz*1.0E9
    S1 = 3.45E4*pow(f_Hz, -0.46)
    S2 = 0.54*pow(f_Hz, 0.076)
    d_corner = 30
    if x3_m > max(S2*S2, d_corner):
        L_2Turn_n = L_LoS + 10.0*log10((x1_m*x2_m*x3_m)/(x1_m+x2_m+x3_m)) - 20.0*log10(S1) \
                    - 20.0*log10(S2)
    else:
        d0 = 0
        L0_dB = L_LoS + 10.0*log10((x1_m*x2_m)/(x1_m+x2_m)) - 20.0*log10(S1)
        L0 = pow(10.0, L0_dB/10.0)
        d1 = max(S2*S2, d_corner)
        L1_dB = L_LoS + 10.0*log10((x1_m*x2_m*d1)/(x1_m+x2_m+d1)) - 20.0*log10(S1) - 20.0*log10(S2)
        L1 = pow(10.0, L1_dB/10.0)
        d = x3_m
        L = (L0*(d1-d) + L1*(d-d0))/(d1-d0)
        L_2Turn_n = 10.0*log10(L)

    return L_2Turn_n


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SiteSpecificNearStreetLevelResidential(f_GHz: float, d_m: float, hTx_m: float, hRx_m: float,
                                           hbTx_m: float, hbRx_m: float, a_m: float, b_m: float,
                                           c_m: float, m_m: float, n_bldgkm2: float,
                                           thetaList_deg: list[float], x1List_m: list[float],
                                           x2List_m: list[float]
                                           ) -> tuple[float, float, float, float]:
    """
    ITU-R P.1411-12, Annex 1, Section 4.3.3
    Site-specific model "that predicts whole path loss L between two terminals of low height in 
    residential environments ...by using path loss along a road Lr, path loss between houses Lb,
    and over-roof basic transmission loss Lv". "Applicable areas are both [line-of-sight] and
    [non-line-of-sight] regions that include areas having two or more corners".

    Args:
        f_GHz (float): Operating frequency (GHz), with 2 <= f_GHz <= 26.
        d_m (float): Distance between the two terminals (m), with 0 <= d_m <= 1000.
        hTx_m (float): Transmitter antenna height (m), with 1.2 <= hTx_m <= 6.
        hRx_m (float): Receiver antenna height (m), with 1.2 <= hRx_m <= 6.
        hbTx_m (float): Height of nearest building from transmitter in receiver direction (m).
        hbRx_m (float): Height of nearest building from receiver in transmitter direction (m).
        a_m (float): Distance between transmitter and nearest building from transmitter (m).
        b_m (float): Distance between nearest buildings from transmitter and receiver (m).
        c_m (float): Distance between receiver and nearest building from receiver (m).
        m_m (float): Average building height of the buildings with less than 3 stories (m).
            Typically between 6 and 12 meters.
        n_bldgkm2 (float): Building density (buildings/km2).
        thetaList_deg (list[float]): List of theta_i, where theta_i is the road angle of i-th
            street corner (degrees), with 0 <= theta_i <= 90.
        x1List_m (list[float]): List of x1_i, where x1_i is the road distance from transmitter to
            i-th street corner (m).
        x2List_m (list[float]): List of x2_i, where x2_i is the road distance from i-th street
            corner to receiver (m). 
    Return:
        L (float): Basic transmission loss (dB).
        Lr (float): Path loss along road (dB).
        Lb (float): Path loss between houses (dB).
        Lv (float): Over-roof basic transmission loss (dB).
    """
    c = 2.998E8 # in m/s
    wavelength = 1.0E-9*c/f_GHz # in meters
    log4pdy = log10(4.0*pi*d_m/wavelength)
    L_rbc = 20.0*log4pdy
    logf = log10(f_GHz)
    L_rac = L_rbc
    for theta_i, x1_i, x2_i in zip(thetaList_deg, x1List_m, x2List_m):
        L_rac += (7.18*log10(theta_i) + 0.97*logf + 6.1) * (1.0 - exp(-3.72E-5*theta_i*x1_i*x2_i))
    Lr = L_rac

    w0 = 15.0
    alpha = 0.55
    beta = 0.18
    l = 6.0
    l3 = 12.0
    gamma = (l3-hRx_m)/(m_m-l)
    delta = 1.0+beta*(m_m-l)
    wp = (4.0/pi)*w0*(1.0 - ((alpha*(-expm1(-delta*gamma)))/(delta*delta*(-expm1(-gamma)))) * exp(-beta*hRx_m))
    R = (1000.0*gamma/(n_bldgkm2*wp*(-expm1(-gamma))))*exp((hRx_m-l)/(m_m-l))
    Lb = 20.0*log4pdy + 30.6*log10(d_m/R) + 6.88*logf + 5.76

    v1 = (hbTx_m-hTx_m)*sqrt((2.0/wavelength)*((1.0/a_m)+(1.0/b_m)))
    v2 = (hbRx_m-hRx_m)*sqrt((2.0/wavelength)*((1.0/b_m)+(1.0/c_m)))
    L1 = 6.9 + 20.0*log10(sqrt((v1-0.1)*(v1-0.1)+1.0) + v1 - 0.1)
    L2 = 6.9 + 20.0*log10(sqrt((v2-0.1)*(v2-0.1)+1.0) + v2 - 0.1)
    Lc = 10.0*log10((a_m+b_m)*(b_m+c_m)/(b_m*(a_m+b_m+c_m)))
    Lv = 20.0*log4pdy + L1 + L2 + Lc

    L = -10.0*log10(1.0/pow(10.0, Lr/10.0) + 1.0/pow(10.0, Lb/10.0) + 1.0/pow(10.0, Lv/10.0))
    return (L, Lr, Lb, Lv)


def FIGURE_4() -> None:
    """
    Calculates values and displays a graph similar to FIGURE 4 from ITU-R P.1411-12.
    """
    import matplotlib.pyplot as plt
    h1 = 5.0 # Station 1 antenna height (m)
    h2 = 1.5 # Station 2 antenna height (m)
    x1 = 200 #  Distance Station 1 to street crossing (m)
    x2 = 400 # Distance Station 2 to street crossing (m)
    w1 = 10.0 # Street width at the position of the Station 1 (m)
    env = EnvironmentC.URBAN
    bldgLayout = BuildingsLayout.WEDGE_SHAPED
    hs = 0.75 #  Effective road height (m)
    n = 2.05 # Basic transmission loss exponent
    dists = []
    freqs = [2,5,10,20,30]
    losses = [[],[],[],[],[]]
    for d_m in range(1, x1+x2+1, 1):
        dists.append(d_m)
        for i, f_GHz in enumerate(freqs):
            if d_m <= x1:
                x1_temp = d_m 
                x2_temp = 0.0
            else:
                x1_temp = x1
                x2_temp = d_m-x1
            L_NLoS2 = SiteSpecificWithinStreetCanyonsSHFNonLoS(f_GHz=f_GHz, x1_m=x1_temp,
                          x2_m=x2_temp, w1_m=w1, h1_m=h1, h2_m=h2, hs_m=hs, n=n, env=env,
                          bldgLayout=bldgLayout)
            losses[i].append(L_NLoS2)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,x1+x2])
    ax1.set_xticks([*range(0, x1+x2+1, 50)])
    ax1.set_ylim([160,40])
    ax1.set_yticks([*range(40, 160+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='2 GHz')
    ax1.plot(dists, losses[1], color='#E31E24', label='5 GHz')
    ax1.plot(dists, losses[2], color='#1D8545', label='10 GHz')
    ax1.plot(dists, losses[3], color='#B73B7B', label='20 GHz')
    ax1.plot(dists, losses[4], color='#A5CFE0', label='30 GHz')
    ax1.set_title('FIGURE 4\nTypical trend of propagation along street canyons with low station\n' \
                  'height for frequency range from 2 to 38 GHz')
    ax1.set_xlabel('Distance of travel from station 1 (m)')
    ax1.set_ylabel('Basic transmission loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_6() -> None:
    """
    Calculates values and displays FIGURE 6 from ITU-R P.1411-12.
    """
    import matplotlib.pyplot as plt
    f_GHz = 0.400
    dists = []
    loc_percents = [1,10,50,90,99]
    losses = [[],[],[],[],[]]
    for d_m in range(1, 2000+1, 1):
        dists.append(d_m)
        for i, p in enumerate(loc_percents):
            L, _ = SiteGeneralNearStreetLevel(f_GHz, d_m, p, EnvironmentB.SUBURBAN)
            losses[i].append(L)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,2000])
    ax1.set_xticks([*range(0, 2000+1, 200)])
    ax1.set_ylim([160,0])
    ax1.set_yticks([*range(0, 160+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='1%')
    ax1.plot(dists, losses[1], color='#E31E24', label='10%')
    ax1.plot(dists, losses[2], color='#1D8545', label='50%')
    ax1.plot(dists, losses[3], color='#B73B7B', label='90%')
    ax1.plot(dists, losses[4], color='#A5CFE0', label='99%')
    ax1.set_title('FIGURE 6\nCurves of basic transmission loss not exceeded for 1, 10, 50, 90 and ' \
                  '99% of locations\n(frequency = 400 MHz, suburban)')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Basic transmission loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def EXTRA_FIGURE_1() -> None:
    """
    Site-specific over roof-tops (urban area) figure. This figure is not part of the
    recommendation.
    """
    import matplotlib.pyplot as plt
    hr = 10 # Average height of buildings (m), with h2_m < hr_m. 
    h1 = 50 # Station 1 antenna height (m), with 4 <= h1_m <= 55.
    h2 = 1.5 # Station 2 antenna height (m), with 1 <= h2_m <= 3.
    length_percent = 95 # Length of the path covered by buildings (%)
    b = 25 # Average building separation (m), with 0 < b_m.
    w = 10 # Street width (m) at station 2's location, with 0 < w2_m.
    phi_deg = 90 # Street orientation with respect to the direct path (deg), with 0 <= phi_deg <= 90.
    env = EnvironmentD.METROPOLITAN_CENTRE
    max_dist = 1000 # Path length (m), with 20 <= d_m <= 5000.
    dists = []
    freqs = [2,5,10,20,26] # Operating frequency (GHz), with 0.8 <= f_GHz <= 26 in general, but with
                           # 2 <= f_GHz <= 16 for h1_m < hr_m and w2_m < 10.
    losses = [[],[],[],[],[]]
    for d_m in range(20, max_dist+1, 1):
        dists.append(d_m)
        for i, f_GHz in enumerate(freqs):
            l = d_m*length_percent/100.0
            L_NoS1 = SiteSpecificOverRoofTopsUrban(f_GHz=f_GHz, d_m=d_m, h1_m=h1, h2_m=h2, hr_m=hr,
                                                   l_m=l, b_m=b, w2_m=w, phi_deg=phi_deg, env=env)
            losses[i].append(L_NoS1)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0,1000])
    ax1.set_xticks([*range(0, max_dist+1, 50)])
    ax1.set_ylim([160,40])
    ax1.set_yticks([*range(40, 160+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='2 GHz')
    ax1.plot(dists, losses[1], color='#E31E24', label='5 GHz')
    ax1.plot(dists, losses[2], color='#1D8545', label='10 GHz')
    ax1.plot(dists, losses[3], color='#B73B7B', label='20 GHz')
    ax1.plot(dists, losses[4], color='#A5CFE0', label='26 GHz')
    ax1.set_title('EXTRA FIGURE\nSite-specific over roof-tops (urban area)')
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Basic transmission loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def EXTRA_FIGURE_2() -> None:
    """
    Site-specific over roof-tops (suburban area) figure. This figure is not part of the
    recommendation.
    """
    import matplotlib.pyplot as plt
    hr = 10 # Average height of buildings (m).
    h1 = hr+40 # Station 1 antenna height (m), with hr_m+1 <= h1_m <= hr_m+100.
    h2 = hr-8.5 # Station 2 antenna height (m), with hr_m-10 <= h2_m <= hr_m-4.
    w = 10 # Street width (m) at station 2's location, with 10 <= w2_m <= 25.
    phi_deg = 90 # Street orientation with respect to the direct path (deg), with 0 <= phi_deg <= 90 
    max_dist = 1000 # Path length (m), with 10 <= d_m <= 5000.
    dists = []
    freqs = [2,5,10,20,30] # Operating frequency (GHz), with 0.8 <= f_GHz <= 38.
    losses = [[],[],[],[],[]]
    for d_m in range(10, max_dist+1, 1):
        dists.append(d_m)
        for i, f_GHz in enumerate(freqs):
            L_NLoS1 = SiteSpecificOverRoofTopsSuburban(f_GHz=f_GHz, d_m=d_m, h1_m=h1, h2_m=h2,
                                                       hr_m=hr, w2_m=w, phi_deg=phi_deg)
            losses[i].append(L_NLoS1)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0, max_dist])
    ax1.set_xticks([*range(0, max_dist+1, 50)])
    ax1.set_ylim([160,40])
    ax1.set_yticks([*range(40, 160+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='2 GHz')
    ax1.plot(dists, losses[1], color='#E31E24', label='5 GHz')
    ax1.plot(dists, losses[2], color='#1D8545', label='10 GHz')
    ax1.plot(dists, losses[3], color='#B73B7B', label='20 GHz')
    ax1.plot(dists, losses[4], color='#A5CFE0', label='30 GHz')
    ax1.set_title('EXTRA FIGURE\nSite-specific over roof-tops (suburban area)')
    ax1.set_xlabel('Path length (m)')
    ax1.set_ylabel('Basic transmission loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def EXTRA_FIGURE_3() -> None:
    """
    Site-specific near street level (urban area) figure. This figure is not part of the
    recommendation.
    """
    import matplotlib.pyplot as plt
    x1 = 300 # Distance between Station 1 and the first street corner (m), with 0 < x1_m.
    x2 = 200 # Distance between the first street corner and the second street corner (m), with 0 <= x2_m.
    x3 = 100 # Distance between the second street corner and Station 2 (m), with 0 <= x3_m.
    h1 = 3.5 # Station 1 antenna height (m), with 1.5 <= h1_m <= 4.
    h2 = 1.5 # Station 2 antenna height (m), with 1.5 <= h2_m <= 4.
    hs = 0.75 # Effective road height (m), with 0 <= hs_m.
    max_dist = x1+x2+x3
    dists = []
    freqs = [0.5,1,2,3,4] # Operating frequency (GHz), with 0.430 <= f_GHz <= 4.860.
    losses = [[],[],[],[],[]]
    for d_m in range(1, max_dist+1, 1):
        dists.append(d_m)
        if d_m <= x1:
            x1_temp = d_m
            x2_temp = 0
            x3_temp = 0
        elif d_m <= x1+x2:
            x1_temp = x1
            x2_temp = d_m - x1
            x3_temp = 0
        else:
            x1_temp = x1
            x2_temp = x2
            x3_temp = d_m - x1 - x2
        for i, f_GHz in enumerate(freqs):
            L = SiteSpecificNearStreetLevelUrban(f_GHz=f_GHz, x1_m=x1_temp, x2_m=x2_temp,
                                                 x3_m=x3_temp, h1_m=h1, h2_m=h2, hs_m=hs)
            losses[i].append(L)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0, max_dist])
    ax1.set_xticks([*range(0, max_dist+1, 50)])
    ax1.set_ylim([160, 40])
    ax1.set_yticks([*range(40, 160+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='0.5 GHz')
    ax1.plot(dists, losses[1], color='#E31E24', label='1 GHz')
    ax1.plot(dists, losses[2], color='#1D8545', label='2 GHz')
    ax1.plot(dists, losses[3], color='#B73B7B', label='3 GHz')
    ax1.plot(dists, losses[4], color='#A5CFE0', label='4 GHz')
    ax1.set_title('EXTRA FIGURE\nSite-specific near street level (urban area)')
    ax1.set_xlabel('Road distance (m)')
    ax1.set_ylabel('Basic transmission loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def EXTRA_FIGURE_4() -> None:
    """
    Site-specific near street level (residential area) figure. This figure is not part of the
    recommendation.
    """
    import matplotlib.pyplot as plt
    f = 14 # Operating frequency (GHz), with 2 <= f_GHz <= 26.
    hTx = 4 # Transmitter antenna height (m), with 1.2 <= hTx_m <= 6.
    hRx = 1.5 # Receiver antenna height (m), with 1.2 <= hRx_m <= 6.
    hbTx = 10 # Height of nearest building from transmitter in receiver direction (m).
    hbRx = 8 # Height of nearest building from receiver in transmitter direction (m).
    a = 25 # Distance between transmitter and nearest building from transmitter (m).
    c = 25 # Distance between receiver and nearest building from receiver (m).
    m = 9  # Average building height of the buildings with less than 3 stories (m).
    n = 200 # Building density (buildings/km2).
    x1 = 200
    x2 = 800
    max_road_dist = x1+x2
    dists = []
    losses = [[],[],[],[]]
    for road_dist in range(x1, max_road_dist+1, 1):
        # one 90 deg street corner at 200 meters from station1
        x2_temp = road_dist-x1
        d = sqrt((x1*x1)+(x2_temp*x2_temp)) # Distance between the two terminals (m), with 0 <= d_m <= 1000.
        dists.append(road_dist)
        thetaList = [90]
        x1List = [x1]
        x2List = [x2_temp]
        b = d-a-c # Distance between nearest buildings from transmitter and receiver (m).

        L, Lr, Lb, Lv = SiteSpecificNearStreetLevelResidential(f_GHz=f, d_m=d, hTx_m=hTx,
                            hRx_m=hRx, hbTx_m=hbTx, hbRx_m=hbRx, a_m=a, b_m=b, c_m=c, m_m=m,
                            n_bldgkm2=n, thetaList_deg=thetaList, x1List_m=x1List, x2List_m=x2List)
        losses[0].append(L)
        losses[1].append(Lr)
        losses[2].append(Lb)
        losses[3].append(Lv)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0, max_road_dist])
    ax1.set_xticks([*range(0, max_road_dist+1, 50)])
    ax1.set_ylim([240, 80])
    ax1.set_yticks([*range(80, 240+1, 20)])
    ax1.plot(dists, losses[0], color='#008DD2', label='L')
    ax1.plot(dists, losses[1], color='#E31E24', label='Lr (along road)')
    ax1.plot(dists, losses[2], color='#1D8545', label='Lb (between houses)')
    ax1.plot(dists, losses[3], color='#B73B7B', label='Lv (over roofs)')
    ax1.set_title('EXTRA FIGURE\nSite-specific near street level (residential area)\n' \
                  'with one 90 degrees street corner at 200m from station 1')
    ax1.set_xlabel('Road distance (m)')
    ax1.set_ylabel('Path loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()
