"""Implementation of ITU-R P.618-14 (partial)
"""

from math import sin, cos, radians, sqrt, log10, log, atan2, exp, degrees
from typing import Union
from numpy import nan
from numba import jit
from . import itur_p453
from . import itur_p676
from . import itur_p837
from . import itur_p838
from . import itur_p839
from . import itur_p840
from . import itur_p1511


__all__ = ['RainAttenuation',
           'ScintillationFading',
           'GaseousAttenuation',
           'CloudsAttenuation',
           'TotalAtmosphericAttenuation',
           'MeanRadiatingTemperature',
           'SkyNoiseTemperature',
           'HydrometeorCrossPolDiscrimination']


@jit(nopython=True)
def RainAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                    polTilt_deg: float, hs_km: Union[float, None]=None) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 2.2.1.1
    Estimates the attenuation due to rain exceeded for p% of an average year (dB).
    
    Args:
        f_GHz (float): Frequency of interest (GHz), with f_GHz <= 55.
        theta_deg (float): Elevation angle of the slant propagation path (degrees).
        p (float): Time percentage, with 0.001 <= p <= 5.
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
        polTilt_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
            for circular polarization.
        hs_km (float): Height above mean sea level of the earth station (km). When set to None, the
            height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        AR (float): Attenuation due to rain exceeded for p% of an average year (dB).
    """
    R001 = itur_p837.RainfallRate001(lat=lat, lon=lon)
    return _RainAttenuation(R001, f_GHz, theta_deg, p, lat, lon, polTilt_deg, hs_km)


@jit(nopython=True)
def _RainAttenuation(R001: float, f_GHz: float, theta_deg: float, p: float,
                     lat: float, lon: float, polTilt_deg: float,
                     hs_km: Union[float, None]=None) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 2.2.1.1
    Estimates the attenuation due to rain exceeded for p% of an average year (dB).
    
    Args:
        R001 (float): Point rainfall rate for the location for 0.01% of an average year (mm/h).
        f_GHz (float): Frequency of interest (GHz), with f_GHz <= 55.
        theta_deg (float): Elevation angle of the slant propagation path (degrees).
        p (float): Time percentage, with 0.001 <= p <= 5.
        lat (float): Latitude (degrees), with -90 <= lat <= 90.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
        polTilt_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
            for circular polarization.
        hs_km (float): Height above mean sea level of the earth station (km). When set to None, the
            height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        AR (float): Attenuation due to rain exceeded for p% of an average year (dB).
    """
    Re = 8500
    if hs_km is None:
        hs_km = itur_p1511.TopographicHeightAMSL(lat, lon)/1000
    hs = hs_km

    # Step 1
    hR = itur_p839.MeanAnnualRainHeight(lat=lat, lon=lon)

    # Step 2
    if (hR-hs) <= 0:
        return 0
    sin_t = sin(radians(theta_deg))
    if theta_deg >= 5:
        Ls = (hR-hs)/sin_t
    else:
        Ls = (2*(hR-hs))/(sqrt((sin_t*sin_t)+(2*(hR-hs)/Re))+sin_t)

    # Step 3
    cos_t = cos(radians(theta_deg))
    LG = Ls*cos_t

    # Step 4
    if R001 == 0:
        return 0
    
    # Step 5
    k, alpha = itur_p838.Coefficients(f_GHz=f_GHz, pathElevAngle_deg=theta_deg,
                                      polTiltAngle_deg=polTilt_deg)
    yR = k*pow(R001, alpha)

    # Step 6
    r001 = 1/(1+(0.78*sqrt(LG*yR/f_GHz))-(0.38*(1-exp(-2*LG))))

    # Step 7
    sigma_deg = degrees(atan2(hR-hs, LG*r001))
    if sigma_deg > theta_deg:
        LR = LG*r001/cos_t
    else:
        LR = (hR-hs)/sin_t
    abs_lat = abs(lat)
    if abs_lat < 36:
        X = 36-abs_lat
    else:
        X = 0
    a = 1-exp(-(theta_deg/(1+X)))
    b = sqrt(LR*yR)/(f_GHz*f_GHz)
    v001 = 1/(1+(sqrt(sin_t)*((31*a*b)-0.45)))

    # Step 8
    LE = LR*v001

    # Step 9
    A001 = yR*LE

    # Step 10
    if p >= 1 or abs_lat >= 36:
        Beta = 0
    elif p < 1 and abs_lat < 36 and theta_deg >= 25:
        Beta = -0.005*(abs_lat-36)
    else:
        Beta = (-0.005*(abs_lat-36)) + 1.8 - (4.25*sin_t)

    Ap = A001*pow(p/0.01, -(0.655 + 0.033*log(p) - 0.045*log(A001) - Beta*(1-p)*sin_t))
    return Ap


@jit(nopython=True)
def ScintillationFading(f_GHz: float, theta0_deg: float, p: float, lat: float, lon: float,
                        D: float, eff: float=0.5) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 2.4.1
    Calculates the tropospheric scintillation fading, exceeded for p% of the time (dB).

    Args:
        f_GHz (float): Frequency (GHz), with 4 <= f_GHz <= 55.
        theta0_deg (float): Free-space elevation angle (degrees), with 5 <= theta0_deg <= 90.
        p (float): Time percentage, with 0.01 < p <= 50.
        lat (float): Latitude (degrees), with -90 <= lat <= 90. Latitude and longitude values are
            used to obtain Nwet, the median value of the wet term of the surface refractivity
            exceeded for the average year, from the digital maps in Recommendation ITU-R P.453.
        lon (float): Longitude (degrees), with -180 <= lon <= 180.
        D (float): Physical diameter of the earth-station antenna (meters).
        eff (float): Antenna efficiency; if unknown, 0.5 is a conservative estimate.
    
    Returns:
        AS (float): Tropospheric scintillation fading, exceeded for p% of the time (dB).
    """
    Nwet = itur_p453.MedianAnnualNwet(lat, lon)
    sigma_ref = 3.6E-3 + (1E-4*Nwet)
    theta0_rad = radians(theta0_deg)
    sin_theta0 = sin(theta0_rad)
    L = 2000/(sqrt((sin_theta0*sin_theta0)+2.35E-4)+sin_theta0)
    Deff = sqrt(eff)*D
    x = 1.22*Deff*Deff*(f_GHz/L)
    sqrt_arg = (3.86*pow((x*x)+1,11/12)*sin((11/6)*atan2(1,x)))-(7.08*pow(x,5/6))
    if sqrt_arg < 0:
        return 0 
    g_x = sqrt(sqrt_arg)
    sigma = sigma_ref*pow(f_GHz,7/12)*g_x/pow(sin_theta0,1.2)
    log_p = log10(p)
    ap = (-0.061*log_p*log_p*log_p) + (0.072*log_p*log_p) - (1.71*log_p) + 3.0
    Ap = ap*sigma
    return Ap


def GaseousAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                       hs_km: Union[float, None]=None) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 2.5
    Gets the gaseous attenuation due to water vapour and oxygen for a fixed probability (dB), as
    estimated by Recommendation ITU-R P.676.

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 350.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        hs_km (float|None): Height of the desired location (km above mean sea level). When set to
            None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        AG (float): The gaseous attenuation due to water vapour and oxygen for a fixed
            probability (dB).
    """
    if hs_km is not None:
        h_mamsl = hs_km*1000
    else:
        h_mamsl = None
    # from Annex 2 of P.676, using annual statistics
    AG = itur_p676.SlantPathStatGaseousAttenuation(f_GHz, theta_deg, p, lat, lon, None, h_mamsl)
    return AG


def CloudAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 2.5
    Gets the attenuation due to clouds for a fixed probability (dB), as estimated by Recommendation
    ITU-R P.840.

    Args:
        f_GHz (float): Frequency of interest (GHz), with 1 <= f_GHz <= 200.
        theta_deg (float): Elevation angle (deg).
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.

    Returns:
        AC (float): The attenuation due to clouds for a fixed probability (dB).
    """
    # using annual statistics
    AC = itur_p840.StatisticalCloudAttenuation(f_GHz, theta_deg, p, lat, lon, None)
    return AC


def TotalAtmosphericAttenuation(f_GHz: float, theta_deg: float, p: float, lat: float, lon: float,
                                polTilt_deg: float, D: float, eff: float=0.5,
                                hs_km: Union[float, None]=None, excludeScintillation: bool=False
                                ) -> tuple[float, float, float, float, float]:
    """
    ITU-R P.618-14, Annex 1, Section 2.5
    Estimation of total attenuation due to multiple sources of simultaneously occurring atmospheric
    attenuation (dB). Total attenuation represents the combined effect of rain, gas, clouds and
    scintillation.
    
    Args:
        f_GHz (float): Frequency (GHz), with f_GHz <= 55.
        theta_deg (float): Elevation angle (deg), with 5 <= theta_deg <= 90.
        p (float): The probability the attenuation is exceeded (i.e. the CCDF), in %, with 
            0.001 <= p <= 50.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        polTilt_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
            for circular polarization.
        D (float): Physical diameter of the earth-station antenna (meters).
        eff (float): Antenna efficiency; if unknown, 0.5 is a conservative estimate.
        hs_km (float|None): Height of the desired location (km above mean sea level). When set to
            None, the height is automatically obtained from Recommendation ITU-R P.1511.
        excludeScintillation (bool): Whether to exclude tropospheric scintillation from the total
            attenuation. Relevant in the context of sky noise temperature calculation. When set to
            True, parameters D and eff are unused.
    
    Returns:
        AT (float): Estimation of total attenuation (dB) due to multiple sources of simultaneously
            occurring atmospheric attenuation (rain, gas, clouds and scintillation).
        AR (float): Attenuation due to rain for a fixed probability (dB).
        AC (float): Attenuation due to clouds for a fixed probability (dB), as estimated by
            Recommendation ITU-R P.840.
        AG (float): Gaseous attenuation due to water vapour and oxygen for a fixed probability (dB),
            as estimated by Recommendation ITU-R P.676.
        AS (float): Attenuation due to tropospheric scintillation for a fixed probability (dB).
    """
    AG = GaseousAttenuation(f_GHz, theta_deg, max(p, 5.0), lat, lon, hs_km)
    AC = CloudAttenuation(f_GHz, theta_deg, max(p, 5.0), lat, lon)
    if excludeScintillation == False:
        AS = ScintillationFading(f_GHz, theta_deg, p, lat, lon, D, eff)
    else:
        AS = 0

    if p <= 5:
        AR = RainAttenuation(f_GHz, theta_deg, p, lat, lon, polTilt_deg, hs_km)
        AT = AG + sqrt(((AR+AC)*(AR+AC))+(AS*AS))
    else:
        AR = 0
        AT = AG + sqrt((AC*AC)+(AS*AS))

    return (AT, AR, AC, AG, AS)


def MeanRadiatingTemperature(Ts_K: float) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 3
    Calculates the mean radiating temperature (K) from the surface temperature.

    Args:
        Ts_K (float): The surface temperature (K).

    Returns:
        Tmr (float): The mean radiating temperature (K).
    """
    Tmr = 37.34 + (0.81*Ts_K)
    return Tmr


def SkyNoiseTemperature(AT_dB: float, Tmr_K: float=275) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 3
    Gets the sky noise temperature at a ground station antenna (K).

    Args:
        AT_dB (float): The total atmospheric attenuation excluding scintillation fading (dB).
        Tmr_K (float): The mean radiating temperature (K).

    Returns:
        Tsky (float): The sky noise temperature at a ground station antenna (K).
    """
    pwA = pow(10, -AT_dB/10)
    Tsky = (Tmr_K*(1-pwA))+(2.7*pwA)
    return Tsky


@jit(nopython=True)
def HydrometeorCrossPolDiscrimination(f_GHz: float, theta_deg: float, p: float,
                                      polTilt_deg: float, AR_dB: float) -> float:
    """
    ITU-R P.618-14, Annex 1, Section 4.1
    Gets the hydrometeor-induced cross-polarization discrimination (dB) not exceeded for p% of the
    time.

    Args:
        f_GHz (float): Frequency (GHz), with 6 <= f_GHz <= 55.
        theta_deg (float): Path elevation angle (deg), with theta_deg <= 60.
        p (float): Time percentage for which the cross-polarization discrimination is not exceeded,
            in %. Must be set to one of the four following values: 1, 0.1, 0.01 and 0.001.
        polTilt_deg (float): Tilt angle of the linearly polarized electric field vector with respect
            to the horizontal (deg). For circular polarization use 45°.
        AR_dB (float): Rain attenuation (dB) exceeded for the required percentage of time, p, for
            the path in question, commonly called co-polar attenuation (CPA). May be calculated
            using RainAttenuation() from this module.
    
    Returns:
        XPDp (float): The hydrometeor-induced cross-polarization discrimination (dB) not exceeded
            for p% of the time.
    """
    tau = polTilt_deg

    # step 1
    if f_GHz >= 6 and f_GHz < 9:
        Cf = 60*log10(f_GHz) - 28.3
    elif f_GHz >= 9 and f_GHz < 36:
        Cf = 26*log10(f_GHz) + 4.1
    elif f_GHz >= 36 and f_GHz <= 55:
        Cf = 35.9*log10(f_GHz) - 11.3
    else:
        return nan
    
    # step 2
    if f_GHz >= 6 and f_GHz < 9:
        V = 30.8*pow(f_GHz, -0.21)
    elif f_GHz >= 9 and f_GHz < 20:
        V = 12.8*pow(f_GHz, 0.19)
    elif f_GHz >= 20 and f_GHz < 40:
        V = 22.6
    elif f_GHz >= 40 and f_GHz <= 55:
        V = 13.0*pow(f_GHz, 0.15)
    else:
        return nan
    CA = V*log10(AR_dB)

    # step 3
    Ctau = -10*log10(1-(0.484*(1+cos(radians(4*tau)))))

    # step 4
    Ctheta = -40*log10(cos(radians(theta_deg)))

    # step 5
    if p == 1:
        sigma = 0
    elif p == 0.1:
        sigma = 5
    elif p == 0.01:
        sigma = 10
    elif p == 0.001:
        sigma = 15
    else:
        return nan
    Csigma = 0.0053*sigma*sigma

    # step 6
    XPDrain = Cf - CA + Ctau + Ctheta + Csigma

    # step 7
    Cice = XPDrain*(0.3+(0.1*log10(p)))/2

    # step 8
    XPDp = XPDrain - Cice

    return XPDp