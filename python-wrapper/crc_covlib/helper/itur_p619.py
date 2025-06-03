# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Partial implementation of ITU-R P.619-5
"""

from math import pi, cos, sin, tan, radians, degrees, sqrt, atan2, log10, exp, acos
from typing import Union
import enum
from . import itur_p676
from .itur_p835 import ReferenceAtmosphere, MAGRA
from . import itur_p835
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['CalculationStatus',
           'FreeSpaceBasicTransmissionLoss',
           'SpaceToEarthGaseousAttenuation',
           'EarthToSpaceGaseousAttenuation',
           'BeamSpreadingLoss',
           'TroposphericScintillationAttenuation',
           'StraightLineEarthSpacePath',
           'FreeSpaceToApparentElevationAngle',
           'ApparentToFreeSpaceElevationAngle',
           'RayHeights',
           'FIGURE_3',
           'FIGURE_4',
           'FIGURE_8',
           'FIGURE_9']


class CalculationStatus(enum.Enum):
    COMPLETED                 = 1
    OUTSIDE_VICTIMS_BEAMWIDTH = 2
    DUCTING_OR_NO_LOS         = 3


def FreeSpaceBasicTransmissionLoss(f_GHz: float, d_km: float) -> float:
    """
    ITU-R P.619-5, Annex 1, Section 2.1
    Calculates the basic transmission loss assuming the complete radio path is in a vacuum with no
    obstruction, in dB.

    Args:
        f_GHz (float): frequency, in GHz.
        d_km (float): path length, in km.

    Returns:
        Lbfs (float): The free space basic transmission loss, in dB.
    """
    Lbfs = 92.45 + 20*log10(f_GHz*d_km)
    return Lbfs


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def BeamSpreadingLoss(theta0_deg: float, h_km: float) -> float:
    """
    ITU-R P.619-5, Annex 1, Section 2.4.2
    The signal loss due to beam spreading for a wave propagating through the total atmosphere in
    the Earth-space and space-Earth (dB). This effect is insignificant for elevation angles above
    5 degrees.
    
    Args:
        theta0_deg (float): Free-space elevation angle (deg), with theta0_deg < 10.
        h_km (float): Altitude of the lower point above sea level (km), 0 <= h_km <= 5.

    Returns:
        Abs (float): The signal loss due to beam spreading for a wave propagating through the total
        atmosphere in the Earth-space and space-Earth (dB).
    """
    t0 = theta0_deg
    h = h_km
    x = 0.5411 + (0.07446*t0) + (h*(0.06272+(0.0276*t0))) + (h*h*0.008288)
    y = 1.728 + (0.5411*t0) + (0.03723*t0*t0) + (h*(0.1815+(0.06272*t0)+(0.0138*t0*t0))) + (h*h*(0.01727+(0.008288*t0)))
    B = 1 - (x/(y*y))

    # ITU-R P.834-9 uses minus sign in eq. (15) instead of ±
    Abs = -10*log10(B)

    return Abs


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def StraightLineEarthSpacePath(He_km: float, lat_e: float, lon_e: float, 
                               Hs_km: float, lat_s: float, lon_s: float, 
                               ) -> tuple[float, float, float]:
    """
    ITU-R P.619-5, Attachment A to Annex 1
    Calculates the distance, elevation angle and azimuth of a space station as viewed from an
    earth-based station. It is based on spherical Earth geometry, and ignores the effect of 
    atmospheric refraction.

    Args:
        He_km (float): Altitude of earth-based station, in km above sea level.
        lat_e (float): Latitude of earth-based station, in degrees, with -90 <= lat_e <= 90.
        lon_e (float): Longitude of earth-based station, in degrees with -180 <= lon_e <= 180.
        Hs_km (float): Altitude of space station, in km above sea level.
        lat_s (float): Latitude of sub-satellite point, in degrees, with -90 <= lat_s <= 90.
        lon_s (float): Longitude of sub-satellite point, in degrees with -180 <= lon_s <= 180.
        
    Returns:
        Dts (float): Straight-line distance between the earth-based station and the space station,
            in km.
        theta0_deg (float): Elevation angle of the straight line from the earth-based station to 
            the space station, in degrees. It is the elevation angle of the ray at the earth-based
            station which would exist in the absence of tropospheric refraction, sometimes referred
            to as the free-space elevation angle. 0°=horizon, +90°=zenith, -90°=nadir.
        azm_deg (float): Azimuth of the space station as viewed from an earth-based station, in 
            degrees from true North. 0°=North, 90°=East, 180°=South, 270°=West. The azimuth is 
            indeterminate if the elevation angle represents a vertical path.
    """
    # Calculate delta, the difference in longitude between the sub-satellite point and the
    # earth-based station, limited to less than half a circle, positive when the space station is
    # to the east of the earth-based station, in degrees.
    delta_deg = lon_s-lon_e
    if delta_deg > 180:
        delta_deg -= 360
    elif delta_deg < -180:
        delta_deg += 360

    phi_t = lat_e
    phi_s = lat_s

    # Step 1
    Re = 6371 # average Earth radius (km)
    Rs = Re + Hs_km
    Rt = Re + He_km

    # Step 2
    delta_rad = radians(delta_deg)
    cos_d     = cos(delta_rad)
    sin_d     = sin(delta_rad)
    cos_phi_s = cos(radians(phi_s))
    cos_phi_t = cos(radians(phi_t))
    sin_phi_s = sin(radians(phi_s))
    sin_phi_t = sin(radians(phi_t))
    X1 = Rs*cos_phi_s*cos_d
    Y1 = Rs*cos_phi_s*sin_d
    Z1 = Rs*sin_phi_s

    # Step 3
    X2 = X1*sin_phi_t - Z1*cos_phi_t
    Y2 = Y1
    Z2 = Z1*sin_phi_t + X1*cos_phi_t - Rt

    # Step 4
    Dts = sqrt(X2*X2 + Y2*Y2 + Z2*Z2)

    # Step 5
    Gts = sqrt(X2*X2 + Y2*Y2)

    # Step 6
    theta0_rad = atan2(Gts, Z2)
    theta0_deg = degrees(theta0_rad)

    # Note: the following line is not in the recommendation, however it seems needed to have the
    #       horizon at 0 deg, zenith at 90 deg and nadir at -90 deg.
    theta0_deg = -theta0_deg + 90

    # Step 7
    psi_rad = atan2(X2, Y2) 

    # Step 8
    #psi_deg = 180 - degrees(psi_rad)
    # Note: Slight change from what is in the recommendation so to have azimuth from true North.
    psi_deg = (degrees(psi_rad) + 450) % 360

    return (Dts, theta0_deg, psi_deg)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def FreeSpaceToApparentElevationAngle(He_km: float, theta0_deg: float) -> float:
    """
    ITU-R P.619-5, Attachment B to Annex 1
    Converts free-space elevation angle (at Earth-based station) to apparent elevation angle (at
    Earth-based station).

    Args:
        He_km (float): Altitude of earth-based station, in km above sea level, with 0 <= He_km <= 3.
        theta0_deg (float): Free-space elevation angle, in degrees, with -1 <= theta_deg <= 10.
            It is he elevation angle calculated without taking atmospheric refraction into account.

    Returns:
        theta_deg (float): Apparent or actual elevation, in degrees. It is the elevation angle
            calculated taking atmospheric refraction into account. This is the optimum elevation
            angle for a high-gain antenna at the earth-based station intended to provide a link to
            the space station.
    """
    Tfs1 = 1.728 + (0.5411*theta0_deg) + (0.03723*theta0_deg*theta0_deg)
    Tfs2 = 0.1815 + (0.06272*theta0_deg) + (0.01380*theta0_deg*theta0_deg)
    Tfs3 = 0.01727 + (0.008288*theta0_deg)
    tau_fs = 1/(Tfs1 + (He_km*Tfs2) + (He_km*He_km*Tfs3))
    theta_deg = theta0_deg + tau_fs
    return theta_deg


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def ApparentToFreeSpaceElevationAngle(He_km: float, theta_deg: float) -> float:
    """
    ITU-R P.619-5, Attachment B to Annex 1
    Converts apparent elevation angle (at Earth-based station) to free-space elevation angle (at
    Earth-based station).
    
    Args:
        He_km (float): Altitude of earth-based station, in km above sea level, with 0 <= He_km <= 3.
        theta_deg (float): Apparent or actual elevation angle, in degrees. It is the elevation angle
            calculated taking atmospheric refraction into account. This is the optimum elevation
            angle for a high-gain antenna at the earth-based station intended to provide a link to
            the space station.
    
    Returns:
        theta0_deg (float): Free-space elevation angle, in degrees. It is he elevation angle 
            calculated without taking atmospheric refraction into account.
    """
    T1 = 1.314 + (0.6437*theta_deg) + (0.02869*theta_deg*theta_deg)
    T2 = 0.2305 + (0.09428*theta_deg) + (0.01096*theta_deg*theta_deg)
    T3 = 0.008583
    tau = 1/(T1 + (He_km*T2) + (He_km*He_km*T3))
    theta0_deg = theta_deg - tau
    return theta0_deg


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def SpaceToEarthGaseousAttenuation(f_GHz: float, He_km: float, Hs_km: float,
                                   phi_e_deg: Union[float, None], phi_s_deg: float,
                                   delta_phi_e_deg: Union[float, None],
                                   refAtm: ReferenceAtmosphere=MAGRA, rho0_gm3: float=7.5
                                   ) -> tuple[float, CalculationStatus]:
    """
    ITU-R P.619-5, Attachment C to Annex 1 (C.4)
    Calculates the attenuation due to atmospheric gases along a space-Earth propagation path
    (descending ray). For a space-Earth propagation path, the antenna of the space-based station
    is the transmitting antenna.

    Args:
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000. Gaseous attenuation can be ignored
            below 1 GHz.
        He_km (float): Height of Earth station (km above mean sea level), with 0 <= He_km < Hs_km.
        Hs_km (float): Height of space-based station (km above mean sea level), with 0 <= He_km < Hs_km.
        phi_e_deg (float|None): Elevation angle of the main beam of the earth based station (deg).
            If set to None, assumes the incident elevation angle is within the half-power beamwidth
            of the earth station. 0°=horizon, +90°=zenith, -90°=nadir.
        phi_s_deg (float): Elevation angle of the main beam of the space-based station antenna (deg).
            0°=horizon, +90°=zenith, -90°=nadir.
        delta_phi_e_deg (float|None): Half-power beamwidth of the Earth based station (deg).
            If set to None, assumes the incident elevation angle is within the half-power beamwidth
            of the earth station.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).
        
    Returns:
        Ag (float): The total gaseous attenuation along the space-Earth slant path (dB). Returns 0
            (zero) when the attenuation could not be calculated (see status value then for more 
            details).
        status (crc_covlib.helper.itur_p619.CalculationStatus): COMPLETED if Ag could be calculated
            successfully. Otherwise the status value indicates the reason for stopping calculations.
    """
    Re = 6371
    n_s = itur_p676.RefractiveIndex(Hs_km, refAtm, rho0_gm3)
    n_e = itur_p676.RefractiveIndex(He_km, refAtm, rho0_gm3)
    phi_s_rad = radians(phi_s_deg)

    if phi_s_deg != -90 and phi_e_deg is not None and delta_phi_e_deg is not None:
        # Step 1
        # Note: The space station main beam is assumed to be directly pointing at the earth station
        #       (taking the effect of atmosphere into account).
        phi_ce_rad = acos(((Re+Hs_km)*n_s)/((Re+He_km)*n_e)*cos(phi_s_rad))
        phi_ce_deg = degrees(phi_ce_rad) # incident elevation angle (i.e. the elevation angle, at the 
                                         # earth station, from the incoming beam from the space station).

        # Step 2
        # Cheks if the incoming beam falls within the vertical half-power beamwidth of the earth station.
        is_within_beamwidth = (abs(phi_ce_deg-phi_e_deg) <= (delta_phi_e_deg/2))
        if is_within_beamwidth is False:
            return (0, CalculationStatus.OUTSIDE_VICTIMS_BEAMWIDTH)
    
    # Step 3, 4, 5, 6
    # also see Section 2.2.1 of Annex 1 from ITU-R P.676-13
    Ag = 0
    h_lower = He_km
    h_upper = Hs_km
    (i_lower, i_upper) = itur_p676._LayerIndices(h_lower, h_upper)
    for i in range(i_lower, i_upper, 1):
        h_i_bot = itur_p676._LayerBottomHeight(i, i_lower, i_upper, h_lower, h_upper)
        delta_i = itur_p676._LayerThickness(i, i_lower, i_upper, h_lower, h_upper)
        h_i_mid = h_i_bot+(delta_i/2)
        n_i = itur_p676.RefractiveIndex(h_i_mid, refAtm, rho0_gm3)

        if phi_s_deg != -90:
            # Step 3
            if cos(phi_s_rad) >= ((Re+h_i_bot)*n_i)/((Re+Hs_km)*n_s):
                # see ITU-R P.676-13, Section 2.2.3 of Annex 1 regarding Earth surface intercepts
                return (0, CalculationStatus.DUCTING_OR_NO_LOS)
        
        r_i = Re + h_i_bot
        r_ip1 = r_i + delta_i
        r_s = Re + Hs_km

        # Step 4
        x = (n_s/n_i)*r_s*cos(phi_s_rad)
        l_is = sqrt((r_ip1*r_ip1)-(x*x)) - sqrt((r_i*r_i)-(x*x))

        # Step 5
        y_i = itur_p676.GaseousAttenuation(f_GHz=f_GHz,
                                           # examples from CG-3M3J-13-ValEx-Rev8.1.1.xlsx use dry pressure
                                           P_hPa=itur_p835.DryPressure(h_i_mid, refAtm),
                                           T_K=itur_p835.Temperature(h_i_mid, refAtm),
                                           rho_gm3=itur_p835.WaterVapourDensity(h_i_mid, refAtm, rho0_gm3))

        # Step 6
        Ag += l_is*y_i

    return (Ag, CalculationStatus.COMPLETED)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def EarthToSpaceGaseousAttenuation(f_GHz: float, He_km: float, Hs_km: float,
                                   phi_e_deg: float, phi_s_deg: Union[float, None],
                                   delta_phi_s_deg: Union[float, None],
                                   refAtm: ReferenceAtmosphere=MAGRA, rho0_gm3: float=7.5
                                   ) -> tuple[float, CalculationStatus]:
    """
    ITU-R P.619-5, Attachment C to Annex 1 (C.5)
    Calculates the attenuation due to atmospheric gases along an Earth-space propagation paths
    (ascending ray). For an Earth-space propagation path, the antenna of the earth based station
    is the transmitting antenna.

    Args:
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000. Gaseous attenuation can be ignored
            below 1 GHz.
        He_km (float): Height of Earth station (km above mean sea level), with 0 <= He_km < Hs_km.
        Hs_km (float): Height of space-based station (km above mean sea level), with 0 <= He_km < Hs_km.
        phi_e_deg (float): Elevation angle of the main beam of the earth based station (deg).
            0°=horizon, +90°=zenith, -90°=nadir.
        phi_s_deg (float|None): Elevation angle of the main beam of the space-based station antenna (deg).
            If set to None, assumes the calculated elevation angle at the space station is within its
            half-power beamwidth. 0°=horizon, +90°=zenith, -90°=nadir.
        delta_phi_s_deg (float|None): Half-power beamwidth of the space-based station antenna (deg).
            If set to None, assumes the calculated elevation angle at the space station is within its
            half-power beamwidth.
        refAtm (crc_covlib.helper.itur_p835.ReferenceAtmosphere): A reference atmosphere from
            ITU-R P.835-6.
        rho0_gm3 (float): Ground-level water vapour density (g/m3). Only applies when refAtm is
            set to MEAN_ANNUAL_GLOBAL (MAGRA).

    Returns:
        Ag (float): The total gaseous attenuation along the Earth-space slant path (dB). Returns 0
            (zero) when the attenuation could not be calculated (see status value then for more 
            details).
        status (crc_covlib.helper.itur_p619.CalculationStatus): COMPLETED if Ag could be calculated
            successfully. Otherwise the status value indicates the reason for stopping calculations.
    """
    if phi_e_deg >= 0:
        # Case 1: Non-negative elevation angles (phi_e >= 0)
        Re = 6371
        phi_e_rad = radians(phi_e_deg)
        n_s = itur_p676.RefractiveIndex(Hs_km, refAtm, rho0_gm3)
        n_e = itur_p676.RefractiveIndex(He_km, refAtm, rho0_gm3)

        # Step 1
        # Note: The earth station main beam is assumed to be directly pointing at the space station
        #       (taking the effect of atmosphere into account).
        phi_cs_rad = acos((((Re+He_km)*n_e)/((Re+Hs_km)*n_s))*cos(phi_e_rad))
        phi_cs_deg = degrees(phi_cs_rad) # incident elevation angle (i.e. the elevation angle, at the 
                                         # space station, from the incoming beam from the earth station).
            
        # Step 2
        # Cheks if the incoming beam falls within the vertical half-power beamwidth of the space station.
        if phi_e_deg != 90 and phi_s_deg is not None and delta_phi_s_deg is not None:
            is_within_beamwidth = (abs(phi_cs_deg-phi_s_deg) <= (delta_phi_s_deg/2))
            if is_within_beamwidth is False:
                return (0, CalculationStatus.OUTSIDE_VICTIMS_BEAMWIDTH)

        # Step 3, 4, 5, 6
        # also see Section 2.2.1 of Annex 1 from ITU-R P.676-13
        Ag = 0
        h_lower = He_km
        h_upper = Hs_km
        (i_lower, i_upper) = itur_p676._LayerIndices(h_lower, h_upper)
        for i in range(i_lower, i_upper, 1):
            h_i_bot = itur_p676._LayerBottomHeight(i, i_lower, i_upper, h_lower, h_upper)
            delta_i = itur_p676._LayerThickness(i, i_lower, i_upper, h_lower, h_upper)
            h_i_mid = h_i_bot+(delta_i/2)
            n_i = itur_p676.RefractiveIndex(h_i_mid, refAtm, rho0_gm3)

            if phi_e_deg != 90:
                # Step 3
                # Note: Seems like there is an error in the recommendation
                #       phi_s should not be used for the line-of-sight between antennas as it is
                #       pointing elev angle of the receive antenna at space. It may not point in
                #       the direction of the radio path.
                #
                # Could use phi_cs instead:
                # if cos(phi_cs_rad) >= ((Re+h_i_bot)*n_i)/((Re+Hs)*n_s):
                # Or, phi_e as proposed by P.B.:
                if cos(phi_e_rad) >= ((Re+h_i_bot)*n_i)/((Re+He_km)*n_e):
                    # see ITU-R P.676-13, Section 2.2.3 of Annex 1 regarding Earth surface intercepts
                    return (0, CalculationStatus.DUCTING_OR_NO_LOS)

            r_i = Re + h_i_bot
            r_ip1 = r_i + delta_i
            r_e = Re + He_km

            # Step 4
            x = (n_e/n_i)*r_e*cos(phi_e_rad)
            l_ie = sqrt((r_ip1*r_ip1)-(x*x)) - sqrt((r_i*r_i)-(x*x))

            # Step 5
            y_i = itur_p676.GaseousAttenuation(f_GHz=f_GHz,
                                            # examples from CG-3M3J-13-ValEx-Rev8.1.1.xlsx use dry pressure
                                            P_hPa=itur_p835.DryPressure(h_i_mid, refAtm),
                                            T_K=itur_p835.Temperature(h_i_mid, refAtm),
                                            rho_gm3=itur_p835.WaterVapourDensity(h_i_mid, refAtm, rho0_gm3))

            # Step 6
            Ag += l_ie*y_i

        return (Ag, CalculationStatus.COMPLETED)
    else:
        # Case 2: Negative elevation angles (phi_e < 0)
        Re = 6371
        phi_e_rad = radians(phi_e_deg)
        H0 = 0
        H1 = He_km
        Hmin = ((H1-H0)/2)+H0
        for _ in range (1, 24):
            parallelism_test = ( ((Re+Hmin)*itur_p676.RefractiveIndex(Hmin, refAtm, rho0_gm3)) -
                                 ((Re+He_km)*itur_p676.RefractiveIndex(He_km, refAtm, rho0_gm3)*cos(phi_e_rad)) )
            if parallelism_test > 0:
                H1 = Hmin
            else:
                H0 = Hmin
            Hmin = ((H1-H0)/2)+H0

        Ag1, status1 = EarthToSpaceGaseousAttenuation(f_GHz=f_GHz, He_km=Hmin, Hs_km=He_km, phi_e_deg=0, 
                                                      phi_s_deg=None, delta_phi_s_deg=None,
                                                      refAtm=refAtm, rho0_gm3=rho0_gm3)

        Ag2, status2 = EarthToSpaceGaseousAttenuation(f_GHz=f_GHz, He_km=Hmin, Hs_km=Hs_km, phi_e_deg=0,
                                                      phi_s_deg=phi_s_deg, delta_phi_s_deg=delta_phi_s_deg,
                                                      refAtm=refAtm, rho0_gm3=rho0_gm3)

        if status1 == CalculationStatus.COMPLETED and status2 == CalculationStatus.COMPLETED:
            return (Ag1+Ag2, CalculationStatus.COMPLETED)
        elif status1 != CalculationStatus.COMPLETED:
            return (0, status1)
        else:
            return (0, status2)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def TroposphericScintillationAttenuation(f_GHz: float, p: float, theta0_deg: float,
                                         Nwet: float, D: float, Ga: Union[float, None]=None,
                                         ) -> float:
    """
    ITU-R P.619-5, Attachment D to Annex 1
    Gets the tropospheric scintillation attenuation (applies to a single path between one
    transmitter and one receiver).

    Args:
        f_GHz (float): Frequency (GHz), with f_GHz <= 100.
        p (float): Time percentage, with 0.001 <= p <= 99.999.
        theta0_deg (float): Free-space elevation angle (degrees), with 5 <= theta0_deg <= 90.
        Nwet (float): The median value of the wet term of the surface refractivity exceeded for the
            average year. See crc_covlib.helper.itur_p453.MedianAnnualNwet(lat, lon) function.
        D (float): Physical diameter of the earth-station antenna, in meters.
        Ga (float|None): Antenna gain of the earth-based antenna, in the direction of the path (dBi). 
            If specified, D is unused.
        
    Returns:
        Ast (float): Tropospheric scintillation attenuation not exceeded for p percent time, in dB.
            Can be positive (attenuation) or negative (enhancement).
    """
    if Ga is not None:
        Deff = 0.3*(10**(0.05*Ga))/(pi*f_GHz)
    else:
        eta = 0.5
        Deff = sqrt(eta)*D

    # see ITU-R P.618-14, Annex 1, Section 2.4.1
    sigma_ref = 3.6E-3 + (1E-4*Nwet)
    theta0_rad = radians(theta0_deg)
    sin_theta0 = sin(theta0_rad)
    L = 2000/(sqrt((sin_theta0*sin_theta0)+2.35E-4)+sin_theta0)
    x = 1.22*Deff*Deff*(f_GHz/L)
    sqrt_arg = (3.86*pow((x*x)+1,11/12)*sin((11/6)*atan2(1,x)))-(7.08*pow(x,5/6))
    if sqrt_arg < 0:
        return 0 
    g_x = sqrt(sqrt_arg)
    sigma_st = sigma_ref*pow(f_GHz,7/12)*g_x/pow(sin_theta0,1.2)

    if p <= 50:
        log_p = log10(p)
        a_ste = 2.672 - (1.258*log_p) - (0.0835*log_p*log_p) - (0.0597*log_p*log_p*log_p)
        Ast = -sigma_st*a_ste
    else:
        log_q = log10(100-p)
        a_stf = 3.0 - (1.71*log_q) + (0.072*log_q*log_q) - (0.061*log_q*log_q*log_q)
        Ast = sigma_st*a_stf

    return Ast


def RayHeights(He_km: float, theta_deg: float, deltaDist_km: float=1
               ) -> tuple[list[float], list[float]]:
    """
    ITU-R P.619-5, Attachment E to Annex 1
    Provides a method for tracing a ray launched from an earth-based station, in order to test
    whether it encounters an obstruction. It can be used to compile a profile of ray height
    relative to sea level, which can then be compared with a terrain profile.
    
    Args:
        He_km (float): Altitude of earth-based station, in km above sea level.
        theta_deg (float): Apparent elevation angle at the earth-based station, in degrees. 
            0°=horizon, +90°=zenith, -90°=nadir.
        deltaDist_km (float): Increment in horizontal distance over curved Earth (km).

    Returns:
        Dc_list (list[float]): Horizontal distances over curved earth from the earth-based
            station, in km.
        Hr_list (list[float]): Ray heights over sea level, in km.
    """
    Ht = He_km
    Hr = Ht # ray altitude (km above sea level)
    Dc = 0 # horizontal distance over curved Earth (km)
    epsilon = radians(theta_deg) # ray elevation angle above local horizontal (radians)
    delta_d = deltaDist_km
    Re = 6371 # average Earth radius (km)

    Dc_list = []
    Hr_list = []

    if theta_deg <= 5:
        while Hr <= 10:
            Dc_list.append(Dc)
            Hr_list.append(Hr)
            delta_eps = delta_d * ((1/Re) - 4.28715E-5 * exp(-Hr/7.348))
            Hr = Hr + (delta_d*epsilon)
            epsilon = epsilon + delta_eps
            Dc = Dc + delta_d
    else:
        tan_theta = tan(radians(theta_deg))
        while Hr <= 10:
            Dc_list.append(Dc)
            Hr_list.append(Hr)
            Dc = Dc + delta_d
            
            # calculates d (in km), the distance from earth-based station
            # (these equations are not in the recommendation)
            alpha_rad = Dc/Re
            gamma_rad = (pi/2) + radians(theta_deg)
            beta_rad = pi - alpha_rad - gamma_rad
            d = sin(alpha_rad)*(Re+Ht)/sin(beta_rad) # from Law of Sines

            Hr = Ht + (d*tan_theta) + ((d*d)/(2*Re))

    return (Dc_list, Hr_list)


def FIGURE_3() -> None:
    import matplotlib.pyplot as plt
    f_GHz = 30
    He_km = 1
    Hs_km = 100
    refAtm = ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL
    elevAngles = []
    rho0s = [2.5, 7.5, 12.5]
    att = [[],[],[]]

    for theta_e_deg in range(-10, 90+1, 1):
        elevAngles.append(theta_e_deg)
        for i, rho0 in enumerate(rho0s):
            Ag_dB, status = EarthToSpaceGaseousAttenuation(f_GHz, He_km, Hs_km, theta_e_deg, None,
                                                           None, refAtm, rho0)
            if status == CalculationStatus.COMPLETED:
                att[i].append(Ag_dB)
            else:
                att[i].append(None)
            print('{:.1f}°, {:.1f} g/m3, {:.3f} dB, {}'.format(theta_e_deg, rho0, Ag_dB, status))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_yscale('log')
    ax1.set_xlim([-10, 90])
    ax1.set_xticks([*range(-10, 90+1, 10)])
    ax1.set_ylim([0.1, 100])
    ax1.plot(elevAngles, att[2], color='#0000FF', label='12.5 g/m^3')
    ax1.plot(elevAngles, att[1], color='#FF0000', label='7.5 g/m^3')
    ax1.plot(elevAngles, att[0], color='#000000', label='2.5 g/m^3')
    ax1.set_title('FIGURE 3\nAtmospheric attenuation vs. elevation angle along Earth-space propagation' \
                  'path (earth station\naltitude = 1 km, space station altitude = 100 km, frequency = 30 GHz)')
    ax1.set_xlabel('Apparent elevation angle (degrees)')
    ax1.set_ylabel('Atmospheric attenuation (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_4() -> None:
    import matplotlib.pyplot as plt
    f_GHz = 30
    He_km = 1
    Hs_km = 100
    refAtm = ReferenceAtmosphere.MEAN_ANNUAL_GLOBAL
    elevAngles = []
    rho0s = [2.5, 7.5, 12.5]
    att = [[],[],[]]

    for theta_s_deg in range(-90, 0+1, 1):
        elevAngles.append(theta_s_deg)
        for i, rho0 in enumerate(rho0s):
            Ag_dB, status = SpaceToEarthGaseousAttenuation(f_GHz, He_km, Hs_km, None, theta_s_deg,
                                                           None, refAtm, rho0)
            if status == CalculationStatus.COMPLETED:
                att[i].append(Ag_dB)
            else:
                att[i].append(None)
            print('{:.1f}°, {:.1f} g/m3, {:.3f} dB, {}'.format(theta_s_deg, rho0, Ag_dB, status))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_yscale('log')
    ax1.set_xlim([-90, 0])
    ax1.set_xticks([*range(-90, 0+1, 10)])
    ax1.set_ylim([0.1, 100])
    ax1.plot(elevAngles, att[2], color='#0000FF', label='12.5 g/m^3')
    ax1.plot(elevAngles, att[1], color='#FF0000', label='7.5 g/m^3')
    ax1.plot(elevAngles, att[0], color='#000000', label='2.5 g/m^3')
    ax1.set_title('FIGURE 4\nAtmospheric attenuation vs. elevation angle along space-Earth propagation' \
                  'path (space station\naltitude = 100 km, earth station altitude = 1 km, frequency = 30 GHz)')
    ax1.set_xlabel('Apparent elevation angle (degrees)')
    ax1.set_ylabel('Atmospheric attenuation (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_8() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    heights = [0,0.5,1,2,3,5]
    theta0s = []
    Abs = [[],[],[],[],[],[]]
    for theta0_deg in np.arange(0, 6.1,0.1):
        theta0s.append(theta0_deg)
        for i, h in enumerate(heights):
            Abs[i].append(BeamSpreadingLoss(theta0_deg, h))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xlim([0, 6])
    ax1.set_xticks([*np.arange(0, 6+0.01, 0.5)])
    ax1.set_ylim([0, 0.9])
    ax1.plot(theta0s, Abs[0], color='#000080', label='0 km')
    ax1.plot(theta0s, Abs[1], color='#FF8027', label='0.5 km')
    ax1.plot(theta0s, Abs[2], color='#808080', label='1 km')
    ax1.plot(theta0s, Abs[3], color='#FFC000', label='2 km')
    ax1.plot(theta0s, Abs[4], color='#00B0FF', label='3 km')
    ax1.plot(theta0s, Abs[5], color='#008000', label='5 km')
    ax1.set_title('FIGURE 8\nBeam spreading loss in both the Earth-to-space and space-to-Earth directions')
    ax1.set_xlabel('Free-space elevation angle (degrees)')
    ax1.set_ylabel('Beam spreading loss (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_9() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    f_GHz = 30
    Ga_dBi = 0
    Nwet = 42.5
    fs_elev_angles = [5,10,20,35,90]
    percentages = [*np.arange(0.01, 0.1, 0.01),
                   *np.arange(0.1, 1, 0.1),
                   *range(1, 10+1, 1)]
    Ast = [[],[],[],[],[],[],[],[],[],[]]
    
    for p in percentages:
        for i, theta0 in enumerate(fs_elev_angles):
            enh = TroposphericScintillationAttenuation(f_GHz, p, theta0, Nwet, 0, Ga_dBi)
            enh = abs(enh) # enhancements are negative sign, fades are positive sign
            Ast[i].append(enh)

    for p in percentages:
        for i, theta0 in enumerate(fs_elev_angles):
            fade = TroposphericScintillationAttenuation(f_GHz, 100-p, theta0, Nwet, 0, Ga_dBi)
            Ast[i+5].append(fade)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6.75)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([0.01, 10])
    ax1.set_xticks([0.01,0.1,1,10])
    ax1.set_ylim([0.1, 10])
    ax1.set_yticks([0.1,1,10])
    ax1.plot(percentages, Ast[0], color='#FF0000', linestyle='dotted', label='5%, enhanc.')
    ax1.plot(percentages, Ast[1], color='#0000FF', linestyle='dotted', label='10%, enhanc.')
    ax1.plot(percentages, Ast[2], color='#FF0000', linestyle='dotted', label='20%, enhanc.')
    ax1.plot(percentages, Ast[3], color='#0000FF', linestyle='dotted', label='35%, enhanc.')
    ax1.plot(percentages, Ast[4], color='#FF0000', linestyle='dotted', label='90%, enhanc.')
    ax1.plot(percentages, Ast[5], color='#FF0000', label='5%, fade')
    ax1.plot(percentages, Ast[6], color='#0000FF', label='10%, fade')
    ax1.plot(percentages, Ast[7], color='#FF0000', label='20%, fade')
    ax1.plot(percentages, Ast[8], color='#0000FF', label='35%, fade')
    ax1.plot(percentages, Ast[9], color='#FF0000', label='90%, fade')
    ax1.set_title('FIGURE 9\nTropospheric scintillation enhancements and fades plotted against free-space' \
                  'elevation angle\nfor Nwet = 42.5 at 30 GHz, antenna gain = 0 dBi')
    ax1.set_xlabel('Percentage time fades and enhancements exceeded')
    ax1.set_ylabel('Enhancement, fade (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def _FreeSpaceElevAngles(He_km: float, Hs_km: float, d_km: float) -> float:
    """
    Not part of the recommendation.
    Uses trigonometry to calculate free-space elevation angle at the Earth and space stations (deg).
    See https://en.wikipedia.org/wiki/Law_of_cosines

    Args:
        He_km (float): Height of the Earth-based station (km above mean sea level).
        Hs_km (float): Height of the space-based station (km above mean sea level).
        d_km (float): Straight-line distance between the Earth and space stations (km).

    Returns:
        (float): The free-space elevation angle at the Earth-based station (deg).
        (float): The free-space elevation angle at the space-based station (deg).
    """
    Re = 6371
    a = Re + Hs_km
    b = Re + He_km
    c = d_km
    alpha_rad = acos(((b*b)+(c*c)-(a*a))/(2*b*c))
    theta0e_deg = degrees(alpha_rad)-90
    beta_rad = acos(((a*a)+(c*c)-(b*b))/(2*a*c))
    theta0s_deg = -(90 - degrees(beta_rad))
    return theta0e_deg, theta0s_deg


def _StraightLineDistance(He_km: float, Hs_km: float, theta0e_deg: Union[float, None],
                          theta0s_deg: Union[float, None]) -> float:
    """
    Not part of the recommendation.
    Uses trigonometry to calculate the straight-line distance between the earth-based station and 
    the space station, in km.
    See https://en.wikipedia.org/wiki/Law_of_cosines

    Args:
        He_km (float): Height of the Earth-based station (km above mean sea level).
        Hs_km (float): Height of the space-based station (km above mean sea level).
        theta0e_deg (float|None): Free-space elevation angle (deg) at the Earth-based station. May
            be set to None but one of theta0e_deg and theta0s_deg needs to be specified.
        theta0s_deg (float|None): Free-space elevation angle (deg) at the space-based station. May
            be set to None but one of theta0e_deg and theta0s_deg needs to be specified.

    Returns:
        (float): The straight-line distance between the earth-based station and the space-based
            station, in km.
    """
    Re = 6371 # average Earth radius (km)
    a = Re + Hs_km
    b = Re + He_km

    if theta0e_deg is not None:
        alpha_rad = radians(90+theta0e_deg)
        cos_alpha = cos(alpha_rad)
        sin_alpha = sin(alpha_rad)
        x = sqrt(a*a - (b*b*sin_alpha*sin_alpha))
        d1_km = b*cos_alpha + x
        d2_km = b*cos_alpha - x
    elif theta0s_deg is not None:
        beta_rad = radians(90+theta0s_deg)
        cos_beta = cos(beta_rad)
        sin_beta = sin(beta_rad)
        x = sqrt(b*b - (a*a*sin_beta*sin_beta))
        d1_km = a*cos_beta + x
        d2_km = a*cos_beta - x
    else:
        raise ValueError('Either theta0e_deg or theta0s_deg needs to be specified.')

    if d1_km > 0 and d2_km <= 0:
        return d1_km
    elif d2_km > 0 and d1_km <= 0:
        return d2_km
    elif d1_km > 0 and d2_km > 0: # if 2 possible solutions
        return min(d1_km, d2_km) # select smallest distance solution for max potential interference
    else:
        raise ArithmeticError('No solution for distance.')


def _FreeSpaceElevAngleAtSpaceStation(He_km: float, Hs_km: float, theta0e_deg: float) -> float:
    """
    Not part of the recommendation.
    Uses trigonometry to calculate the free-space elevation angle at the space station (deg).

    Args:
        He_km (float): Height of the Earth-based station (km above mean sea level).
        Hs_km (float): Height of the space-based station (km above mean sea level).
        theta0e_deg (float): Free-space elevation angle (deg) at the Earth-based station.

    Returns:
        (float): The free space elevation angle (deg) at the space-based station.
    """
    d_km = _StraightLineDistance(He_km, Hs_km, theta0e_deg, None)
    _, theta0s_deg = _FreeSpaceElevAngles(He_km, Hs_km, d_km)
    return theta0s_deg


def _FreeSpaceElevAngleAtEarthStation(He_km: float, Hs_km: float, theta0s_deg: float) -> float:
    """
    Not part of the recommendation.
    Uses trigonometry to calculate the free-space elevation angle at the space station (deg).

    Args:
        He_km (float): Height of the Earth-based station (km above mean sea level).
        Hs_km (float): Height of the space-based station (km above mean sea level).
        theta0s_deg (float): Free-space elevation angle (deg) at the space-based station.

    Returns:
        (float): The free-space elevation angle (deg) at the Earth-based station.
    """
    d_km = _StraightLineDistance(He_km, Hs_km, None, theta0s_deg)
    theta0e_deg, _ = _FreeSpaceElevAngles(He_km, Hs_km, d_km)
    return theta0e_deg


# Not tested
def __UnwantedPowerDueToRainScatteringEstimate(r1: float, r2: float, unwanted_cyl: int, scatter_angle_deg: float,
                                               Rrain: float, f_GHz: float, Peirp: float, d_tx: float, d_rx: float,
                                               el_deg: float, tau_deg: float, P_hPa: float=1013.25,
                                               rho_gm3: float=7.5, T_K: float=288.15) -> float:
    """
    ITU-R P.619-5, Attachment F to Annex 1
    Provides a simple test which estimates the power received due to rain scattering in a common
    volume between two cylindrical antenna beams for a rain rate of Rrain mm/hr. The test estimates
    the unwanted rain-scattered power received by the victim receiver exceeded for a given 
    percentage time p.
    
    Args:
        r1 (float): Radius of cylinder representing the first antenna beam, in meters, with r1 <= r2.
        r2 (float): Radius of cylinder representing the second antenna beam, in meters, with r1 <= r2.
        unwanted_cyl (int): Cylinder representing the unwanted beam. Should be set to either 1 or 2.
        scatter_angle_deg (float): The angle between the direction of propagation of the incoming
            power in one cylinder and the scattered power travelling towards the victim receiver.
            The test is considered reliable for 10 <= scatter_angle_deg <= 90 degrees.
        Rrain (float): Point rainfall rate in mm/h for a 1 minute integration time exceeded for p% time.
        f_GHz (float): Frequency in GHz.
        Peirp (float): E.i.r.p. of the unwanted transmitter in dBW.
        d_tx (float): Distance from the unwanted transmitter to the common volume in km.
        d_rx (float): Distance in km of the interfered-with antenna from the common volume represented
            by the cylinders.
        el_deg (float): The path elevation angle, in degrees.
        tau_deg (float): The polarization tilt angle relative to the horizontal (tau = 45 deg for
            circular polarization).
        P_hPa (float): Atmospheric pressure, in hPa.
        rho_gm3 (float): Water vapour density, in g/m3.
        T_K (float): Temperature, in Kelvins.

    Returns:
        Ptxs (float): An estimate of the unwanted scattered power, in dBW. A full rain-scatter
        calculation should be conducted if Pint - Ptxs < 20 dB, where Pint is the receiver's
        interference threshold.
    """
    from . import itur_p838

    theta_rad = radians(scatter_angle_deg)

    # Step 1
    L1 = 2*r2/sin(theta_rad)
    L2 = max(L1*cos(theta_rad), 2*r1*sin(theta_rad))

    # Step 2
    gamma_g = itur_p676.GaseousAttenuation(f_GHz=f_GHz, P_hPa=P_hPa, T_K=T_K, rho_gm3=rho_gm3)
    S = Peirp - (20*log10(d_tx)) - (gamma_g*d_tx) - 71.0

    # Step 3
    if unwanted_cyl == 1:
        Pin = S + 10*log10(pi*r1*r1)
    else:
        Pin = S + 10*log10(pi*r2*r2)

    # Step 4
    k, alpha = itur_p838.Coefficients(f_GHz, el_deg, tau_deg)
    gamma_r = k*(Rrain**alpha)
    if unwanted_cyl == 1:
        Pout = Pin - 0.001*gamma_r*L1
    else:
        Pout = Pin - 0.001*gamma_r*L2

    # Step 5
    Pscat = 10*log10(10**(0.1*Pin) - 10**(0.1*Pout))

    # Step 6
    if unwanted_cyl == 1:
        Peirp_s = Pscat
    else:
        Peirp_s = Pscat - 10*log10((r2*r2*L2)/(r1*r1*L1))

    # Step 7
    if f_GHz > 10:
        Fnis = 0.001*(Rrain**0.4)*cos(theta_rad)*(2*(f_GHz-10)**1.6 - 2.5*(f_GHz-10)**1.7)
    else:
        Fnis = 0

    # Step 8
    Ptxs = Peirp_s + Fnis - (20*log10(d_rx*f_GHz)) - (gamma_g*d_rx) - 92.4

    return Ptxs
