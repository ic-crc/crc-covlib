"""Implementation of Section 3.7 and Attachment H from ITU-R P.2001-5.
"""

from math import sin, cos, radians, acos, atan2, fabs, asin, degrees, atan
import numpy.typing as npt
from . import itur_p1144
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['IsLOS',
           'ElevationAngles',
           'PathLength',
           'Bearing',
           'IntermediatePathPoint']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _Nd1km50(lat: float, lon: float) -> float:
    """
    ITU-R P.2001-5, Annex, Section 3.4.1
    Gets the median value of average refractivity gradient in the lowest 1 km of the atmosphere.
    (N-Units). Numerically equal to delta N as defined in ITU-R P.452 but with opposite sign.

    Args:
        lat (float): Mid-path latitude (degrees), with -90 <= lat <= 90.
        lon (float): Mid-path longitude (degrees), with -180 <= lon <= 180.
    
    Returns:
        (float): The median value of average refractivity gradient in the lowest 1 km of the
        atmosphere (N-units).
    """
    # lat from +90 to -90, lon from 0 to 360 in DN_Median.txt
    latInterval = 1.5
    lonInterval = 1.5
    numRows = 121 # (180/1.5)+1
    rowSize = 241 # (360/1.5)+1

    if lon < 0:
        lon += 360
    r = (90.0 - lat) / latInterval
    c = lon / lonInterval
    delta_N = itur_p1144.SquareGridBilinearInterpolation(_DN50, numRows, rowSize, r, c)
    return -delta_N


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _ElevAngle(h_from_mamsl, h_to_mamsl, dist_km, ae_km) -> float:
    """
    Returns the elevation angle (mrad) relative to the horizontal, does not appear to be suitable
    for short paths (about 100m or less, but possibly more depending on provided heights).
    """
    return ((h_to_mamsl-h_from_mamsl)/dist_km)-(500*dist_km/ae_km) # eq. (16) without max


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _ElevAngleFromP1812(h_from_mamsl, h_to_mamsl, dist_km, ae_km) -> float:
    """
    Returns the elevation angle (mrad) relative to the horizontal, using equation from
    ITU-R P.1812-7, Annex 1, Section 4
    """
    return 1000.0*atan(((h_to_mamsl-h_from_mamsl)/(1000.0*dist_km))-(dist_km/(2.0*ae_km)))


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def IsLOS(latt: float, lont: float, ht_mamsl: float, latr: float, lonr: float, hr_mamsl: float,
          dProfile_km: npt.ArrayLike, hProfile_mamsl: npt.ArrayLike) -> bool:
    """
    ITU-R P.2001-5, Annex, Section 3.7
    Determines whether a path is line-of-sight or non-line-of-sight under median refractivity
    conditions.

    Args:
        latt (float): Latitude of transmitter (degrees), with -90 <= lat <= 90.
        lont (float): Longitude of transmitter (degrees), with -180 <= lon <= 180.
        ht_mamsl (float): Transmitter height (meters above mean sea level).
        latr (float): Latitude of receiver (degrees), with -90 <= lat <= 90.
        lonr (float): Longitude of receiver (degrees), with -180 <= lon <= 180.
        hr_mamsl (float): Receiver height (meters above mean sea level).
        dProfile_km (numpy.typing.ArrayLike): Great-cirlcle distance from transmitter (km) profile.
        hProfile_mamsl (numpy.typing.ArrayLike): Terrain height profile (meters above mean sea
            level) from the transmitter to the receiver. hProfile and dProfile must have the same
            number of values.
    
    Returns:
        (bool): True when the path is line-of-sight, False otherwise.
    """
    path_length_km = dProfile_km[-1]
    if path_length_km < 1E-5: # to avoid dividing by 0 later on
        return True
    Re_km = 6371 # average Earth radius (km)
    n = len(dProfile_km)
    mid_path_lat, mid_path_lon = IntermediatePathPoint(latt, lont, latr, lonr, path_length_km/2.0)
    ae_km = 157*Re_km/(157+_Nd1km50(mid_path_lat, mid_path_lon)) # median effective Earth radius (km)
    theta_tr = _ElevAngle(ht_mamsl, hr_mamsl, path_length_km, ae_km)
    theta_tim = -1E20
    for i in range(1, n-1):
        theta_tim_i = _ElevAngle(ht_mamsl, hProfile_mamsl[i], dProfile_km[i], ae_km)
        theta_tim = max(theta_tim, theta_tim_i)
    # Note: Using _ElevAngle() or _ElevAngleFromP1812() yields the same LOS/NLOS result since the
    #       angle comparison is consistent.
    return theta_tim < theta_tr


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def ElevationAngles(latt: float, lont: float, ht_mamsl: float,
                    latr: float, lonr: float, hr_mamsl: float,
                    dProfile_km: npt.ArrayLike, hProfile_mamsl: npt.ArrayLike,
                    useP1812Variation: bool=True) -> tuple[float, float]:
    """
    ITU-R P.2001-5, Annex, Section 3.7
    Calculates the transmitter and receiver elevation angles (degrees) under median refractivity
    conditions.

    Args:
        latt (float): Latitude of transmitter (degrees), with -90 <= lat <= 90.
        lont (float): Longitude of transmitter (degrees), with -180 <= lon <= 180.
        ht_mamsl (float): Transmitter height (meters above mean sea level).
        latr (float): Latitude of receiver (degrees), with -90 <= lat <= 90.
        lonr (float): Longitude of receiver (degrees), with -180 <= lon <= 180.
        hr_mamsl (float): Receiver height (meters above mean sea level).
        dProfile_km (numpy.typing.ArrayLike): Great-cirlcle distance from transmitter (km) profile.
        hProfile_mamsl (numpy.typing.ArrayLike): Terrain height profile (meters above mean sea
            level) from the transmitter to the receiver. hProfile and dProfile must have the same
            number of values.
        useP1812Variation (bool): The formula provided in ITU-R P.2001 to calculate elevation
            angles does not appear to be suitable for short distances. When set to True, the
            formula from ITU-R P.1812-7 is used instead.
    
    Returns:
        (float): Transmitter elevation angle (degrees). 0°=horizon, +90°=zenith, -90°=nadir.
        (float): Receiver elevation angle (degrees). 0°=horizon, +90°=zenith, -90°=nadir.
    """
    path_length_km = dProfile_km[-1]
    if path_length_km < 1E-5: # to avoid dividing by 0 later on
        if ht_mamsl > hr_mamsl:
            return (-90, 90)
        elif ht_mamsl < hr_mamsl:
            return (90, -90)
        else:
            return (0, 0)
    Re_km = 6371
    n = len(dProfile_km)
    mid_path_lat, mid_path_lon = IntermediatePathPoint(latt, lont, latr, lonr, path_length_km/2.0)
    ae_km = 157*Re_km/(157+_Nd1km50(mid_path_lat, mid_path_lon))
    # not using function type here since it is experimental in Numba
    if useP1812Variation == True:
        theta_t = _ElevAngleFromP1812(ht_mamsl, hr_mamsl, path_length_km, ae_km)
        theta_r = _ElevAngleFromP1812(hr_mamsl, ht_mamsl, path_length_km, ae_km)
        for i in range(1, n-1):
            theta_t_i = _ElevAngleFromP1812(ht_mamsl, hProfile_mamsl[i], dProfile_km[i], ae_km)
            theta_r_i = _ElevAngleFromP1812(hr_mamsl, hProfile_mamsl[i], path_length_km-dProfile_km[i], ae_km)
            theta_t = max(theta_t, theta_t_i)
            theta_r = max(theta_r, theta_r_i)
    else:
        theta_t = _ElevAngle(ht_mamsl, hr_mamsl, path_length_km, ae_km)
        theta_r = _ElevAngle(hr_mamsl, ht_mamsl, path_length_km, ae_km)
        for i in range(1, n-1):
            theta_t_i = _ElevAngle(ht_mamsl, hProfile_mamsl[i], dProfile_km[i], ae_km)
            theta_r_i = _ElevAngle(hr_mamsl, hProfile_mamsl[i], path_length_km-dProfile_km[i], ae_km)
            theta_t = max(theta_t, theta_t_i)
            theta_r = max(theta_r, theta_r_i)
    theta_t_deg = degrees(theta_t/1000) # convert from mrad to degrees
    theta_r_deg = degrees(theta_r/1000)
    return (theta_t_deg, theta_r_deg)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def PathLength(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """
    ITU-R P.2001-5, Attachment H (H.2)
    Calculates the great-circle path length between two points on the surface of the Earth (km).
    
    Args:
        lat0 (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
        lon0 (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
        lat1 (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
        lon1 (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.

    Returns:
        (float): The great-circle path length between two points on the surface of the Earth (km).
    """
    Re = 6371.0
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    path_length_km = acos( (sin(lat0)*sin(lat1)) + (cos(lat0)*cos(lat1)*cos(lon1-lon0)) ) * Re
    return path_length_km


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Bearing(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """
    ITU-R P.2001-5, Attachment H (H.2)
    Calculates the bearing of the great-circle path between two points on the surface of the Earth,
    in degrees.

    Args:
        lat0 (float): Latitude of first point (degrees), with -90 <= lat0 <= 90.
        lon0 (float): Longitude of first point (degrees), with -180 <= lon0 <= 180.
        lat1 (float): Latitude of second point (degrees), with -90 <= lat1 <= 90.
        lon1 (float): Longitude of second point (degrees), with -180 <= lon1 <= 180.

    Returns:
        (float): The bearing of the great-circle path from the first point towards the second point
            (degrees), with 0 <= bearing <= 360. Corresponds to the angle between due north at the
            first point eastwards (clockwise) to the direction of the path.
    """
    phi_tn = radians(lat0)
    sin_phi_tn = sin(phi_tn)
    cos_phi_tn = cos(phi_tn)
    phi_te = radians(lon0)
    phi_rn = radians(lat1)
    sin_phi_rn = sin(phi_rn)
    cos_phi_rn = cos(phi_rn)
    phi_re = radians(lon1)

    delta_lon = phi_re-phi_te
    r = sin_phi_tn*sin_phi_rn+cos_phi_tn*cos_phi_rn*cos(delta_lon)
    x1 = sin_phi_rn-r*sin_phi_tn
    y1 = cos_phi_tn*cos_phi_rn*sin(delta_lon)
    if fabs(x1) < 1E-9 and fabs(y1) < 1E-9:
        Bt2r = phi_re
    else:
        Bt2r = atan2(y1, x1)
    
    bearing_deg = degrees(Bt2r)
    if bearing_deg < 0:
        bearing_deg += 360
    return bearing_deg


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def IntermediatePathPoint(lat0: float, lon0: float, lat1: float, lon1: float,
                          dist_km: float) -> tuple[float, float]:
    """
    ITU-R P.2001-5, Attachment H (H.3)
    Calculates the latitude and longitude of any point along a great-circle path on the surface of
    the Earth.

    Args:
        lat0 (float): Latitude at the first end of the path (degrees), with -90 <= lat0 <= 90.
        lon0 (float): Longitude at the first end of the path (degrees), with -180 <= lon0 <= 180.
        lat1 (float): Latitude at the second end of the path (degrees), with -90 <= lat1 <= 90.
        lon1 (float): Longitude at the second end of the path (degrees), with -180 <= lon1 <= 180.
        dist_km (float): Great-circle distance of an intermediate point from the first end of the
            path (km).

    Returns:
        lat (float): Latitude of the intermediate point along the great-circle path (degrees).
        lon (float): Longitude of the intermediate point along the great-circle path(degrees).
    """
    Re = 6371.0
    phi_tn = radians(lat0)
    sin_phi_tn = sin(phi_tn)
    cos_phi_tn = cos(phi_tn)
    phi_te = radians(lon0)
    phi_rn = radians(lat1)
    sin_phi_rn = sin(phi_rn)
    cos_phi_rn = cos(phi_rn)
    phi_re = radians(lon1)
    phi_pnt = dist_km/Re
    sin_phi_pnt = sin(phi_pnt)
    cos_phi_pnt = cos(phi_pnt)

    delta_lon = phi_re-phi_te
    r = sin_phi_tn*sin_phi_rn+cos_phi_tn*cos_phi_rn*cos(delta_lon)
    x1 = sin_phi_rn-r*sin_phi_tn
    y1 = cos_phi_tn*cos_phi_rn*sin(delta_lon)
    if fabs(x1) < 1E-9 and fabs(y1) < 1E-9:
        Bt2r = phi_re
    else:
        Bt2r = atan2(y1, x1)

    s = sin_phi_tn*cos_phi_pnt+cos_phi_tn*sin_phi_pnt*cos(Bt2r)
    phi_pntn = asin(s)
    x2 = cos_phi_pnt-s*sin_phi_tn
    y2 = cos_phi_tn*sin_phi_pnt*sin(Bt2r)
    if fabs(x2) < 1E-9 and fabs(y2) < 1E-9:
        phi_pnte = Bt2r
    else:
        phi_pnte = phi_te+atan2(y2, x2)

    interm_lat = degrees(phi_pntn)
    interm_lon = degrees(phi_pnte)
    if interm_lon < -180:
        interm_lon += 360
    if interm_lon > 180:
        interm_lon -= 360

    return (interm_lat, interm_lon)


# Data originally from ITU file DN_Median.txt within 'R-REC-P.2001-5-202308-I!!ZIP-E.zip'
_DN50 = itur_p1144.LoadITUDigitalMapFile('data/itu_proprietary/p2001/DN_Median.txt')
