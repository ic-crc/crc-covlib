"""Implementation of ITU-R P.2145-0

This module supports using both annual and monthly statistics. Statistics data files can be
obtained at https://www.itu.int/rec/R-REC-P.2145/en or they can be installed by running the
install_ITU_data.py script (use the Custom mode for the option to install monthly statistics,
the default mode will only install annual statistics). Another option is to use functions from the
itur_data.py module.

When installing files manually (i.e. when not using the install_ITU_data.py script):
For each month, 4 directories must be added in the helper/data/itu_proprietary/p2145/ directory,
namely RHO_Month03, P_Month03, T_Month03 and V_Month03 (using the month of March as an example).
Annual statistics must be placed in RHO_Annual, P_Annual, T_Annual, V_Annual directories.
"""

from . import itur_p1144
from . import itur_p1511
from typing import Union, Callable
import numpy.typing as npt
from math import exp, log10


__all__ = ['SurfaceTotalPressure',
           'SurfaceTemperature',
           'SurfaceWaterVapourDensity',
           'IntegratedWaterVapourContent',
           'MeanSurfaceTotalPressure',
           'MeanSurfaceTemperature',
           'MeanSurfaceWaterVapourDensity',
           'MeanIntegratedWaterVapourContent',
           'StdDevSurfaceTotalPressure',
           'StdDevSurfaceTemperature',
           'StdDevSurfaceWaterVapourDensity',
           'StdDevIntegratedWaterVapourContent',
           'WeibullParameters']


def SurfaceTotalPressure(p: float, lat: float, lon: float, month: Union[int, None]=None,
                         h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.1 of Annex.
    Gets the surface total (barometric) pressure (hPa) for the specified exceedence probabiliby p
    (CCDF - complementary cumulative distribution function) at the specified location on the 
    surface of the Earth.
    
    Args:
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        P (float): The surface total (barometric) pressure (hPa) for the specified exceedence
            probabiliby p (CCDF) at the specified location on the surface of the Earth.
    """
    maps = _GetCCDFMaps(p, month, 'P')
    if maps is not None:
        return _CCDFInterpolation(p, lat, lon, h_mamsl, maps, _ExpScalingFunc)
    else:
        return float('nan')


def SurfaceTemperature(p: float, lat: float, lon: float, month: Union[int, None]=None,
                       h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.1 of Annex.
    Gets the surface temperature (K) for the specified exceedence probabiliby p (CCDF - 
    complementary cumulative distribution function) at the specified location on the surface of the
    Earth.
    
    Args:
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        T (float): The surface temperature (K) for the specified exceedence probabiliby p (CCDF) at
            the specified location on the surface of the Earth.
    """
    maps = _GetCCDFMaps(p, month, 'T')
    if maps is not None:
        return _CCDFInterpolation(p, lat, lon, h_mamsl, maps, _AddScalingFunc)
    else:
        return float('nan')


def SurfaceWaterVapourDensity(p: float, lat: float, lon: float, month: Union[int, None]=None,
                              h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.1 of Annex.
    Gets the surface water vapour density (g/m3) for the specified exceedence probabiliby p (CCDF - 
    complementary cumulative distribution function) at the specified location on the surface of the
    Earth.
    
    Args:
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        rho (float): The surface water vapour density (g/m3) for the specified exceedence
            probabiliby p (CCDF) at the specified location on the surface of the Earth.
    """
    maps = _GetCCDFMaps(p, month, 'RHO')
    if maps is not None:
        return _CCDFInterpolation(p, lat, lon, h_mamsl, maps, _ExpScalingFunc)
    else:
        return float('nan')


def IntegratedWaterVapourContent(p: float, lat: float, lon: float, month: Union[int, None]=None,
                                 h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.1 of Annex.
    Gets the integrated water vapour content (kg/m2) for the specified exceedence probabiliby p
    (CCDF - complementary cumulative distribution function) at the specified location on the 
    surface of the Earth.
    
    Args:
        p (float): Exceedance probability (CCDF) of interest, in %, with 0.01 <= p <= 99 for annual
            statistics and with 0.1 <= p <= 99 for monthly statistics.
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        V (float): The integrated water vapour content (kg/m2) for the specified exceedence
            probabiliby p (CCDF) at the specified location on the surface of the Earth.
    """
    maps = _GetCCDFMaps(p, month, 'V')
    if maps is not None:
        return _CCDFInterpolation(p, lat, lon, h_mamsl, maps, _ExpScalingFunc)
    else:
        return float('nan')


def MeanSurfaceTotalPressure(lat: float, lon: float, month: Union[int, None]=None,
                             h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the mean of the surface total (barometric) pressure (hPa) at the specified location on the 
    surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The mean of the surface total (barometric) pressure (hPa) at the specified
            location on the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'P', True, False)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.mean_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc)


def MeanSurfaceTemperature(lat: float, lon: float, month: Union[int, None]=None,
                           h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the mean of the surface temperature (K) at the specified location on the surface of the
    Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The mean of the surface temperature (K) at the specified location on the surface
            of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'T', True, False)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.mean_map, maps.sch_map, maps.z_map,
                                    _AddScalingFunc)


def MeanSurfaceWaterVapourDensity(lat: float, lon: float, month: Union[int, None]=None,
                                  h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the mean of the surface water vapour density (g/m3) at the specified location on the
    surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The mean of the surface water vapour density (g/m3) at the specified location on
            the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'RHO', True, False)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.mean_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc)


def MeanIntegratedWaterVapourContent(lat: float, lon: float, month: Union[int, None]=None,
                                     h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the mean of the integrated water vapour content (kg/m2) at the specified location on the 
    surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The mean of the integrated water vapour content (kg/m2) at the specified location
            on the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'V', True, False)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl,  maps.mean_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc)


def StdDevSurfaceTotalPressure(lat: float, lon: float, month: Union[int, None]=None,
                               h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the standard deviation of the surface total (barometric) pressure (hPa) at the specified
    location on the surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The standard deviation of the surface total (barometric) pressure (hPa) at the
            specified location on the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'P', False, True)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.std_dev_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc)


def StdDevSurfaceTemperature(lat: float, lon: float, month: Union[int, None]=None,
                             h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the standard deviation of the surface temperature (K) at the specified location on the
    surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The standard deviation of the surface temperature (K) at the specified location on
            the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'T', False, True)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.std_dev_map, maps.sch_map, maps.z_map,
                                    _DirectScalingFunc)


def StdDevSurfaceWaterVapourDensity(lat: float, lon: float, month: Union[int, None]=None,
                                    h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the standard deviation of the surface water vapour density (g/m3) at the specified
    location on the surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The standard deviation of the surface water vapour density (g/m3) at the specified
            location on the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'RHO', False, True)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.std_dev_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc)


def StdDevIntegratedWaterVapourContent(lat: float, lon: float, month: Union[int, None]=None,
                                       h_mamsl: Union[float, None]=None) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the standard deviation of the integrated water vapour content (kg/m2) at the specified
    location on the surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        month (int|None): Month to get the statistics from (1=Jan, ..., 12=Dec). When set to None,
            annual statistics are used.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        (float): The standard deviation of the integrated water vapour content (kg/m2) at the
            specified location on the surface of the Earth.
    """
    maps = _GetMeanStdDevMaps(month, 'V', False, True)
    return _MeanStdDevInterpolation(lat, lon, h_mamsl, maps.std_dev_map, maps.sch_map, maps.z_map,
                                    _ExpScalingFunc) 


def WeibullParameters(lat: float, lon: float, h_mamsl: Union[float, None]=None
                      ) -> tuple[float, float]:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    Gets the Weibull integrated water vapour content shape and scale parameters at the specified
    location on the surface of the Earth.

    Args:
        lat (float): Latitude of the desired location (deg), with -90 <= lat <= 90.
        lon (float): Longitude of the desired location (deg), with -180 <= lon <= 180.
        h_mamsl (float|None): Height of the desired location (meters above mean sea level). When
            set to None, the height is automatically obtained from Recommendation ITU-R P.1511.

    Returns:
        kVS (float): The Weibull integrated water vapour content shape parameters.
        lambdaVS (float): The Weibull integrated water vapour content scale parameters.
    """
    dir = 'data/itu_proprietary/p2145/Weibull_Annual/'

    kV_file = dir + 'kV.TXT'
    lambdaV_file = dir + 'lambdaV.TXT'
    sch_file = dir + 'VSCH.TXT'
    z_file = dir + 'Z_ground.TXT'

    sch_map = _GetDigitalMap(sch_file)
    z_map = _GetDigitalMap(z_file)

    kV_map = _GetDigitalMap(kV_file)
    kV = _MeanStdDevInterpolation(lat, lon, h_mamsl, kV_map, sch_map, z_map, _DirectScalingFunc)

    lambdaV_map = _GetDigitalMap(lambdaV_file)
    labmdaV = _MeanStdDevInterpolation(lat, lon, h_mamsl, lambdaV_map, sch_map, z_map, _ExpScalingFunc)

    return (kV, labmdaV)


class _CCDFMaps:
    p_below: float
    p_below_map: npt.ArrayLike
    p_above: float
    p_above_map: npt.ArrayLike
    sch_map: npt.ArrayLike
    z_map: npt.ArrayLike


class _MeanStdDevMaps:
    mean_map: Union[npt.ArrayLike, None]
    std_dev_map: Union[npt.ArrayLike, None]
    sch_map: npt.ArrayLike
    z_map: npt.ArrayLike


def _GetCCDFMaps(p: float, month: Union[int, None], param: str) -> Union[_CCDFMaps, None]:
    if month is None:
        nominal_percentages = _ANNUAL_P
        dir = 'data/itu_proprietary/p2145/{}_Annual/'.format(param)
    else:
        nominal_percentages = _MONTHLY_P
        dir = 'data/itu_proprietary/p2145/{}_Month{:02d}/'.format(param, month)

    for i in range(0, len(nominal_percentages)-1):
        if p >= nominal_percentages[i][0] and p <= nominal_percentages[i+1][0]:
            ccdf_maps = _CCDFMaps()
            ccdf_maps.p_below = nominal_percentages[i][0]
            ccdf_maps.p_above = nominal_percentages[i+1][0]

            p_below_file = dir + '{}_{}.TXT'.format(param, nominal_percentages[i][1])
            p_above_file = dir + '{}_{}.TXT'.format(param, nominal_percentages[i+1][1])
            if param == 'RHO':
                sch_file = dir + 'VSCH.TXT'
            else:
                sch_file = dir + '{}SCH.TXT'.format(param)
            z_file = dir + 'Z_ground.TXT'

            ccdf_maps.p_below_map = _GetDigitalMap(p_below_file)
            ccdf_maps.p_above_map = _GetDigitalMap(p_above_file)
            ccdf_maps.sch_map = _GetDigitalMap(sch_file)
            ccdf_maps.z_map = _GetDigitalMap(z_file)

            return ccdf_maps
        
    return None


def _GetMeanStdDevMaps(month: Union[int, None], param: str, getMeanMap: bool, getStdDevMap: bool
                       ) -> _MeanStdDevMaps:
    if month is None:
        dir = 'data/itu_proprietary/p2145/{}_Annual/'.format(param)
    else:
        dir = 'data/itu_proprietary/p2145/{}_Month{:02d}/'.format(param, month)

    mean_std_dev_maps = _MeanStdDevMaps()

    mean_std_dev_maps.mean_map = None
    if getMeanMap == True:
        mean_file = dir + '{}_mean.TXT'.format(param)
        mean_std_dev_maps.mean_map = _GetDigitalMap(mean_file)

    mean_std_dev_maps.std_dev_map = None
    if getStdDevMap == True:
        std_dev_file = dir + '{}_std.TXT'.format(param)
        mean_std_dev_maps.std_dev_map = _GetDigitalMap(std_dev_file)

    if param == 'RHO':
        sch_file = dir + 'VSCH.TXT'
    else:
        sch_file = dir + '{}SCH.TXT'.format(param)
    mean_std_dev_maps.sch_map = _GetDigitalMap(sch_file)

    z_file = dir + 'Z_ground.TXT'
    mean_std_dev_maps.z_map = _GetDigitalMap(z_file)

    return mean_std_dev_maps


def _CCDFInterpolation(p: float, lat: float, lon: float,
                       h_mamsl: Union[float, None], maps: _CCDFMaps,
                       scalingFunc: Callable[[float, float, float, float], float]) -> float:
    """
    ITU-R P.2145-0, Section 2.1 of Annex.
    """
    # a)
    alt = h_mamsl
    if alt is None:
        alt = itur_p1511.TopographicHeightAMSL(lat, lon) # returns meters
    alt = alt/1000 # convert from m to km

    # lat from -90 to +90, lon from -180 to +180, intervals of 0.25 deg
    num_rows = 721 # (180/0.25)+1
    row_size = 1441 # (360/0.25)+1
    r = (90.0 + lat) / 0.25
    c = (180.0 + lon) / 0.25
    R = int(r)
    R = max(R, 0)
    R = min(R, num_rows-2)
    C = int(c)
    C = max(C, 0)
    C = min(C, row_size-2)

    i0 = R*row_size + C
    i1 = (R+1)*row_size + C
    i2 = R*row_size + (C+1)
    i3 = (R+1)*row_size + (C+1)

    # c)
    Xpa0, Xpa1, Xpa2, Xpa3 = maps.p_above_map[i0], maps.p_above_map[i1], maps.p_above_map[i2], \
                             maps.p_above_map[i3]
    Xpb0, Xpb1, Xpb2, Xpb3 = maps.p_below_map[i0], maps.p_below_map[i1], maps.p_below_map[i2], \
                             maps.p_below_map[i3]

    # d)
    sch0, sch1, sch2, sch3 = maps.sch_map[i0], maps.sch_map[i1], maps.sch_map[i2], maps.sch_map[i3]
    
    # e)
    alt0, alt1, alt2, alt3 = maps.z_map[i0], maps.z_map[i1], maps.z_map[i2], maps.z_map[i3]

    # f)
    Xa0, Xa1, Xa2, Xa3 = scalingFunc(Xpa0, sch0, alt, alt0), scalingFunc(Xpa1, sch1, alt, alt1), \
                         scalingFunc(Xpa2, sch2, alt, alt2), scalingFunc(Xpa3, sch3, alt, alt3)
    Xb0, Xb1, Xb2, Xb3 = scalingFunc(Xpb0, sch0, alt, alt0), scalingFunc(Xpb1, sch1, alt, alt1), \
                         scalingFunc(Xpb2, sch2, alt, alt2), scalingFunc(Xpb3, sch3, alt, alt3)

    # g)
    X_above = Xa0*((R+1-r)*(C+1-c)) + Xa1*((r-R)*(C+1-c)) + Xa2*((R+1-r)*(c-C)) + Xa3*((r-R)*(c-C))
    X_below = Xb0*((R+1-r)*(C+1-c)) + Xb1*((r-R)*(C+1-c)) + Xb2*((R+1-r)*(c-C)) + Xb3*((r-R)*(c-C))

    # h)
    X0 = X_below
    X1 = X_above
    p0 = maps.p_below
    p1 = maps.p_above
    X = (X0*log10(p1/p) + X1*log10(p/p0)) / log10(p1/p0)
    
    return X


def _MeanStdDevInterpolation(lat: float, lon: float, h_mamsl: Union[float, None],
                             mainMap: npt.ArrayLike, schMap: npt.ArrayLike, zMap: npt.ArrayLike,
                             scalingFunc: Callable[[float, float, float, float], float]) -> float:
    """
    ITU-R P.2145-0, Section 2.2 of Annex.
    """
    # a)
    alt = h_mamsl
    if alt is None:
        alt = itur_p1511.TopographicHeightAMSL(lat, lon) # returns meters
    alt = alt/1000 # convert from m to km

    # lat from -90 to +90, lon from -180 to +180, intervals of 0.25 deg
    num_rows = 721 # (180/0.25)+1
    row_size = 1441 # (360/0.25)+1
    r = (90.0 + lat) / 0.25
    c = (180.0 + lon) / 0.25
    R = int(r)
    R = max(R, 0)
    R = min(R, num_rows-2)
    C = int(c)
    C = max(C, 0)
    C = min(C, row_size-2)

    i0 = R*row_size + C
    i1 = (R+1)*row_size + C
    i2 = R*row_size + (C+1)
    i3 = (R+1)*row_size + (C+1)

    # b)
    Xp0, Xp1, Xp2, Xp3 = mainMap[i0], mainMap[i1], mainMap[i2], mainMap[i3]

    # c)
    sch0, sch1, sch2, sch3 = schMap[i0], schMap[i1], schMap[i2], schMap[i3]

    # d)
    alt0, alt1, alt2, alt3 = zMap[i0], zMap[i1], zMap[i2], zMap[i3]

    # e)
    X0 = scalingFunc(Xp0, sch0, alt, alt0)
    X1 = scalingFunc(Xp1, sch1, alt, alt1)
    X2 = scalingFunc(Xp2, sch2, alt, alt2)
    X3 = scalingFunc(Xp3, sch3, alt, alt3)

    # f)
    X = X0*((R+1-r)*(C+1-c)) + X1*((r-R)*(C+1-c)) + X2*((R+1-r)*(c-C)) + X3*((r-R)*(c-C))

    return X


def _ExpScalingFunc(Xp, sch, alt, alti):
    return Xp*exp(-(alt-alti)/sch)


def _AddScalingFunc(Xp, sch, alt, alti):
    return Xp+(sch*(alt-alti))


def _DirectScalingFunc(Xp, sch, alt, alti):
    return Xp


def _GetDigitalMap(pathname: str) -> npt.ArrayLike:
    if pathname not in _digital_maps:
        _digital_maps[pathname] = itur_p1144._LoadITUDigitalMapFile(pathname)
    map = _digital_maps[pathname]
    return map


_ANNUAL_P = [(0.01, '001'), (0.02, '002'), (0.03, '003'), (0.05, '005'), (0.1, '01'), (0.2, '02'),
             (0.3, '03'), (0.5, '05'), (1, '1'), (2, '2'), (3, '3'), (5, '5'), (10, '10'),
             (20, '20'), (30, '30'), (50, '50'), (60, '60'), (70, '70'), (80, '80'), (90, '90'),
             (95, '95'), (99, '99')]


_MONTHLY_P = [(0.1, '01'), (0.2, '02'), (0.3, '03'), (0.5, '05'), (1, '1'), (2, '2'), (3, '3'),
              (5, '5'), (10, '10'), (20, '20'), (30, '30'), (50, '50'), (60, '60'), (70, '70'),
              (80, '80'), (90, '90'), (95, '95'), (99, '99')]


_digital_maps = {}
