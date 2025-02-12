"""Implementation of ITU-R P.838-3.
"""

from math import log10, exp, radians, cos
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['RainAttenuation',
           'Coefficients',
           'FIGURE_1',
           'FIGURE_2',
           'FIGURE_3',
           'FIGURE_4']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def RainAttenuation(f_GHz: float, rainRate_mmhr: float, pathElevAngle_deg: float,
                    polTiltAngle_deg: float) -> float:
    """
    ITU-R P.838-3.
    Attenuation due to rain (dB/km).

    Args:
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
        rainRate_mmhr (float): Rain rate (mm/hr).
        pathElevAngle_deg (float): Path elevation angle (deg).
        polTiltAngle_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
          for circular polarization.

    Returns:
        (float): Attenuation due to rain (dB/km).
    """
    k, alpha = Coefficients(f_GHz, pathElevAngle_deg, polTiltAngle_deg)
    gamma_R = k*pow(rainRate_mmhr, alpha)
    return gamma_R


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Coefficients(f_GHz: float, pathElevAngle_deg: float, polTiltAngle_deg: float) -> tuple[float, float]:
    """
    ITU-R P.838-3.
    Gets the coefficients k and alpha from equations (4) and (5).

    Args:
        f_GHz (float): Frequency (GHz), with 1 <= f_GHz <= 1000.
        pathElevAngle_deg (float): Path elevation angle (deg).
        polTiltAngle_deg (float): Polarization tilt angle relative to the horizontal (deg). Use 45°
          for circular polarization.

    Returns:
        k (float): Coefficient k.
        alpha (float): Coefficient alpha.
    """
    theta_rad = radians(pathElevAngle_deg)
    tau_rad = radians(polTiltAngle_deg)
    kH = _kH(f_GHz)
    kV = _kV(f_GHz)
    aH = _alphaH(f_GHz)
    aV = _alphaV(f_GHz)
    cos_th = cos(theta_rad)
    cos_2tau = cos(2*tau_rad)
    k = (kH+kV+((kH-kV)*cos_th*cos_th*cos_2tau))/2
    alpha = ((kH*aH)+(kV*aV)+(((kH*aH)-(kV*aV))*cos_th*cos_th*cos_2tau))/(2*k)
    return (k, alpha)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _kH(f_GHz: float) -> float:
    mk = -0.18961
    ck = 0.71147
    a = [-5.33980, -0.35351, -0.23789, -0.94158]
    b = [-0.10008, 1.26970, 0.86036, 0.64552]
    c = [1.13098, 0.45400, 0.15354, 0.16817]
    return _k(f_GHz, a, b, c, mk, ck)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _kV(f_GHz: float) -> float:
    mk = -0.16398
    ck = 0.63297
    a = [-3.80595, -3.44965, -0.39902, 0.50167]
    b = [0.56934, -0.22911, 0.73042, 1.07319]
    c = [0.81061, 0.51059, 0.11899, 0.27195]
    return _k(f_GHz, a, b, c, mk, ck)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _alphaH(f_GHz: float) -> float:
    ma = 0.67849
    ca = -1.95537
    a = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721]
    b = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980]
    c = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
    return _alpha(f_GHz, a, b, c, ma, ca)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _alphaV(f_GHz: float) -> float:
    ma = -0.053739
    ca = 0.83433
    a = [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833]
    b = [2.33840, 0.95545, 1.14520, 0.791669, 0.791459]
    c = [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479]
    return _alpha(f_GHz, a, b, c, ma, ca)


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _k(f_GHz: float, a: list, b: list, c: list, mk: float, ck: float) -> float:
    log10_f = log10(f_GHz)
    sum = 0
    for j in range(0, 4, 1):
        base = (log10_f-b[j])/c[j]
        sum += a[j]*exp(-(base*base))
    log10_k = sum+(mk*log10_f)+ck
    k = pow(10, log10_k)
    return k


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def _alpha(f_GHz: float, a: list, b: list, c: list, ma: float, ca: float) -> float:
    log10_f = log10(f_GHz)
    sum = 0
    for j in range(0, 5, 1):
        base = (log10_f-b[j])/c[j]
        sum += a[j]*exp(-(base*base))
    alpha = sum+(ma*log10_f)+ca
    return alpha


def FIGURE_1() -> None:
    """
    Calculates values and display FIGURE 1 from ITU-R P.838-3.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [*np.arange(1, 10, 0.1), *np.arange(10, 100, 1), *np.arange(100, 1000+1, 5)]
    k_coeffs = []
    for f_GHz in freqs:
        k_coeffs.append(_kH(f_GHz))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6.75)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([1, 1000])
    ax1.set_xticks([1, 10, 100, 1000])
    ax1.set_ylim([1E-5, 10])
    ax1.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10])
    ax1.plot(freqs, k_coeffs, color='#000000')
    ax1.set_title('FIGURE 1\nk coefficient for horizontal polarization')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Coefficient kH')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_2() -> None:
    """
    Calculates values and display FIGURE 2 from ITU-R P.838-3.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [*np.arange(1, 10, 0.1), *np.arange(10, 100, 1), *np.arange(100, 1000+1, 5)]
    alpha_coeffs = []
    for f_GHz in freqs:
        alpha_coeffs.append(_alphaH(f_GHz))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6.75)
    ax1.set_xscale('log')
    ax1.set_xlim([1, 1000])
    ax1.set_xticks([1, 10, 100, 1000])
    ax1.set_ylim([0.4, 1.8])
    ax1.set_yticks([*np.arange(0.4, 1.8+0.001, 0.2)])
    ax1.plot(freqs, alpha_coeffs, color='#000000')
    ax1.set_title('FIGURE 2\nalpha coefficient for horizontal polarization')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Coefficient alphaH')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_3() -> None:
    """
    Calculates values and display FIGURE 3 from ITU-R P.838-3.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [*np.arange(1, 10, 0.1), *np.arange(10, 100, 1), *np.arange(100, 1000+1, 5)]
    k_coeffs = []
    for f_GHz in freqs:
        k_coeffs.append(_kV(f_GHz))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6.75)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([1, 1000])
    ax1.set_xticks([1, 10, 100, 1000])
    ax1.set_ylim([1E-5, 10])
    ax1.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10])
    ax1.plot(freqs, k_coeffs, color='#000000')
    ax1.set_title('FIGURE 3\nk coefficient for vertical polarization')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Coefficient kV')
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_4() -> None:
    """
    Calculates values and display FIGURE 4 from ITU-R P.838-3.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    freqs = [*np.arange(1, 10, 0.1), *np.arange(10, 100, 1), *np.arange(100, 1000+1, 5)]
    alpha_coeffs = []
    for f_GHz in freqs:
        alpha_coeffs.append(_alphaV(f_GHz))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6.75)
    ax1.set_xscale('log')
    ax1.set_xlim([1, 1000])
    ax1.set_xticks([1, 10, 100, 1000])
    ax1.set_ylim([0.4, 1.8])
    ax1.set_yticks([*np.arange(0.4, 1.8+0.001, 0.2)])
    ax1.plot(freqs, alpha_coeffs, color='#000000')
    ax1.set_title('FIGURE 4\nalpha coefficient for vertical polarization')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Coefficient alphaV')
    plt.grid(True, 'both','both')
    plt.show()
