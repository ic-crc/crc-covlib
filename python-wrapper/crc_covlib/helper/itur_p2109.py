"""Implementation of ITU-R P.2109-2.
"""

from math import log10
from . import itur_p1057
import enum
import numpy as np
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['BuildingType',
           'BuildingEntryLoss',
           'FIGURE_1',
           'FIGURE_2']


class BuildingType(enum.Enum):
    TRADITIONAL         = 1
    THERMALLY_EFFICIENT = 2


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def BuildingEntryLoss(f_GHz: float, prob_percent: float, bldgType: BuildingType,
                      elevAngle_deg: float) -> float:
    """
    ITU-R P.2109-2, Annex 1, Section 3
    Building entry loss (dB).

    Args:
        f_GHz (float): Frequency (GHz), with 0.08 <= f_GHz <= 100.
        prob_percent (float): The probability with which the loss is not exceeded (%), 
            with 0 < prob_percent < 100.
        bldgType (crc_covlib.helper.itur_p2109.BuildingType): Building type (traditional or
            thermally efficient).
        elevAngle_deg (float): Elevation angle of the path at the building façade (degrees
            above/below the horizontal), with -90 < elevAngle_deg < 90.

    Returns:
        (float): Building entry loss (dB).
    """
    i = bldgType.value - BuildingType.TRADITIONAL.value
    r = _BLDG_COEFFS[i][0]
    s = _BLDG_COEFFS[i][1]
    t = _BLDG_COEFFS[i][2]
    u = _BLDG_COEFFS[i][3]
    v = _BLDG_COEFFS[i][4]
    w = _BLDG_COEFFS[i][5]
    x = _BLDG_COEFFS[i][6]
    y = _BLDG_COEFFS[i][7]
    z = _BLDG_COEFFS[i][8]

    C = -3.0
    logf = log10(f_GHz)
    Lh = r + s*logf + t*logf*logf
    Le = 0.212*abs(elevAngle_deg)
    mu1 = Lh + Le
    mu2 = w + x*logf
    sigma1 = u + v*logf
    sigma2 = y + z*logf
    FinvP = itur_p1057.Finv(prob_percent/100.0)
    AP = FinvP*sigma1 + mu1
    BP = FinvP*sigma2 + mu2

    LomniBEL = 10.0*log10(pow(10.0, 0.1*AP) + pow(10.0, 0.1*BP) + pow(10.0, 0.1*C))
    return LomniBEL


_BLDG_COEFFS = np.array([
    [12.64, 3.72, 0.96, 9.6, 2.0, 9.1, -3.0, 4.5, -2.0],
    [28.19, -3.00, 8.48, 13.5, 3.8, 27.8, -2.9, 9.4, -2.1]
])


def FIGURE_1() -> None:
    """
    Calculates values and display FIGURE 1 from ITU-R P.2109-2.
    """
    import matplotlib.pyplot as plt
    freqs = [*np.arange(0.1, 1+0.01, 0.05), *np.arange(1, 10+0.1, 0.5), *np.arange(10, 100+0.1, 5)]
    losses = [[],[]]
    for f_GHz in freqs:
        losses[0].append(BuildingEntryLoss(f_GHz, 50, BuildingType.TRADITIONAL, 0))
        losses[1].append(BuildingEntryLoss(f_GHz, 50, BuildingType.THERMALLY_EFFICIENT, 0))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6.75)
    ax1.set_xscale('log')
    ax1.set_xlim([0.1,100])
    ax1.set_xticks([0.1,1,10,100])
    ax1.set_ylim([10, 60])
    ax1.set_yticks([*range(10, 60+1, 5)])
    ax1.plot(freqs, losses[0], color='#000000', label='Traditional')
    ax1.plot(freqs, losses[1], color='#FF0000', label='Thermally-efficient')
    ax1.set_title('FIGURE 1\nMedian building entry loss predicted at horizontal incidence')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('L_omni_BEL (dB)')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()


def FIGURE_2() -> None:
    """
    Calculates values and display FIGURE 2 from ITU-R P.2109-2.
    """
    import matplotlib.pyplot as plt
    freqs = [0.1, 1, 10, 100]
    probs = [*np.arange(0.001, 1, 0.001)]
    losses = [[],[],[],[],[],[],[],[]]
    for i, f_GHz in enumerate(freqs):
        for p in probs:
            losses[i*2].append(BuildingEntryLoss(f_GHz, p*100, BuildingType.TRADITIONAL, 0))
            losses[(i*2)+1].append(BuildingEntryLoss(f_GHz, p*100, BuildingType.THERMALLY_EFFICIENT, 0))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6.75)
    ax1.set_xlim([-20, 140])
    ax1.set_xticks([*range(-20, 140+1, 20)])
    ax1.set_ylim([0, 1])
    ax1.set_yticks([*np.arange(0, 1+0.01, 0.1)])
    ax1.plot(losses[0], probs, color='#000000', linestyle='dashed', label='0.1 GHz, Traditional')
    ax1.plot(losses[1], probs, color='#000000', linestyle='dotted', label='0.1 GHz, Thermally-efficient')
    ax1.plot(losses[2], probs, color='#FF0000', linestyle='dashed', label='  1 GHz, Traditional')
    ax1.plot(losses[3], probs, color='#FF0000', linestyle='dotted', label='  1 GHz, Thermally-efficient')
    ax1.plot(losses[4], probs, color='#00FF00', linestyle='dashed', label=' 10 GHz, Traditional')
    ax1.plot(losses[5], probs, color='#00FF00', linestyle='dotted', label=' 10 GHz, Thermally-efficient')
    ax1.plot(losses[6], probs, color='#0000FF', linestyle='dashed', label='100 GHz, Traditional')
    ax1.plot(losses[7], probs, color='#0000FF', linestyle='dotted', label='100 GHz, Thermally-efficient')
    ax1.set_title('FIGURE 2\nBuilding entry loss predicted at horizontal incidence')
    ax1.set_xlabel('L_omni_BEL (dB)')
    ax1.set_ylabel('Probability')
    ax1.legend()
    plt.grid(True, 'both','both')
    plt.show()
