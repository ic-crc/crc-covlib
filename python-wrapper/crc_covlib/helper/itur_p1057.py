"""Implementation of ITU-R P.1057-7 (partial)
"""

from math import sqrt, log
from . import jit, COVLIB_NUMBA_CACHE


__all__ = ['Qinv',
           'Finv']


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Qinv(p: float) -> float:
    """
    ITU-R P.1057-7, Annex 1, Section 3
    Approximation for the inverse complementary cumulative distribution function, with 0 < p < 1.
    """
    a0 = 2.506628277459239
    a1 = -30.66479806614716
    a2 = 138.3577518672690
    a3 = -275.9285104469687
    a4 = 220.9460984245205
    a5 = -39.69683028665376
    b1 = -13.28068155288572
    b2 = 66.80131188771972
    b3 = -155.6989798598866
    b4 = 161.5858368580409
    b5 = -54.47609879822406
    c0 = 2.938163982698783
    c1 = 4.374664141464968
    c2 = -2.549732539343734
    c3 = -2.400758277161838
    c4 = -0.3223964580411365
    c5 = -0.007784894002430293
    d1 = 3.754408661907416
    d2 = 2.445134137142996
    d3 = 0.3224671290700398
    d4 = 0.007784695709041462
    pcopy = p

    if p > 0.5 and p < 1:
        p = 1.0 - p

    if p > 0 and p <= 0.02425:
        t = sqrt(-2.0*log(p))
        t2 =t*t
        t3 = t2*t
        t4 = t3*t
        t5 = t4*t
        Uinv = (c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5) / (1.0 + d1*t + d2*t2 + d3*t3 + d4*t4)
    elif p > 0.02425 and p <= 0.5:
        t = (p-0.5)*(p-0.5)
        t2 =t*t
        t3 = t2*t
        t4 = t3*t
        t5 = t4*t
        Uinv = (p-0.5) * (a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5) / (1.0 + b1*t + b2*t2 + b3*t3 + b4*t4 + b5*t5)
    else:
        raise ArithmeticError('Invalid input parameter p in Qinv()')

    if pcopy < 0.5:
        return -Uinv
    else:
        return Uinv


@jit(nopython=True, cache=COVLIB_NUMBA_CACHE)
def Finv(p: float) -> float:
    """
    ITU-R P.1057-7, Annex 1, Section 3
    Approximation for the inverse cumulative distribution function, with 0 < p < 1.
    """
    return Qinv(1.0-p)
