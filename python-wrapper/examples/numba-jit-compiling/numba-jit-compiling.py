# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
import time

from crc_covlib.helper import EnableNumbaJIT
EnableNumbaJIT(onDiskCaching=False) # call before imports from crc_covlib.helper

from crc_covlib.helper import itur_p676


if __name__ == '__main__':
    # "Numba translates Python functions to optimized machine code at runtime using the industry-
    # standard LLVM compiler library." (https://numba.pydata.org/)
    #
    # Many functions under the crc_covlib.helper sub-package can be compiled using Numba. Numba
    # compilation is disabled by default in crc-covlib, you may enable it by calling EnableNumbaJIT()
    # before loading modules from crc_covlib.helper. Not all functions will benefit from enabling 
    # Numba though. One example of a function that much benefits from it is SlantPathGaseousAttenuation()
    # from the itur_p676 module.

    print('\ncrc-covlib - Numba JIT compiling')

    # The first time the function is called, it needs to be compiled, which requires additional time.
    start = time.perf_counter()
    Agas, bending, deltaL = itur_p676.SlantPathGaseousAttenuation(f_GHz=1, h1_km=0.0, h2_km=100.01, phi1_deg=90.0)
    end = time.perf_counter()
    print("Elapsed (first call) = {:.4f} secs".format(end-start))

    # Subsequent calls complete much faster. Run this script a second time with EnableNumbaJIT() commented out
    # to see the impact of using and not using Numba in this specific scenario.
    start = time.perf_counter()
    count = 0
    for f_GHz in range(1, 1001, 5):
        Agas, bending, deltaL = itur_p676.SlantPathGaseousAttenuation(f_GHz=f_GHz, h1_km=0.0, h2_km=100.01, phi1_deg=90.0)
        count += 1
    end = time.perf_counter()
    print("Elapsed average = {:.4f} secs/call".format((end - start)/count))
