# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib # crc-covlib core functionalities (C++)
from crc_covlib.helper import antennas      # additional functionalities (python only)


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R M.2101 (Section 5) - Beamforming antenna pattern')

    # Parameters for generating a multi-beam pattern
    phi_3dB = 65     # horizontal 3dB bandwidth of single element, in degrees
    theta_3dB = 65   # vertical 3dB bandwidth of single element, in degrees
    Am = 30          # front-to-back ratio, in dB
    SLAv = 30        # vertical sidelobe attenuation, in dB
    GEmax = 5        # maximum gain of single element, in dBi
    NH = 8           # number of columns in the array of elements
    NV = 8           # number of rows in the array of elements
    dH_over_wl = 0.5 # horizontal elements spacing over wavelength (dH/ʎ)
    dV_over_wl = 0.5 # vertical elements spacing over wavelength (dV/ʎ)
    # list of bearings (h angles) of formed beams, in degrees
    phi_escan_list =   [-15,   0,  15, -15, 0, 15, -15,  0, 15]
    # list of tilts (v angles) of formed beams, in degrees
    theta_etilt_list = [-10, -10, -10,   0, 0,  0,  10, 10, 10]


    sim = covlib.Simulation()
    TX = covlib.Terminal.TRANSMITTER

    # Generate an antenna pattern for the simulation's transmitter.
    print('Generating beamforming antenna pattern...')
    antennas.GenerateBeamformingAntennaPattern(sim, TX, phi_3dB, theta_3dB, Am, SLAv, GEmax, NH, NV, 
                                               dH_over_wl, dV_over_wl, phi_escan_list, theta_etilt_list)
    
    # Save the generated pattern to a file so it can be re-used in another simulation.
    antennas.SaveAs3DCsvFile(sim, TX, os.path.join(script_dir, 'composite_antenna_pattern.csv'))

    # The saved pattern may be loaded using the Load3DCsvFile function. This is faster than
    # re-generating the pattern each time.
    #antennas.Load3DCsvFile(sim, TX, os.path.join(script_dir, 'composite_antenna_pattern.csv'))

    # Available options to display the simulation's transmitter or receiver antenna pattern.
    antennas.PlotPolar(sim, TX)
    antennas.PlotCartesian(sim, TX)
    antennas.Plot3D(sim, TX)
    
    # ...continue as needed with the simulation here.

    print('Example completed\n')
