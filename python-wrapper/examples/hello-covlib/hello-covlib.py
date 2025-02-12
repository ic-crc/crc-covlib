# The following lines are to specify the location of the crc_covlib package,
# they are not required when the crc_covlib package is in the same folder as
# the python script using it.
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the full path to the directory containing this script (hello-covlib.py)
sys.path.insert(0, os.path.join(script_dir, '../../'))  # crc_covlib folder is located 2 folders up relatively to this script

from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('crc-covlib version {}'.format(covlib.__version__))

    sim = covlib.Simulation()

    print('Default transmitter height: {} m'.format(sim.GetTransmitterHeight()))
    sim.SetTransmitterHeight(30)
    print('New transmitter height: {} m\n'.format(sim.GetTransmitterHeight()))
