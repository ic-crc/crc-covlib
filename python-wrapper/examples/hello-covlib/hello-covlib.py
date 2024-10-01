# The folowing two lines are to specify the location of the crc_covlib module,
# they are not required when the module is in the same folder as the python script using it.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('crc-covlib version {}'.format(covlib.__version__))

    sim = covlib.Simulation()

    print('Default transmitter height: {} m'.format(sim.GetTransmitterHeight()))
    sim.SetTransmitterHeight(30)
    print('New transmitter height: {} m\n'.format(sim.GetTransmitterHeight()))
