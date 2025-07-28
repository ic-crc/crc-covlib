# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - CRC-MLPL & CRC Path Obscura (machine learning-based path loss models)\n')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.536, -75.493)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Important note: CRC-MLPL and CRC Path Obscura usage requires BOTH terrain and surface elevation data.

    # Set terrain elevation data source
    HRDEM_DTM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM_DTM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSamplingResolution(10)

    # Set surface elevation data source
    HRDEM_DSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_HRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(HRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(HRDEM_DSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_HRDEM_DSM'))

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.515, -75.512, 45.557, -75.474)
    sim.SetReceptionAreaNumHorizontalPoints(120)
    sim.SetReceptionAreaNumVerticalPoints(120)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting coverage results using CRC-MLPL...\n')
    sim.SetPropagationModel(covlib.PropagationModel.CRC_MLPL)
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'crc-mlpl.bil'))

    print('Generating and exporting coverage results using CRC Path Obscura...\n')
    sim.SetPropagationModel(covlib.PropagationModel.CRC_PATH_OBSCURA)
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'crc-path-obscura.bil'))

    print('Simulation completed\n')
