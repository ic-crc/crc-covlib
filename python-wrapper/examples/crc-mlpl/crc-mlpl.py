import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - CRC-MLPL (machine learning-based path loss model)\n')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.536, -75.493)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    sim.SetPropagationModel(covlib.PropagationModel.CRC_MLPL)

    # Important note: CRC-MLPL usage requires BOTH terrain and surface elevation data.

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
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting coverage results...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'crc-mlpl.bil'))

    print('Simulation completed\n')
