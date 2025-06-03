# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


def SetReceptionAreaParams(sim: covlib.Simulation) -> None:
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(120)
    sim.SetReceptionAreaNumVerticalPoints(120)


def SetSimulationParams(sim: covlib.Simulation, sampleResolution_m: float) -> None:
    # Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    sim.SetPropagationModel(covlib.PropagationModel.ITU_R_P_1812)

    # Specify file to get ITU radio climate zones from
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    sim.SetTerrainElevDataSamplingResolution(sampleResolution_m)

    # Set land cover data parameters
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    SetReceptionAreaParams(sim)


if __name__ == '__main__':

    print('\ncrc-covlib - Area results comparison')

    sim25m = covlib.Simulation()
    SetSimulationParams(sim25m, 25)

    sim50m = covlib.Simulation()
    SetSimulationParams(sim50m, 50)

    print('\nRunning simulation with 25 meters terrain sampling resolution...')
    sim25m.GenerateReceptionAreaResults()

    print('\nRunning simulation with 50 meters terrain sampling resolution...')
    sim50m.GenerateReceptionAreaResults()

    # The simsDiff simulation object will be used to hold comparison results (i.e. the difference) between the
    # two previous simulations
    simsDiff = covlib.Simulation()
    SetReceptionAreaParams(simsDiff)

    print('\nCalculating difference between simulations...')
    numHPoints = simsDiff.GetReceptionAreaNumHorizontalPoints()
    numVPoints = simsDiff.GetReceptionAreaNumVerticalPoints()
    for x in range(numHPoints):
        for y in range(numVPoints):
            diff = sim25m.GetReceptionAreaResultValue(x, y) - sim50m.GetReceptionAreaResultValue(x, y)
            simsDiff.SetReceptionAreaResultValue(x, y, diff)

    # Export simulation results and difference between simulations as raster files
    sim25m.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'sim25m.bil'))
    sim50m.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'sim50m.bil'))
    simsDiff.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'difference.bil'))

    # Export difference between simulations as vector file
    #   - shades of red when sim25m result is greater than corresponding sim50m result
    #   - shades of blue when sim25m result is smaller than corresponding sim50m result
    simsDiff.ClearCoverageDisplayFills()
    simsDiff.AddCoverageDisplayFill(0, 1, 0xFFD0D0)
    simsDiff.AddCoverageDisplayFill(1, 3, 0xFF9090)
    simsDiff.AddCoverageDisplayFill(3, 10, 0xFF4444)
    simsDiff.AddCoverageDisplayFill(10, 100, 0xFF0000)
    simsDiff.AddCoverageDisplayFill(0, -1, 0xD0D0FF)
    simsDiff.AddCoverageDisplayFill(-1, -3, 0x9090FF)
    simsDiff.AddCoverageDisplayFill(-3, -10, 0x4444FF)
    simsDiff.AddCoverageDisplayFill(-10, -100, 0x0000FF)
    simsDiff.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'difference.kml'), 50, 50, 'dB')

    print('\nSimulations completed\n')
