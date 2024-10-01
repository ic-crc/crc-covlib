import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R P. 1812 propagation model - using surface elevation data')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.536, -75.493)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)

    # Set ITU-R P.1812 propagation model parameters
    sim.SetITURP1812TimePercentage(50)
    sim.SetITURP1812LocationPercentage(50)
    sim.SetITURP1812AverageRadioRefractivityLapseRate(covlib.AUTOMATIC) # use ITU digital map (DN50.TXT)
    sim.SetITURP1812SeaLevelSurfaceRefractivity(covlib.AUTOMATIC) # use ITU digital map (N050.TXT)
    sim.SetITURP1812PredictionResolution(100) # Width (in meters) of the square area over which the variability applies (see Annex 1, Section 4.7 of ITU-R P.1812 recommendation)
    sim.SetITURP1812SurfaceProfileMethod(covlib.P1812SurfaceProfileMethod.P1812_USE_SURFACE_ELEV_DATA) # using surface elevation rather than clutter data

    # Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    HRDEM_DTM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM_DTM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSamplingResolution(10)

    # Set surface elevation data parameters (see Annex 1, section 3.2.2 of ITU-R P.1812-7)
    HRDEM_DSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_HRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(HRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(HRDEM_DSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_HRDEM_DSM'))

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.515, -75.512, 45.557, -75.474)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp1812-surface.bil'))

    print('Simulations completed\n')
