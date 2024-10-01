import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib


if __name__ == '__main__':

    print('\ncrc-covlib - ITU-R P. 452-17 propagation model')

    sim = covlib.Simulation()

    # Set transmitter parameters (i.e. interfering station's parameters)
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters (i.e. interfered-with station's parameters)
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    P452_17 = covlib.PropagationModel.ITU_R_P_452_V17
    sim.SetPropagationModel(P452_17)

    # Set ITU-R P.452-17 propagation model parameters
    sim.SetITURP452TimePercentage(50)
    sim.SetITURP452PredictionType(covlib.P452PredictionType.P452_AVERAGE_YEAR)
    sim.SetITURP452AverageRadioRefractivityLapseRate(covlib.AUTOMATIC) # use ITU digital map (DN50.TXT)
    sim.SetITURP452SeaLevelSurfaceRefractivity(covlib.AUTOMATIC) # use ITU digital map (N050.TXT)
    sim.SetITURP452AirTemperature(covlib.AUTOMATIC) # use annual mean surface temperature from ITU-R P.1510 (T_Annual.TXT)
    sim.SetITURP452AirPressure(covlib.AUTOMATIC) # use mean annual global reference atmosphere pressure from ITU-R P.835

    # When desired, nominal heights and distances from TABLE 4 of ITU-R P.452-17 can be modified this way:
    CLUT = covlib.P452HeightGainModelClutterCategory
    NOMINAL_HEIGHT_M = covlib.P452HeightGainModelClutterParam.P452_NOMINAL_HEIGHT_M
    NOMINAL_DIST_KM = covlib.P452HeightGainModelClutterParam.P452_NOMINAL_DISTANCE_KM
    sim.SetITURP452HeightGainModelClutterValue(CLUT.P452_HGM_HIGH_CROP_FIELDS, NOMINAL_HEIGHT_M, 4.1)
    sim.SetITURP452HeightGainModelClutterValue(CLUT.P452_HGM_HIGH_CROP_FIELDS, NOMINAL_DIST_KM, 0.11)

    # Specify how to handle additional clutter losses (ITU-R P.452-17, Section 4.5) for the height-gain model.
    # Clutter losses may be handled differently at the tx (interferer) and rx (interfered-with) locations.
    # For each location, you may choose between one of these:
    #   P452_NO_SHIELDING:            No losses (i.e. not using the additional clutter losses procedure).
    #   P452_USE_CUSTOM_AT_CATEGORY:  Use nominal height and distance from the P452_HGM_CUSTOM_AT_TRANSMITTER or
    #                                 P452_HGM_CUSTOM_AT_RECEIVER clutter category to compute losses.
    #   P452_USE_CLUTTER_PROFILE:     Using specified land cover data source(s) and mappings, get the clutter category
    #                                 with the highest nominal height within 100m of the station along the transmission
    #                                 path. Use nominal height and distance for that category to calculate losses.
    #   P452_USE_CLUTTER_AT_ENDPOINT: Using specified land cover data source(s) and mappings, get the clutter category
    #                                 at the station's location. Use nominal height and distance for that category to
    #                                 calculate losses.
    sim.SetITURP452HeightGainModelMode(covlib.Terminal.TRANSMITTER, covlib.P452HeightGainModelMode.P452_USE_CUSTOM_AT_CATEGORY)
    sim.SetITURP452HeightGainModelMode(covlib.Terminal.RECEIVER, covlib.P452HeightGainModelMode.P452_USE_CLUTTER_PROFILE)

    # Specify the nominal clutter height and distance to use at the transmitter (interferer) location for the height-gain model.
    sim.SetITURP452HeightGainModelClutterValue(CLUT.P452_HGM_CUSTOM_AT_TRANSMITTER, NOMINAL_HEIGHT_M, 7.0)
    sim.SetITURP452HeightGainModelClutterValue(CLUT.P452_HGM_CUSTOM_AT_TRANSMITTER, NOMINAL_DIST_KM, 0.33)

    # Specify the land cover data source that will be used to determine the clutter category at the receiver (interfered-with) location.
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER # data from https://esa-worldcover.org/
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    # Some default mappings between ESA WorldCover's land cover classes and ITU-R P.452-17's clutter categories (TABLE 4)
    # are already defined in crc-covlib, however these mappings may be modified when desired. For example:
    sim.ClearLandCoverClassMappings(WORLDCOVER, P452_17); # delete existing default mappings
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P452_17, CLUT.P452_HGM_MIXED_TREE_FOREST) # map 'Tree cover' (10) to 'Mixed tree forest'
    sim.SetLandCoverClassMapping(WORLDCOVER, 20, P452_17, CLUT.P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES) # map 'shrubland'(20) to 'Irregularly spaced sparse trees'
    sim.SetLandCoverClassMapping(WORLDCOVER, 40, P452_17, CLUT.P452_HGM_HIGH_CROP_FIELDS)  # map 'cropland' (40) to 'High crop fields'
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P452_17, CLUT.P452_HGM_SUBURBAN) # map 'built-up' (50) to 'Suburban'
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P452_17, CLUT.P452_HGM_OTHER) # map all other ESA WorldCover classes to 'other' (i.e. nominal height and distance of 0)
    # See https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf for a definition of ESA WorldCover classes.

    # Specify file to get ITU radio climate zones from. When not specified, "inland" zone is assumed everywhere. 
    sim.SetITURP452RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    sim.SetTerrainElevDataSamplingResolution(25) # Note: this sampling resolution also applies to landcover/clutter data.

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    print('Generating and exporting coverage results ...\n')
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'iturp452v17.bil'))

    print('Simulation completed\n')
