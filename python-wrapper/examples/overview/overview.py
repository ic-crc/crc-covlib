import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation


def printStatus(status):
    GEN = simulation.GenerateStatus
    if status == GEN.STATUS_OK:
        print('STATUS_OK')
    else:
        if (status & GEN.STATUS_SOME_ELEV_DATA_MISSING) != 0:
            print('STATUS_SOME_ELEV_DATA_MISSING')
        if (status & GEN.STATUS_NO_ELEV_DATA) != 0:
            print('STATUS_NO_ELEV_DATA')
        if (status & GEN.STATUS_SOME_LAND_COVER_DATA_MISSING) != 0:
            print('STATUS_SOME_LAND_COVER_DATA_MISSING')
        if (status & GEN.STATUS_NO_LAND_COVER_DATA) != 0:
            print('STATUS_NO_LAND_COVER_DATA')
        if (status & GEN.STATUS_NO_ITU_RCZ_DATA) != 0:
            print('STATUS_NO_ITU_RCZ_DATA')
        if (status & GEN.STATUS_SOME_ITU_RCZ_DATA_MISSING) != 0:
            print('STATUS_SOME_ITU_RCZ_DATA_MISSING')
        if (status & GEN.STATUS_NO_SURFACE_HEIGHT_DATA) != 0:
            print('STATUS_NO_SURFACE_HEIGHT_DATA')            
        if (status & GEN.STATUS_SOME_SURFACE_HEIGHT_DATA_MISSING) != 0:
            print('STATUS_SOME_SURFACE_HEIGHT_DATA_MISSING')


if __name__ == '__main__':

    # Note: the following example is to demonstrate usage of the different crc-covlib
    #       methods, used values are not necessarily relevant.

    sim = simulation.Simulation()

    # Modify and display transmitter parameters
    sim.SetTransmitterLocation(45.3, -75.9)
    print('TX lat: {}'.format(sim.GetTransmitterLatitude()))
    print('TX lon: {}'.format(sim.GetTransmitterLongitude()))
    sim.SetTransmitterHeight(30)
    print('TX height: {} m'.format(sim.GetTransmitterHeight()))
    sim.SetTransmitterFrequency(300)
    print('TX freq: {} MHz'.format(sim.GetTransmitterFrequency()))
    sim.SetTransmitterPower(1500, simulation.PowerType.EIRP)
    print('TX power (ERP): {:.2f} W'.format(sim.GetTransmitterPower(simulation.PowerType.ERP))) # may use different power type
    sim.SetTransmitterLosses(3)
    print('TX losses: {} dB'.format(sim.GetTransmitterLosses()))
    sim.SetTransmitterPolarization(simulation.Polarization.HORIZONTAL_POL)
    print('TX pol: {}'.format(sim.GetTransmitterPolarization()))


    # Modify and display receiver parameters
    sim.SetReceiverHeightAboveGround(2.0)
    print('\nRX height: {} m'.format(sim.GetReceiverHeightAboveGround()))
    sim.SetReceiverLosses(4)
    print('RX losses: {} dB'.format(sim.GetReceiverLosses()))


    # Modify and display antenna parameters
    TX = simulation.Terminal.TRANSMITTER
    sim.ClearAntennaPatterns(TX, True, True)
    sim.AddAntennaHorizontalPatternEntry(TX, 0, -10)
    sim.AddAntennaVerticalPatternEntry(TX, 0, 0, -10)
    sim.SetAntennaPatternApproximationMethod(TX, simulation.PatternApproximationMethod.SUMMING)
    print('\nTX antenna gain approximation method: {}'.format(sim.GetAntennaPatternApproximationMethod(TX)))
    print('TX antenna gain at (azm=90, elv=-3): {} dBi'.format(sim.GetAntennaGain(TX, 90, -3)))
    print('TX antenna horiz pattern normalization: {} dB adjusment'.format(sim.NormalizeAntennaHorizontalPattern(TX)))
    print('TX antenna vert pattern normalization: {} dB adjusment'.format(sim.NormalizeAntennaVerticalPattern(TX)))
    print('TX antenna gain at (azm=90, elv=-3): {} dBi'.format(sim.GetAntennaGain(TX, 90, -3)))
    sim.SetAntennaMaximumGain(TX, 16)
    print('TX antenna max gain: {} dBi'.format(sim.GetAntennaMaximumGain(TX)))
    sim.SetAntennaBearing(TX, simulation.BearingReference.TRUE_NORTH, 88)
    print('TX antenna bearing reference: {}'.format(sim.GetAntennaBearingReference(TX)))
    print('TX antenna bearing: {} degrees'.format(sim.GetAntennaBearing(TX)))
    sim.SetAntennaElectricalTilt(TX, -3.9)
    print('TX antenna electrical tilt: {} degrees'.format(sim.GetAntennaElectricalTilt(TX)))
    sim.SetAntennaMechanicalTilt(TX, -3.1, 0)
    print('TX antenna mechanical tilt: {} degrees'.format(sim.GetAntennaMechanicalTilt(TX)))
    print('TX antenna mechanical tilt applied at {} degree of azimuth in horizontal pattern'.format(sim.GetAntennaMechanicalTiltAzimuth(TX)))
    # Receiver antenna parameters can be modified similarly
    RX = simulation.Terminal.RECEIVER
    sim.SetAntennaMaximumGain(RX, 14)
    print('RX antenna max gain: {} dBi'.format(sim.GetAntennaMaximumGain(RX)))


    # Modify and display Longley-Rice propagation model parameters
    sim.SetLongleyRiceSurfaceRefractivity(303.0)
    print('\nLR surf refract: {} N-units'.format(sim.GetLongleyRiceSurfaceRefractivity()))
    sim.SetLongleyRiceGroundDielectricConst(16.0)
    print('LR grd diel const: {}'.format(sim.GetLongleyRiceGroundDielectricConst()))
    sim.SetLongleyRiceGroundConductivity(0.004)
    print('LR grd conduct: {} S/m'.format(sim.GetLongleyRiceGroundConductivity()))
    sim.SetLongleyRiceClimaticZone(simulation.LRClimaticZone.LR_EQUATORIAL)
    print('LR clim zone: {}'.format(sim.GetLongleyRiceClimaticZone()))
    sim.SetLongleyRiceActivePercentageSet(simulation.LRPercentageSet.LR_CONFIDENCE_RELIABILITY)
    print('LR active percentage set: {}'.format(sim.GetLongleyRiceActivePercentageSet()))
    sim.SetLongleyRiceTimePercentage(50.1)
    print('LR time: {} %'.format(sim.GetLongleyRiceTimePercentage()))
    sim.SetLongleyRiceLocationPercentage(50.2)
    print('LR location: {} %'.format(sim.GetLongleyRiceLocationPercentage()))
    sim.SetLongleyRiceSituationPercentage(50.3)
    print('LR situation: {} %'.format(sim.GetLongleyRiceSituationPercentage()))
    sim.SetLongleyRiceConfidencePercentage(50.4)
    print('LR confidence: {} %'.format(sim.GetLongleyRiceConfidencePercentage()))
    sim.SetLongleyRiceReliabilityPercentage(50.5)
    print('LR reliability: {} %'.format(sim.GetLongleyRiceReliabilityPercentage()))
    sim.SetLongleyRiceModeOfVariability(30)
    print('LR mode of variability id: {}'.format(sim.GetLongleyRiceModeOfVariability()))
    mdVar = simulation.LRModeOfVariability
    sim.SetLongleyRiceModeOfVariability(mdVar.LR_BROADCAST_MODE + mdVar.LR_ELIMINATE_LOCATION_VARIABILITY + mdVar.LR_ELIMINATE_SITUATION_VARIABILITY)
    print('LR mode of variability id: {}'.format(sim.GetLongleyRiceModeOfVariability()))


    # Modify and display ITU-R P.1812 propagation model parameters
    sim.SetITURP1812TimePercentage(44.0)
    print('\nP1812 time percent: {} %'.format(sim.GetITURP1812TimePercentage()))
    sim.SetITURP1812LocationPercentage(45.0)
    print('P1812 loc percent: {} %'.format(sim.GetITURP1812LocationPercentage()))
    sim.SetITURP1812PredictionResolution(72.0)
    sim.SetITURP1812AverageRadioRefractivityLapseRate(44)
    print('P1812 avg radio refractivity lapse-rate: {} N-units/km'.format(sim.GetITURP1812AverageRadioRefractivityLapseRate()))
    sim.SetITURP1812SeaLevelSurfaceRefractivity(333)
    print('P1812 sea level surface refractivitye: {} N-units'.format(sim.GetITURP1812SeaLevelSurfaceRefractivity()))
    print('P1812 pred res: {} m'.format(sim.GetITURP1812PredictionResolution()))
    CLUT1812 = simulation.P1812ClutterCategory
    sim.SetITURP1812RepresentativeClutterHeight(CLUT1812.P1812_DENSE_URBAN, 24.0)
    print('P1812 dense urban repr height: {} m'.format(sim.GetITURP1812RepresentativeClutterHeight(CLUT1812.P1812_DENSE_URBAN)))
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))
    print('P1812 radio climatic zones file: {}'.format(sim.GetITURP1812RadioClimaticZonesFile()))
    sim.SetITURP1812LandCoverMappingType(simulation.P1812LandCoverMappingType.P1812_MAP_TO_CLUTTER_CATEGORY)
    print('P1812 land cover mapping type: {}'.format(sim.GetITURP1812LandCoverMappingType()))
    sim.SetITURP1812SurfaceProfileMethod(simulation.P1812SurfaceProfileMethod.P1812_ADD_REPR_CLUTTER_HEIGHT)
    print('P1812 surface profile method: {}'.format(sim.GetITURP1812SurfaceProfileMethod()))
    

    # Modify and display ITU-R P.452 propagation model parameters
    sim.SetITURP452TimePercentage(24.0)
    print('\nP452 time percent: {} %'.format(sim.GetITURP452TimePercentage()))
    sim.SetITURP452PredictionType(simulation.P452PredictionType.P452_WORST_MONTH)
    print('P452 prediction type: {}'.format(sim.GetITURP452PredictionType()))
    sim.SetITURP452AverageRadioRefractivityLapseRate(45.2)
    print('P452 avg radio refractivity lapse-rate: {} N-units/km'.format(sim.GetITURP452AverageRadioRefractivityLapseRate()))
    sim.SetITURP452SeaLevelSurfaceRefractivity(313)
    print('P452 sea level surface refractivitye: {} N-units'.format(sim.GetITURP452SeaLevelSurfaceRefractivity()))
    sim.SetITURP452AirTemperature(15)
    print('P452 air temperature: {} C'.format(sim.GetITURP452AirTemperature()))
    sim.SetITURP452AirPressure(1013.25)
    print('P452 air pressure: {} hPa'.format(sim.GetITURP452AirPressure()))
    sim.SetITURP452RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))
    print('P452 radio climatic zones file: {}'.format(sim.GetITURP452RadioClimaticZonesFile()))
    CLUT452HGM = simulation.P452HeightGainModelClutterCategory
    NOMINAL_HEIGHT_M = simulation.P452HeightGainModelClutterParam.P452_NOMINAL_HEIGHT_M
    sim.SetITURP452HeightGainModelClutterValue(CLUT452HGM.P452_HGM_HIGH_CROP_FIELDS, NOMINAL_HEIGHT_M, 4.4)
    print('P452-17 nominal clutter height for P452_HGM_HIGH_CROP_FIELDS: {} m'.format(sim.GetITURP452HeightGainModelClutterValue(CLUT452HGM.P452_HGM_HIGH_CROP_FIELDS, NOMINAL_HEIGHT_M)))
    sim.SetITURP452HeightGainModelMode(TX, simulation.P452HeightGainModelMode.P452_USE_CUSTOM_AT_CATEGORY)
    sim.SetITURP452HeightGainModelMode(RX, simulation.P452HeightGainModelMode.P452_USE_CLUTTER_PROFILE)
    print('P452-17 height gain model mode at tx, rx: {}, {}'.format(sim.GetITURP452HeightGainModelMode(TX), sim.GetITURP452HeightGainModelMode(RX)))
    CLUT452 = simulation.P452ClutterCategory
    sim.SetITURP452RepresentativeClutterHeight(CLUT452.P452_SUBURBAN, 12.0)
    print('P452-18 dense urban repr height: {} m'.format(sim.GetITURP452RepresentativeClutterHeight(CLUT452.P452_SUBURBAN)))
    sim.SetITURP452LandCoverMappingType(simulation.P452LandCoverMappingType.P452_MAP_TO_REPR_CLUTTER_HEIGHT)
    print('P452-18 land cover mapping type: {}'.format(sim.GetITURP452LandCoverMappingType()))
    sim.SetITURP452SurfaceProfileMethod(simulation.P452SurfaceProfileMethod.P452_ADD_REPR_CLUTTER_HEIGHT)
    print('P452-18 surface profile method: {}'.format(sim.GetITURP452SurfaceProfileMethod()))


    # Modify and display Extended Hata propagation model parameters
    sim.SetEHataClutterEnvironment(simulation.EHataClutterEnvironment.EHATA_SUBURBAN)
    print('\neHata clutter env: {}'.format(sim.GetEHataClutterEnvironment()))
    sim.SetEHataReliabilityPercentage(95)
    print('eHata reliability percent: {} %'.format(sim.GetEHataReliabilityPercentage()))


    # Modify propagation model selection (i.e. the model used when generating results)
    P1812 = simulation.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)
    print('\nPropag model set to: {}'.format(sim.GetPropagationModel()))
    LR = simulation.PropagationModel.LONGLEY_RICE
    sim.SetPropagationModel(LR)
    print('Propag model set to: {}'.format(sim.GetPropagationModel()))


    # Modify and display ITU-R P.2108 clutter loss model parameters
    sim.SetITURP2108TerrestrialStatModelActiveState(True)
    print('\nP2108 active state: {}'.format(sim.GetITURP2108TerrestrialStatModelActiveState()))
    sim.SetITURP2108TerrestrialStatModelLocationPercentage(90)
    print('P2108 location percent: {} %'.format(sim.GetITURP2108TerrestrialStatModelLocationPercentage()))
    print('P2108 losses at 3 GHz, 1.12 km, 90% loc: {:.2f} dB'.format(sim.GetITURP2108TerrestrialStatModelLoss(3, 1.12)))
    sim.SetITURP2108TerrestrialStatModelActiveState(False)


    # Modify and display ITU-R P.2109 building entry loss model parameters
    sim.SetITURP2109ActiveState(True)
    print('\nP2109 active state: {}'.format(sim.GetITURP2109ActiveState()))
    sim.SetITURP2109Probability(24)
    print('P2109 loss not exceeded probability: {} %'.format(sim.GetITURP2109Probability()))
    sim.SetITURP2109DefaultBuildingType(simulation.P2109BuildingType.P2109_THERMALLY_EFFICIENT)
    print('P2109 default building type: {}'.format(sim.GetITURP2109DefaultBuildingType()))
    print('P2109 bldg entry losses at 4 GHz, 24% prob., therm. efficent bldgs, 10 deg elevation angle: {:.2f} dB'.format(sim.GetITURP2109BuildingEntryLoss(4, 10.0)))
    sim.SetITURP2109ActiveState(False)


    # Modify and display ITU-R P.676 gaseous attenuation model for terrestrial paths
    sim.SetITURP676TerrPathGaseousAttenuationActiveState(True, simulation.AUTOMATIC, simulation.AUTOMATIC, 7.5)
    print('\nP676 active state: {}'.format(sim.GetITURP676TerrPathGaseousAttenuationActiveState()))
    print('P676 Gaseous attenuation at {:.1f} GHz: {:.2f} dB/km'.format(60, sim.GetITURP676GaseousAttenuation(60, 1013.25, 15, 7.5)))
    sim.SetITURP676TerrPathGaseousAttenuationActiveState(False)


    # Get values from some ITU digital maps
    print('\nValues of some ITU digital maps at (lat={:.1f}, lon={:.1f}):'.format(45,-75))
    print('Average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere: {:.2f} km^-1'.format(sim.GetITUDigitalMapValue(simulation.ITUDigitalMap.ITU_MAP_DN50, 45, -75)))
    print('Sea-level surface refractivity: {:.2f}'.format(sim.GetITUDigitalMapValue(simulation.ITUDigitalMap.ITU_MAP_N050, 45, -75)))
    print('Annual mean surface temperature at 2 meters above the surface of the Earth: {:.2f} K'.format(sim.GetITUDigitalMapValue(simulation.ITUDigitalMap.ITU_MAP_N050, 45, -75)))
    print('Surface water-vapour density under non-rain conditions, exceeded for 50% of an average year: {:.2f} g/m^3'.format(sim.GetITUDigitalMapValue(simulation.ITUDigitalMap.ITU_MAP_SURFWV_50, 45, -75)))


    # Modify and display terrain elevation data parameters
    sim.SetTerrainElevDataSamplingResolution(50)
    print('\nElev sampling res: {} m'.format(sim.GetTerrainElevDataSamplingResolution()))
    # NRCan HRDEM (primary source)
    HRDEM_DTM = simulation.TerrainElevDataSource.TERR_ELEV_NRCAN_HRDEM_DTM
    sim.SetPrimaryTerrainElevDataSource(HRDEM_DTM)
    sim.SetTerrainElevDataSourceDirectory(HRDEM_DTM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_HRDEM_DTM'))
    sim.SetTerrainElevDataSourceSamplingMethod(HRDEM_DTM, simulation.SamplingMethod.NEAREST_NEIGHBOR)
    print('Elev 1st src type: {}'.format(sim.GetPrimaryTerrainElevDataSource()))
    print('  HRDEM_DTM dir: {}'.format(sim.GetTerrainElevDataSourceDirectory(HRDEM_DTM)))
    print('  HRDEM_DTM Sampling method: {}'.format(sim.GetTerrainElevDataSourceSamplingMethod(HRDEM_DTM)))
    # NRCan CDEM (secondary source: used when no terrain elev data available from primary source)
    CDEM = simulation.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim. SetSecondaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    print('Elev 2nd src type: {}'.format(sim.GetSecondaryTerrainElevDataSource()))
    print('  CDEM dir: {}'.format(sim.GetTerrainElevDataSourceDirectory(CDEM)))
    print('  CDEM Sampling method: {}'.format(sim.GetTerrainElevDataSourceSamplingMethod(CDEM)))
    print('Elev at (45.4,-75.7): {:.2f} m'.format(sim.GetTerrainElevation(45.4, -75.7)))
    # Custom terrain elevation data...
    sim.SetPrimaryTerrainElevDataSource(simulation.TerrainElevDataSource.TERR_ELEV_CUSTOM)
    sim.SetSecondaryTerrainElevDataSource(simulation.TerrainElevDataSource.TERR_ELEV_NONE)
    sim.SetTertiaryTerrainElevDataSource(simulation.TerrainElevDataSource.TERR_ELEV_NONE)
    print('Elev src type: {}'.format(sim.GetPrimaryTerrainElevDataSource()))
    elevData = [30, 36, 32, 34]
    sim.AddCustomTerrainElevData(44.0, -76.0, 45.0, -75.0, 2, 2, elevData)
    elevData = None # Note: terrain elev data has been copied, no requirements to keep it any longer
    print('Elev at (44.5,-75.5): {:.2f} m'.format(sim.GetTerrainElevation(44.5, -75.5)))
    

    # Modify and display land cover data parameters
    landCoverData = [300, 301, 302, 303]
    sim.AddCustomLandCoverData(44.0, -76.0, 45.0, -75.0, 2, 2, landCoverData)
    landCoverData = None # Note: land cover data has been copied, no requirements to keep it any longer
    sim.SetPrimaryLandCoverDataSource(simulation.LandCoverDataSource.LAND_COVER_CUSTOM)
    print('\n(Custom) land cover class at (44.9, -75.1): {}'.format(sim.GetLandCoverClass(44.9, -75.1)))
    # ...mappings with ITU-R P.1812 in this example
    WORLDCOVER = simulation.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))
    print('Land cover 1st src type: {}'.format(sim.GetPrimaryLandCoverDataSource()))
    print('  WORLDCOVER dir: {}'.format(sim.GetLandCoverDataSourceDirectory(WORLDCOVER)))
    print('Land cover classification id (from WORLDCOVER) at (45.43, -75.71): {}'.format(sim.GetLandCoverClass(45.43,-75.71)))
    print('Mapped land cover value (WORLDCOVER --> P1812) at (45.43, -75.71): {}'.format(sim.GetLandCoverClassMappedValue(45.43,-75.71, P1812)))
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P1812, CLUT1812.P1812_SUBURBAN)
    print('New mapping for WORLDCOVER classification id 50: {}'.format(sim.GetLandCoverClassMapping(WORLDCOVER, 50, P1812)))
    print('Mapped land cover value (WORLDCOVER --> P1812) at (45.43, -75.71): {}'.format(sim.GetLandCoverClassMappedValue(45.43,-75.71, P1812)))
    print('Deleting mapping table between WORLDCOVER and P1812')
    sim.ClearLandCoverClassMappings(WORLDCOVER, P1812)
    print('Mapped land cover value (WORLDCOVER --> P1812) at (45.43, -75.71): {}'.format(sim.GetLandCoverClassMappedValue(45.43,-75.71, P1812)))
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P1812, CLUT1812.P1812_OPEN_RURAL)
    print('Setting default mapping (i.e. mapping for all classifications unless otherwise specified) between WORLDCOVER and P1812 to: {}'.format(sim.GetDefaultLandCoverClassMapping(WORLDCOVER, P1812)))
    print('Mapped land cover value (WORLDCOVER --> P1812) at (45.43, -75.71): {}'.format(sim.GetLandCoverClassMappedValue(45.43,-75.71, P1812)))
    sim.SetITURP1812LandCoverMappingType(simulation.P1812LandCoverMappingType.P1812_MAP_TO_REPR_CLUTTER_HEIGHT)
    print('Changed ITU-R P.1812 land cover mapping type to {}'.format(sim.GetITURP1812LandCoverMappingType()))
    sim.SetITURP1812LandCoverMappingType(simulation.P1812LandCoverMappingType.P1812_MAP_TO_CLUTTER_CATEGORY)
    print('Changed ITU-R P.1812 land cover mapping type back to its default value of {}'.format(sim.GetITURP1812LandCoverMappingType()))


    # Modify and display surface elevation data parameters
    surfaceElevData = [400, 401, 402, 403]
    sim.AddCustomSurfaceElevData(44.0, -76.0, 45.0, -75.0, 2, 2, surfaceElevData)
    surfaceElevData = None # Note: surface elevation data has been copied, no requirements to keep it any longer
    sim.SetPrimarySurfaceElevDataSource(simulation.SurfaceElevDataSource.SURF_ELEV_CUSTOM)
    print('\nSurface elevation at (44.9, -75.1): {:.2f} m'.format(sim.GetSurfaceElevation(44.9, -75.1)))
    sim.SetSurfaceAndTerrainDataSourcePairing(False)
    print('Surface and terrain elevation pairing?: {}'.format(sim.GetSurfaceAndTerrainDataSourcePairing()))
    # NRCan HRDEM (primary source)
    HRDEM_DSM = simulation.SurfaceElevDataSource.SURF_ELEV_NRCAN_HRDEM_DSM
    sim.SetPrimarySurfaceElevDataSource(HRDEM_DSM)
    sim.SetSurfaceElevDataSourceDirectory(HRDEM_DSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_HRDEM_DSM'))
    sim.SetSurfaceElevDataSourceSamplingMethod(HRDEM_DSM, simulation.SamplingMethod.NEAREST_NEIGHBOR)
    print('Surf 1st src type: {}'.format(sim.GetPrimarySurfaceElevDataSource()))
    print('  HRDEM_DSM dir: {}'.format(sim.GetSurfaceElevDataSourceDirectory(HRDEM_DSM)))
    print('  HRDEM_DSM Sampling method: {}'.format(sim.GetSurfaceElevDataSourceSamplingMethod(HRDEM_DSM)))
    # NRCan CDDM (secondary source: used when no surface elevation data is available from primary source)
    CDSM = simulation.SurfaceElevDataSource.SURF_ELEV_NRCAN_CDSM
    sim.SetSecondarySurfaceElevDataSource(CDSM)
    sim.SetSurfaceElevDataSourceDirectory(CDSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_CDSM'))
    print('Surf 2nd src type: {}'.format(sim.GetSecondarySurfaceElevDataSource()))
    print('  CDSM dir: {}'.format(sim.GetSurfaceElevDataSourceDirectory(CDSM)))
    print('  CDSM Sampling method: {}'.format(sim.GetSurfaceElevDataSourceSamplingMethod(CDSM)))
    # SRTM (tertiary source: used when no surface elevation data is available from both primary and secondary sources)
    SRTM = simulation.SurfaceElevDataSource.SURF_ELEV_SRTM
    sim.SetTertiarySurfaceElevDataSource(SRTM)
    sim.SetSurfaceElevDataSourceDirectory(SRTM, os.path.join(script_dir, '../../../data/surface-elev-samples/SRTMGL30'))
    print('Surf 3rd src type: {}'.format(sim.GetTertiarySurfaceElevDataSource()))
    print('  SRTM dir: {}'.format(sim.GetSurfaceElevDataSourceDirectory(SRTM)))
    print('  SRTM Sampling method: {}'.format(sim.GetSurfaceElevDataSourceSamplingMethod(SRTM)))
    print('Surface elevation at (45.4,-75.7): {:.2f} m'.format(sim.GetSurfaceElevation(45.4, -75.7)))


    # Modify and display reception area parameters
    sim.SetReceptionAreaCorners(45.02875000, -76.0, 45.52791667, -75.24104167)
    print('\nRX area LLC: ({:.6f}, {:.6f})'.format(sim.GetReceptionAreaLowerLeftCornerLatitude(), sim.GetReceptionAreaLowerLeftCornerLongitude()))
    print('RX area URC: ({:.6f}, {:.6f})'.format(sim.GetReceptionAreaUpperRightCornerLatitude(), sim.GetReceptionAreaUpperRightCornerLongitude()))
    sim.SetReceptionAreaNumHorizontalPoints(120)
    print('RX area num horiz pts: {}'.format(sim.GetReceptionAreaNumHorizontalPoints()))
    sim.SetReceptionAreaNumVerticalPoints(100)
    print('RX area num vert pts: {}'.format(sim.GetReceptionAreaNumVerticalPoints()))


    # Modify and display result type parameters
    sim.SetResultType(simulation.ResultType.PATH_LOSS_DB)
    print('\nResult type: {}'.format(sim.GetResultType()))


    # Modify and display coverage display parameters for vector files (.mif and .kml)
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(0, 110, simulation.RGBtoInt(255, 0, 0))
    sim.AddCoverageDisplayFill(110, 145, simulation.RGBtoInt(255, 70, 70))
    sim.AddCoverageDisplayFill(145, 200, simulation.RGBtoInt(255, 128, 128))
    print('\nMIF/KML display params:')
    num_fills = sim.GetCoverageDisplayNumFills()
    for i in range(num_fills):
        print('  {} to {} : RGB{}'.format(sim.GetCoverageDisplayFillFromValue(i),
                                          sim.GetCoverageDisplayFillToValue(i),
                                          simulation.IntToRGB(sim.GetCoverageDisplayFillColor(i))))


    # Generating results using user-defined terrain profiles
    rx_lat = sim.GetTransmitterLatitude() + 0.002
    rx_lon = sim.GetTransmitterLongitude() + 0.002
    elevProfile = [33, 10, 10, 10, 16]
    print('\nResult with a custom terrain elev profile: {:.2f} dB'.format(sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(elevProfile), elevProfile)))
    elevProfile2 = [33, 10, 1000, 10, 16]
    print('With huge obstacle in middle of profile: {:.2f} dB'.format(sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(elevProfile2), elevProfile2)))
    sim.SetPropagationModel(P1812)
    print('ITU-R P.1812 with custom terrain elev profile: {:.2f} dB'.format(sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(elevProfile), elevProfile)))
    clutterCategoryProfile = [5, 5, 5, 5, 5]
    print('ITU-R P.1812 with custom terrain elev and clutter profiles: {:.2f} dB'.format(sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(elevProfile), elevProfile, clutterCategoryProfile)))
    print('ITU-R P.1812 with custom clutter category profile only: {:.2f} dB'.format(sim.GenerateProfileReceptionPointResult(rx_lat, rx_lon, len(clutterCategoryProfile), None, clutterCategoryProfile)))
    sim.SetPropagationModel(LR)
    

    # Generating results using the NRCAN CDEM
    print('\nSingle point simulation')
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    print('Result at ({:.6f}, {:.6f}): {:.2f}  dB'.format(rx_lat, rx_lon, sim.GenerateReceptionPointResult(rx_lat, rx_lon)))
    printStatus(sim.GetGenerateStatus())
    sim.ExportProfilesToCsvFile(os.path.join(script_dir, './profiles.csv'), rx_lat, rx_lon)
    print('Terrain and result profiles have been exported to .csv file for reception point at (lat={:.6f}, lon={:.6f})'.format(rx_lat, rx_lon))
    print('\nArea simulation')
    sim.GenerateReceptionAreaResults()
    printStatus(sim.GetGenerateStatus())
    print('RX area result at (x=60, y=50): {:.2f} dB'.format(sim.GetReceptionAreaResultValue(60, 50)))
    rx_lat = sim.GetReceptionAreaResultLatitude(60, 50)
    print('Rx area latitude at (x=60, y=50): {:.6f}'.format(rx_lat))
    rx_lon = sim.GetReceptionAreaResultLongitude(60, 50)
    print('Rx area longitude at (x=60, y=50): {:.6f}'.format(rx_lon))
    print('Rx area result at (lat={:.6f}, lon={:.6f}): {:.2f}  dB'.format(rx_lat, rx_lon, sim.GetReceptionAreaResultValueAtLatLon(rx_lat, rx_lon)))
    sim.SetReceptionAreaResultValue(60, 50, 144.0)
    print('Changing Rx area result at (x=60, y=50) to: {}  dB'.format(sim.GetReceptionAreaResultValue(60, 50)))
    sim.ExportReceptionAreaResultsToTextFile(os.path.join(script_dir, 'sim.txt'))
    sim.ExportReceptionAreaResultsToMifFile(os.path.join(script_dir, 'sim.mif'))
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'sim.kml'))
    sim.ExportReceptionAreaResultsToBilFile(os.path.join(script_dir, 'sim.bil'))
    print('\nRx area simulation results have been exported to .kml, .mif, .bil and .txt files')

    sim.ExportReceptionAreaTerrainElevationToBilFile(os.path.join(script_dir, 'terrain-elev.bil'), 1000, 1000)
    print('\nTerrain elevation data has been exported to .bil file')

    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))
    sim.ExportReceptionAreaLandCoverClassesToBilFile(os.path.join(script_dir, 'terrain-land-cover.bil'), 1000, 1000, False)
    print('\nLand cover data has been exported to .bil file')

    sim.ExportReceptionAreaSurfaceElevationToBilFile(os.path.join(script_dir, 'surface-elev.bil'), 1000, 1000)
    print('\nSurface elevation data has been exported to .bil file')

    # Note: do not call sim.Release() explicitely, it will be called from the destructor of the sim object

    print('\n')