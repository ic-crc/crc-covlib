import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib



def LoadRadioMobileV3File(sim: covlib.Simulation, terminal: covlib.Terminal, pathname: str) -> None:
    sim.ClearAntennaPatterns(terminal, True, True)

    with open(pathname, 'r', encoding='UTF-8') as f:
        for azm in range(0, 360):
            sim.AddAntennaHorizontalPatternEntry(terminal, azm, float(f.readline().strip()))

        gain_dB = float(f.readline().strip())
        sim.AddAntennaVerticalPatternEntry(terminal, 0, -90, gain_dB)
        sim.AddAntennaVerticalPatternEntry(terminal, 180, -90, gain_dB)
        for elv in range(-89, 90):
            sim.AddAntennaVerticalPatternEntry(terminal, 0, elv, float(f.readline().strip()))

        gain_dB = float(f.readline().strip())
        sim.AddAntennaVerticalPatternEntry(terminal, 0, 90, gain_dB)
        sim.AddAntennaVerticalPatternEntry(terminal, 180, 90, gain_dB)
        for elv in range(89, -90, -1):
            sim.AddAntennaVerticalPatternEntry(terminal, 180, elv, float(f.readline().strip()))


if __name__ == '__main__':

    print('\ncrc-covlib - antenna pattern usage\n')

    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(2600)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.0)

    # Set antenna parameters (at transmitter):
    # Load antenna pattern file
    TX = covlib.Terminal.TRANSMITTER
    # Have a look at LoadRadioMobileV3File's implementation above for example usage of the
    # AddAntennaVerticalPatternEntry and AddAntennaVerticalPatternEntry methods. However
    # an implemenation of this function is also available from crc-covlib's helper package.
    LoadRadioMobileV3File(sim, TX, os.path.join(script_dir, 'generic_antenna.ant'))
    # Make sure the antenna patterns are normalized (although in this particular case this does not
    # have any impact since the patterns in the generic_antenna.ant file are already normalized).
    sim.NormalizeAntennaHorizontalPattern(TX)
    sim.NormalizeAntennaVerticalPattern(TX)
    # Set max antenna gain to 16 dBi
    sim.SetAntennaMaximumGain(TX, 16)
    # Points antenna towards east (0=north, 90=east, 180=south, 270=west)
    sim.SetAntennaBearing(TX, covlib.BearingReference.TRUE_NORTH, 90)
    # Set antenna tilt (elec. or mech.) (-90=zenith, 0=horizon, +90=nadir)
    sim.SetAntennaElectricalTilt(TX, 0)
    sim.SetAntennaMechanicalTilt(TX, 0)
    # Select method to interpolate antenna gain from horizontal and vertical patterns
    sim.SetAntennaPatternApproximationMethod(TX, covlib.PatternApproximationMethod.HYBRID)

    # Set antenna parameters (at receiver):
    RX = covlib.Terminal.RECEIVER
    # Load the same antenna pattern file at the receiver but this time using crc-covlib's
    # helper package. Functions for other file formats are also available.
    from crc_covlib.helper import antennas
    antennas.LoadRadioMobileV3File(sim, RX, os.path.join(script_dir, 'generic_antenna.ant'), True)
    # Set max antenna gain to 8 dBi
    sim.SetAntennaMaximumGain(RX, 8)
    # For every reception point, have the receiver antenna points directly towards the transmitter
    # (0=towards other terminal, 180=away from other terminal).
    sim.SetAntennaBearing(RX, covlib.BearingReference.OTHER_TERMINAL, 0)

    # Select propagation model
    sim.SetPropagationModel(covlib.PropagationModel.LONGLEY_RICE)

    # Use no terrain elevation to better see the impact of antennas
    sim.SetPrimaryTerrainElevDataSource(covlib.TerrainElevDataSource.TERR_ELEV_NONE)

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)

    # Select a result type that takes both transmitter and receiver antennas into account.
    sim.SetResultType(covlib.ResultType.RECEIVED_POWER_DBM)

    # Set contour values and colors when exporting results to .mif or .kml files
    sim.ClearCoverageDisplayFills()
    sim.AddCoverageDisplayFill(-85, -75, covlib.RGBtoInt(85, 85, 255))
    sim.AddCoverageDisplayFill(-75, -65, covlib.RGBtoInt(0, 0, 255))
    sim.AddCoverageDisplayFill(-65, 0, covlib.RGBtoInt(0, 0, 136))

    print('Generating results (rx pointing at tx)...')

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'antenna-pattern-rx-towards-tx.kml'))

    print('Generating results (rx pointing away from tx)...')

    # Now have the receiver antenna point away from the transmitter. The resulting coverage
    # should be much smaller.
    sim.SetAntennaBearing(RX, covlib.BearingReference.OTHER_TERMINAL, 180)
    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToKmlFile(os.path.join(script_dir, 'antenna-pattern-rx-away-from-tx.kml'))

    print('Simulations completed\n')
