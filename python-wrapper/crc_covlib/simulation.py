# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

"""Python wrapper for crc-covlib
"""

from crc_covlib.__init__ import __version__

__all__ = ['__version__',
           'DisableWargings',
           'IntToRGB',
           'RGBtoInt',
           'Polarization',
           'LRClimaticZone',
           'LRPercentageSet',
           'LRModeOfVariability',
           'P1812ClutterCategory',
           'P1812LandCoverMappingType',
           'P1812SurfaceProfileMethod',
           'P452PredictionType',
           'P452HeightGainModelClutterCategory',
           'P452HeightGainModelClutterParam',
           'P452HeightGainModelMode',
           'P452ClutterCategory',
           'P452LandCoverMappingType',
           'P452SurfaceProfileMethod',
           'EHataClutterEnvironment',
           'P2109BuildingType',
           'ResultType',
           'TerrainElevDataSource',
           'LandCoverDataSource',
           'SurfaceElevDataSource',
           'GenerateStatus',
           'SamplingMethod',
           'PropagationModel', 
           'AUTOMATIC',
           'Terminal',
           'BearingReference',
           'PatternApproximationMethod',
           'PowerType',
           'ITURadioClimaticZone',
           'ITUDigitalMap',
           'ReceptionPointDetailedResult',
           'Simulation']


import ctypes
import sys
import os
import enum
from typing import List
import warnings



def _load_covlib_library():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if sys.maxsize > 2**32:
        bin_dir = 'bin_64bit'
    else:
        bin_dir = 'bin_32bit'
        raise ImportError('\nNo 32-bit version of crc-covlib-wrap is currently available, ' \
            'you may want to use a 64-bit python interpreter instead.\n')
    if sys.platform.startswith('win32'):
        covlib_pathname = os.path.join(cur_dir, bin_dir, 'crc-covlib.dll')
    else:
        covlib_pathname = os.path.join(cur_dir, bin_dir, 'libcrc-covlib.so')
    try:
        covlib = ctypes.CDLL(covlib_pathname)
        #print("crc-covlib successfully loaded ", covlib)
        _set_args_and_return_ctypes(covlib)
        itu_digital_maps_dir = os.path.join(cur_dir, 'data', 'itu_proprietary')
        if covlib.SetITUProprietaryDataDirectory(itu_digital_maps_dir.encode()) == False:
            print('\nWARNING (crc_covlib): Reading of ITU digital maps unsuccessful, default values will be used.\n')
        return covlib
    except Exception as e:
        print(e)
        return None



class CovlibDeprecationWarning(Warning): ...

def DisableWargings() -> None:
    warnings.simplefilter('ignore', CovlibDeprecationWarning)


def IntToRGB(colorInt):
    b = colorInt & 255
    g = (colorInt >> 8) & 255
    r = (colorInt >> 16) & 255
    return [r, g, b]


def RGBtoInt(r,g,b):
    colorInt = b + (g << 8) + (r << 16)
    return colorInt


class Polarization(enum.Enum):
    HORIZONTAL_POL = 0
    VERTICAL_POL   = 1


class LRClimaticZone(enum.Enum):
    LR_EQUATORIAL                   = 1
    LR_CONTINENTAL_SUBTROPICAL      = 2
    LR_MARITIME_SUBTROPICAL         = 3
    LR_DESERT                       = 4
    LR_CONTINENTAL_TEMPERATE        = 5
    LR_MARITIME_TEMPERATE_OVER_LAND = 6
    LR_MARITIME_TEMPERATE_OVER_SEA  = 7


class LRPercentageSet(enum.Enum):
    LR_TIME_LOCATION_SITUATION = 1
    LR_CONFIDENCE_RELIABILITY  = 2


class LRModeOfVariability(enum.IntEnum):
    LR_SINGLE_MESSAGE_MODE             =  0
    LR_ACCIDENTAL_MESSAGE_MODE         =  1
    LR_MOBILE_MODE                     =  2
    LR_BROADCAST_MODE                  =  3
    LR_ELIMINATE_LOCATION_VARIABILITY  = 10
    LR_ELIMINATE_SITUATION_VARIABILITY = 20


class P1812ClutterCategory(enum.IntEnum):
    P1812_WATER_SEA           = 1
    P1812_OPEN_RURAL          = 2
    P1812_SUBURBAN            = 3
    P1812_URBAN_TREES_FOREST  = 4
    P1812_DENSE_URBAN         = 5


class P1812LandCoverMappingType(enum.Enum):
    P1812_MAP_TO_CLUTTER_CATEGORY    = 1
    P1812_MAP_TO_REPR_CLUTTER_HEIGHT = 2


class P1812SurfaceProfileMethod(enum.Enum):
    P1812_ADD_REPR_CLUTTER_HEIGHT = 1
    P1812_USE_SURFACE_ELEV_DATA   = 2


class P452PredictionType(enum.Enum):
    P452_AVERAGE_YEAR = 1
    P452_WORST_MONTH  = 2


class P452HeightGainModelClutterCategory(enum.IntEnum):
    P452_HGM_HIGH_CROP_FIELDS                    = 13
    P452_HGM_PARK_LAND                           = 19
    P452_HGM_IRREGULARLY_SPACED_SPARSE_TREES     = 21
    P452_HGM_ORCHARD_REGULARLY_SPACED            = 22
    P452_HGM_SPARSE_HOUSES                       = 31
    P452_HGM_VILLAGE_CENTRE                      = 32
    P452_HGM_DECIDUOUS_TREES_IRREGULARLY_SPACED  = 23
    P452_HGM_DECIDUOUS_TREES_REGULARLY_SPACED    = 24
    P452_HGM_MIXED_TREE_FOREST                   = 27
    P452_HGM_CONIFEROUS_TREES_IRREGULARLY_SPACED = 25
    P452_HGM_CONIFEROUS_TREES_REGULARLY_SPACED   = 26
    P452_HGM_TROPICAL_RAIN_FOREST                = 28
    P452_HGM_SUBURBAN                            = 33
    P452_HGM_DENSE_SUBURBAN                      = 34
    P452_HGM_URBAN                               = 35
    P452_HGM_DENSE_URBAN                         = 36
    P452_HGM_HIGH_RISE_URBAN                     = 38
    P452_HGM_INDUSTRIAL_ZONE                     = 37
    P452_HGM_OTHER                               = 90
    P452_HGM_CUSTOM_AT_TRANSMITTER               = 200
    P452_HGM_CUSTOM_AT_RECEIVER                  = 201


class P452HeightGainModelClutterParam(enum.Enum):
    P452_NOMINAL_HEIGHT_M        = 1
    P452_NOMINAL_DISTANCE_KM     = 2


class P452HeightGainModelMode(enum.Enum):
    P452_NO_SHIELDING            = 1
    P452_USE_CUSTOM_AT_CATEGORY  = 2
    P452_USE_CLUTTER_PROFILE     = 3
    P452_USE_CLUTTER_AT_ENDPOINT = 4


class P452ClutterCategory(enum.IntEnum):
    P452_WATER_SEA           = 1
    P452_OPEN_RURAL          = 2
    P452_SUBURBAN            = 3
    P452_URBAN_TREES_FOREST  = 4
    P452_DENSE_URBAN         = 5


class P452LandCoverMappingType(enum.Enum):
    P452_MAP_TO_CLUTTER_CATEGORY    = 1
    P452_MAP_TO_REPR_CLUTTER_HEIGHT = 2


class P452SurfaceProfileMethod(enum.Enum):
    P452_ADD_REPR_CLUTTER_HEIGHT               = 1
    P452_EXPERIMENTAL_USE_OF_SURFACE_ELEV_DATA = 2


class EHataClutterEnvironment(enum.Enum):
    EHATA_URBAN    = 24
    EHATA_SUBURBAN = 22
    EHATA_RURAL    = 20


class P2109BuildingType(enum.Enum):
    P2109_TRADITIONAL         = 1
    P2109_THERMALLY_EFFICIENT = 2


class ResultType(enum.Enum):
    FIELD_STRENGTH_DBUVM = 1
    PATH_LOSS_DB         = 2
    TRANSMISSION_LOSS_DB = 3
    RECEIVED_POWER_DBM   = 4


class TerrainElevDataSource(enum.Enum):
    TERR_ELEV_NONE            = 0
    TERR_ELEV_SRTM            = 1
    TERR_ELEV_CUSTOM          = 2
    TERR_ELEV_NRCAN_CDEM      = 3
    TERR_ELEV_NRCAN_HRDEM_DTM = 4
    TERR_ELEV_GEOTIFF         = 5
    TERR_ELEV_NRCAN_MRDEM_DTM = 6


class SurfaceElevDataSource(enum.Enum):
    SURF_ELEV_NONE            = 100
    SURF_ELEV_SRTM            = 101
    SURF_ELEV_CUSTOM          = 102
    SURF_ELEV_NRCAN_CDSM      = 103
    SURF_ELEV_NRCAN_HRDEM_DSM = 104
    SURF_ELEV_GEOTIFF         = 105
    SURF_ELEV_NRCAN_MRDEM_DSM = 106


class LandCoverDataSource(enum.Enum):
    LAND_COVER_NONE           = 200
    LAND_COVER_GEOTIFF        = 201
    LAND_COVER_ESA_WORLDCOVER = 202
    LAND_COVER_CUSTOM         = 203
    LAND_COVER_NRCAN          = 204


class GenerateStatus(enum.IntEnum):
    STATUS_OK                             =   0
    STATUS_NO_TERRAIN_ELEV_DATA           =   1
    STATUS_SOME_TERRAIN_ELEV_DATA_MISSING =   2
    STATUS_NO_LAND_COVER_DATA             =   4
    STATUS_SOME_LAND_COVER_DATA_MISSING   =   8
    STATUS_NO_ITU_RCZ_DATA                =  16
    STATUS_SOME_ITU_RCZ_DATA_MISSING      =  32
    STATUS_NO_SURFACE_ELEV_DATA           =  64
    STATUS_SOME_SURFACE_ELEV_DATA_MISSING = 128


class SamplingMethod(enum.Enum):
    NEAREST_NEIGHBOR       = 0
    BILINEAR_INTERPOLATION = 1


class PropagationModel(enum.Enum):
    LONGLEY_RICE    = 0
    ITU_R_P_1812    = 1
    ITU_R_P_452_V17 = 2
    ITU_R_P_452_V18 = 3
    FREE_SPACE      = 4
    EXTENDED_HATA   = 5
    CRC_MLPL        = 6


AUTOMATIC = float('nan')


class Terminal(enum.Enum):
    TRANSMITTER  = 1
    RECEIVER     = 2


class BearingReference(enum.Enum):
    TRUE_NORTH     = 1
    OTHER_TERMINAL = 2


class PatternApproximationMethod(enum.Enum):
    H_PATTERN_ONLY   = 1
    V_PATTERN_ONLY   = 2
    SUMMING          = 3
    WEIGHTED_SUMMING = 4
    HYBRID           = 5


class PowerType(enum.Enum):
    TPO  = 1 # Transmitter power output
    ERP  = 2 # Effective radiated power
    EIRP = 3 # Effective isotropic radiated power


class ITURadioClimaticZone(enum.Enum):
    ITU_COASTAL_LAND = 3
    ITU_INLAND       = 4
    ITU_SEA          = 1


class ITUDigitalMap(enum.Enum):
    ITU_MAP_DN50      = 1
    ITU_MAP_N050      = 2
    ITU_MAP_T_ANNUAL  = 3
    ITU_MAP_SURFWV_50 = 4


class ReceptionPointDetailedResult(ctypes.Structure):
    _fields_ = [("result", ctypes.c_double),
                ("pathLoss_dB", ctypes.c_double),
                ("pathLength_km", ctypes.c_double),
                ("transmitterHeightAMSL_m", ctypes.c_double),
                ("receiverHeightAMSL_m", ctypes.c_double),
                ("transmitterAntennaGain_dBi", ctypes.c_double),
                ("receiverAntennaGain_dBi", ctypes.c_double),
                ("azimuthFromTransmitter_degrees", ctypes.c_double),
                ("azimuthFromReceiver_degrees", ctypes.c_double),
                ("elevAngleFromTransmitter_degrees", ctypes.c_double),
                ("elevAngleFromReceiver_degrees", ctypes.c_double)]


class Simulation(object):
    def __init__(self, cnew: bool=True):
        self._lib = _covlib_cdll
        self._sim_ptr = None
        if cnew == True:
            self._sim_ptr = self._lib.NewSimulation()

    def __del__(self):
        self.Release()
        
    def Release(self):
        if self._lib is not None:
            self._lib.Release(self._sim_ptr)
        self._sim_ptr = None
        self._lib = None

    def __copy__(self):
        # Intentional deep copy here
        new_sim = Simulation(cnew=False)
        new_sim._sim_ptr = self._lib.DeepCopySimulation(self._sim_ptr)
        return new_sim

    def __deepcopy__(self, memo):
        new_sim = Simulation(cnew=False)
        new_sim._sim_ptr = self._lib.DeepCopySimulation(self._sim_ptr)
        return new_sim


    # Transmitter parameters

    def SetTransmitterLocation(self, latitude_degrees: float, longitude_degrees: float) -> None:
        return self._lib.SetTransmitterLocation(self._sim_ptr, latitude_degrees, longitude_degrees)

    def GetTransmitterLatitude(self) -> float:
        return self._lib.GetTransmitterLatitude(self._sim_ptr)

    def GetTransmitterLongitude(self) -> float:
        return self._lib.GetTransmitterLongitude(self._sim_ptr)

    def SetTransmitterHeight(self, height_meters: float) -> None:
        return self._lib.SetTransmitterHeight(self._sim_ptr, height_meters)

    def GetTransmitterHeight(self) -> float:
        return self._lib.GetTransmitterHeight(self._sim_ptr)

    def SetTransmitterFrequency(self, frequency_MHz: float) -> None:
        return self._lib.SetTransmitterFrequency(self._sim_ptr, frequency_MHz)

    def GetTransmitterFrequency(self) -> float:
        return self._lib.GetTransmitterFrequency(self._sim_ptr)
    
    def SetTransmitterPower(self, power_watts: float, powerType: PowerType=PowerType.EIRP) -> None:
        if isinstance(powerType, PowerType) == False:
            raise TypeError('parameter powerType: PowerType')
        return self._lib.SetTransmitterPower(self._sim_ptr, power_watts, powerType.value)

    def GetTransmitterPower(self, powerType: PowerType=PowerType.EIRP) -> float:
        if isinstance(powerType, PowerType) == False:
            raise TypeError('parameter powerType: PowerType')
        return self._lib.GetTransmitterPower(self._sim_ptr, powerType.value)

    def SetTransmitterLosses(self, losses_dB: float) -> None:
        return self._lib.SetTransmitterLosses(self._sim_ptr, losses_dB)

    def GetTransmitterLosses(self) -> float:
        return self._lib.GetTransmitterLosses(self._sim_ptr)

    def SetTransmitterPolarization(self, polarization: Polarization) -> None:
        if isinstance(polarization, Polarization) == False:
            raise TypeError('parameter polarization: Polarization')
        return self._lib.SetTransmitterPolarization(self._sim_ptr, polarization.value)

    def GetTransmitterPolarization(self) -> Polarization:
        return Polarization(self._lib.GetTransmitterPolarization(self._sim_ptr))


    # Receiver parameters

    def SetReceiverHeightAboveGround(self, height_meters: float) -> None:
        return self._lib.SetReceiverHeightAboveGround(self._sim_ptr, height_meters)

    def GetReceiverHeightAboveGround(self) -> float:
        return self._lib.GetReceiverHeightAboveGround(self._sim_ptr)

    def SetReceiverLosses(self, losses_dB: float) -> None:
        return self._lib.SetReceiverLosses(self._sim_ptr, losses_dB)

    def GetReceiverLosses(self) -> float:
        return self._lib.GetReceiverLosses(self._sim_ptr)


    # Antenna parameters

    def ClearAntennaPatterns(self, terminal: Terminal, clearHorizontalPattern: bool=True, clearVerticalPattern: bool=True) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.ClearAntennaPatterns(self._sim_ptr, terminal.value, clearHorizontalPattern, clearVerticalPattern)

    def AddAntennaHorizontalPatternEntry(self, terminal: Terminal, azimuth_degrees: float, gain_dB: float) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.AddAntennaHorizontalPatternEntry(self._sim_ptr, terminal.value, azimuth_degrees, gain_dB)

    def AddAntennaVerticalPatternEntry(self, terminal: Terminal, azimuth_degrees: int, elevAngle_degrees: float, gain_dB: float) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.AddAntennaVerticalPatternEntry(self._sim_ptr, terminal.value, azimuth_degrees, elevAngle_degrees, gain_dB)

    def SetAntennaElectricalTilt(self, terminal: Terminal, elecricalTilt_degrees: float) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.SetAntennaElectricalTilt(self._sim_ptr, terminal.value, elecricalTilt_degrees)

    def GetAntennaElectricalTilt(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaElectricalTilt(self._sim_ptr, terminal.value)

    def SetAntennaMechanicalTilt(self, terminal: Terminal, mechanicalTilt_degrees: float, azimuth_degrees: float=0) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.SetAntennaMechanicalTilt(self._sim_ptr, terminal.value, mechanicalTilt_degrees, azimuth_degrees)

    def GetAntennaMechanicalTilt(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaMechanicalTilt(self._sim_ptr, terminal.value)

    def GetAntennaMechanicalTiltAzimuth(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaMechanicalTiltAzimuth(self._sim_ptr, terminal.value)

    def SetAntennaMaximumGain(self, terminal: Terminal, maxGain_dBi: float) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.SetAntennaMaximumGain(self._sim_ptr, terminal.value, maxGain_dBi)

    def GetAntennaMaximumGain(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaMaximumGain(self._sim_ptr, terminal.value)

    def SetAntennaBearing(self, terminal: Terminal, bearingRef: BearingReference, bearing_degrees: float) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        if isinstance(bearingRef, BearingReference) == False:
            raise TypeError('parameter bearingRef: BearingReference')
        return self._lib.SetAntennaBearing(self._sim_ptr, terminal.value, bearingRef.value, bearing_degrees)

    def GetAntennaBearingReference(self, terminal: Terminal) -> BearingReference:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return BearingReference(self._lib.GetAntennaBearingReference(self._sim_ptr, terminal.value))

    def GetAntennaBearing(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaBearing(self._sim_ptr, terminal.value)

    def NormalizeAntennaHorizontalPattern(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.NormalizeAntennaHorizontalPattern(self._sim_ptr, terminal.value)

    def NormalizeAntennaVerticalPattern(self, terminal: Terminal) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.NormalizeAntennaVerticalPattern(self._sim_ptr, terminal.value)

    def SetAntennaPatternApproximationMethod(self, terminal: Terminal, method: PatternApproximationMethod) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        if isinstance(method, PatternApproximationMethod) == False:
            raise TypeError('parameter method: PatternApproximationMethod')
        return self._lib.SetAntennaPatternApproximationMethod(self._sim_ptr, terminal.value, method.value)

    def GetAntennaPatternApproximationMethod(self, terminal: Terminal) -> PatternApproximationMethod:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return PatternApproximationMethod(self._lib.GetAntennaPatternApproximationMethod(self._sim_ptr, terminal.value))

    def GetAntennaGain(self, terminal: Terminal, azimuth_degrees: float, elevAngle_degrees: float, receiverLatitude_degrees: float=0, receiverLongitude_degrees:float=0) -> float:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return self._lib.GetAntennaGain(self._sim_ptr, terminal.value, azimuth_degrees, elevAngle_degrees, receiverLatitude_degrees, receiverLongitude_degrees)


    # Propagation model selection

    def SetPropagationModel(self, propagationModel: PropagationModel) -> None:
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.SetPropagationModel(self._sim_ptr, propagationModel.value)

    def GetPropagationModel(self) -> PropagationModel:
        return PropagationModel(self._lib.GetPropagationModel(self._sim_ptr))


    # Longley-Rice propagation model parameters

    def SetLongleyRiceSurfaceRefractivity(self, refractivity_NUnits: float) -> None:
        return self._lib.SetLongleyRiceSurfaceRefractivity(self._sim_ptr, refractivity_NUnits)

    def GetLongleyRiceSurfaceRefractivity(self) -> float:
        return self._lib.GetLongleyRiceSurfaceRefractivity(self._sim_ptr)

    def SetLongleyRiceGroundDielectricConst(self, dielectricConst: float) -> None:
        return self._lib.SetLongleyRiceGroundDielectricConst(self._sim_ptr, dielectricConst)

    def GetLongleyRiceGroundDielectricConst(self) -> float:
        return self._lib.GetLongleyRiceGroundDielectricConst(self._sim_ptr)
    
    def SetLongleyRiceGroundConductivity(self, groundConduct_Sm: float) -> None:
        return self._lib.SetLongleyRiceGroundConductivity(self._sim_ptr, groundConduct_Sm)

    def GetLongleyRiceGroundConductivity(self) -> float:
        return self._lib.GetLongleyRiceGroundConductivity(self._sim_ptr)

    def SetLongleyRiceClimaticZone(self, climaticZone: LRClimaticZone) -> None:
        if isinstance(climaticZone, LRClimaticZone) == False:
            raise TypeError('parameter climaticZone: LRClimaticZone')
        return self._lib.SetLongleyRiceClimaticZone(self._sim_ptr, climaticZone.value)

    def GetLongleyRiceClimaticZone(self) -> LRClimaticZone:
        return LRClimaticZone(self._lib.GetLongleyRiceClimaticZone(self._sim_ptr))

    def SetLongleyRiceActivePercentageSet(self, percentageSet: LRPercentageSet) -> None:
        if isinstance(percentageSet, LRPercentageSet) == False:
            raise TypeError('parameter percentageSet: LRPercentageSet')
        return self._lib.SetLongleyRiceActivePercentageSet(self._sim_ptr, percentageSet.value)
    
    def GetLongleyRiceActivePercentageSet(self) -> LRPercentageSet:
        return LRPercentageSet(self._lib.GetLongleyRiceActivePercentageSet(self._sim_ptr))

    def SetLongleyRiceTimePercentage(self, time_percent: float) -> None:
        return self._lib.SetLongleyRiceTimePercentage(self._sim_ptr, time_percent)

    def GetLongleyRiceTimePercentage(self) -> float:
        return self._lib.GetLongleyRiceTimePercentage(self._sim_ptr)

    def SetLongleyRiceLocationPercentage(self, location_percent: float) -> None:
        return self._lib.SetLongleyRiceLocationPercentage(self._sim_ptr, location_percent)
    
    def GetLongleyRiceLocationPercentage(self) -> float:
        return self._lib.GetLongleyRiceLocationPercentage(self._sim_ptr)

    def SetLongleyRiceSituationPercentage(self, situation_percent: float) -> None:
        return self._lib.SetLongleyRiceSituationPercentage(self._sim_ptr, situation_percent)

    def GetLongleyRiceSituationPercentage(self) -> float:
        return self._lib.GetLongleyRiceSituationPercentage(self._sim_ptr)
    
    def SetLongleyRiceConfidencePercentage(self, confidence_percent: float) -> None:
        return self._lib.SetLongleyRiceConfidencePercentage(self._sim_ptr, confidence_percent)
    
    def GetLongleyRiceConfidencePercentage(self) -> float:
        return self._lib.GetLongleyRiceConfidencePercentage(self._sim_ptr)

    def SetLongleyRiceReliabilityPercentage(self, reliability_percent: float) -> None:
        return self._lib.SetLongleyRiceReliabilityPercentage(self._sim_ptr, reliability_percent)
    
    def GetLongleyRiceReliabilityPercentage(self) -> float:
        return self._lib.GetLongleyRiceReliabilityPercentage(self._sim_ptr)

    def SetLongleyRiceModeOfVariability(self, mode: int) -> None:
        return self._lib.SetLongleyRiceModeOfVariability(self._sim_ptr, mode)
    
    def GetLongleyRiceModeOfVariability(self) -> int:
        return self._lib.GetLongleyRiceModeOfVariability(self._sim_ptr)


    # ITU-R P.1812 propagation model parameters

    def SetITURP1812TimePercentage(self, time_percent: float) -> None:
        return self._lib.SetITURP1812TimePercentage(self._sim_ptr, time_percent)

    def GetITURP1812TimePercentage(self) -> float:
        return self._lib.GetITURP1812TimePercentage(self._sim_ptr)

    def SetITURP1812LocationPercentage(self, location_percent: float) -> None:
        return self._lib.SetITURP1812LocationPercentage(self._sim_ptr, location_percent)

    def GetITURP1812LocationPercentage(self) -> float:
        return self._lib.GetITURP1812LocationPercentage(self._sim_ptr)

    def SetITURP1812AverageRadioRefractivityLapseRate(self, deltaN_Nunitskm: float) -> None:
        return self._lib.SetITURP1812AverageRadioRefractivityLapseRate(self._sim_ptr, deltaN_Nunitskm)

    def GetITURP1812AverageRadioRefractivityLapseRate(self) -> float:
        return self._lib.GetITURP1812AverageRadioRefractivityLapseRate(self._sim_ptr)

    def SetITURP1812SeaLevelSurfaceRefractivity(self, N0_Nunits: float) -> None:
        return self._lib.SetITURP1812SeaLevelSurfaceRefractivity(self._sim_ptr, N0_Nunits)

    def GetITURP1812SeaLevelSurfaceRefractivity(self) -> float:
        return self._lib.GetITURP1812SeaLevelSurfaceRefractivity(self._sim_ptr)

    def SetITURP1812PredictionResolution(self, resolution_meters: float) -> None:
        return self._lib.SetITURP1812PredictionResolution(self._sim_ptr, resolution_meters)

    def GetITURP1812PredictionResolution(self) -> float:
        return self._lib.GetITURP1812PredictionResolution(self._sim_ptr)

    def SetITURP1812RepresentativeClutterHeight(self, clutterCategory: P1812ClutterCategory, reprHeight_meters: float) -> None:
        if isinstance(clutterCategory, P1812ClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P1812ClutterCategory')
        return self._lib.SetITURP1812RepresentativeClutterHeight(self._sim_ptr, clutterCategory.value, reprHeight_meters)

    def GetITURP1812RepresentativeClutterHeight(self, clutterCategory: P1812ClutterCategory) -> float:
        if isinstance(clutterCategory, P1812ClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P1812ClutterCategory')
        return self._lib.GetITURP1812RepresentativeClutterHeight(self._sim_ptr, clutterCategory.value)

    def SetITURP1812RadioClimaticZonesFile(self, pathname: str) -> None:
        return self._lib.SetITURP1812RadioClimaticZonesFile(self._sim_ptr, pathname.encode())

    def GetITURP1812RadioClimaticZonesFile(self) -> str:
        return self._lib.GetITURP1812RadioClimaticZonesFile(self._sim_ptr).decode()

    def SetITURP1812LandCoverMappingType(self, mappingType: P1812LandCoverMappingType) -> None:
        if isinstance(mappingType, P1812LandCoverMappingType) == False:
            raise TypeError('parameter mappingType: P1812LandCoverMappingType')
        return self._lib.SetITURP1812LandCoverMappingType(self._sim_ptr, mappingType.value)

    def GetITURP1812LandCoverMappingType(self) -> P1812LandCoverMappingType:
        return P1812LandCoverMappingType(self._lib.GetITURP1812LandCoverMappingType(self._sim_ptr))

    def SetITURP1812SurfaceProfileMethod(self, method: P1812SurfaceProfileMethod) -> None:
        if isinstance(method, P1812SurfaceProfileMethod) == False:
            raise TypeError('parameter method: P1812SurfaceProfileMethod')
        return self._lib.SetITURP1812SurfaceProfileMethod(self._sim_ptr, method.value)

    def GetITURP1812SurfaceProfileMethod(self) -> P1812SurfaceProfileMethod:
        return P1812SurfaceProfileMethod(self._lib.GetITURP1812SurfaceProfileMethod(self._sim_ptr))


    # ITU-R P.452 propagation model parameters

    def SetITURP452TimePercentage(self, time_percent: float) -> None:
        return self._lib.SetITURP452TimePercentage(self._sim_ptr, time_percent)

    def GetITURP452TimePercentage(self) -> float:
        return self._lib.GetITURP452TimePercentage(self._sim_ptr)

    def SetITURP452PredictionType(self, predictionType: P452PredictionType) -> None:
        if isinstance(predictionType, P452PredictionType) == False:
            raise TypeError('parameter predictionType: P452PredictionType')
        return self._lib.SetITURP452PredictionType(self._sim_ptr, predictionType.value)
    
    def GetITURP452PredictionType(self) -> P452PredictionType:
        return P452PredictionType(self._lib.GetITURP452PredictionType(self._sim_ptr))

    def SetITURP452AverageRadioRefractivityLapseRate(self, deltaN_Nunitskm: float) -> None:
        return self._lib.SetITURP452AverageRadioRefractivityLapseRate(self._sim_ptr, deltaN_Nunitskm)
    
    def GetITURP452AverageRadioRefractivityLapseRate(self) -> float:
        return self._lib.GetITURP452AverageRadioRefractivityLapseRate(self._sim_ptr)

    def SetITURP452SeaLevelSurfaceRefractivity(self, N0_Nunits: float) -> None:
        return self._lib.SetITURP452SeaLevelSurfaceRefractivity(self._sim_ptr, N0_Nunits)

    def GetITURP452SeaLevelSurfaceRefractivity(self) -> float:
        return self._lib.GetITURP452SeaLevelSurfaceRefractivity(self._sim_ptr)

    def SetITURP452AirTemperature(self, temperature_C: float) -> None:
        return self._lib.SetITURP452AirTemperature(self._sim_ptr, temperature_C)
    
    def GetITURP452AirTemperature(self) -> float:
        return self._lib.GetITURP452AirTemperature(self._sim_ptr)

    def SetITURP452AirPressure(self, pressure_hPa: float) -> None:
        return self._lib.SetITURP452AirPressure(self._sim_ptr, pressure_hPa)
    
    def GetITURP452AirPressure(self) -> float:
        return self._lib.GetITURP452AirPressure(self._sim_ptr)

    def SetITURP452RadioClimaticZonesFile(self, pathname: str) -> None:
        return self._lib.SetITURP452RadioClimaticZonesFile(self._sim_ptr, pathname.encode())

    def GetITURP452RadioClimaticZonesFile(self) -> str:
        return self._lib.GetITURP452RadioClimaticZonesFile(self._sim_ptr).decode()

    def SetITURP452HeightGainModelClutterValue(self, clutterCategory: P452HeightGainModelClutterCategory, nominalParam: P452HeightGainModelClutterParam, nominalValue: float) -> None:
        if isinstance(clutterCategory, P452HeightGainModelClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P452HeightGainModelClutterCategory')
        if isinstance(nominalParam, P452HeightGainModelClutterParam) == False:
            raise TypeError('parameter nominalParam: P452HeightGainModelClutterParam')
        return self._lib.SetITURP452HeightGainModelClutterValue(self._sim_ptr, clutterCategory.value, nominalParam.value, nominalValue)

    def GetITURP452HeightGainModelClutterValue(self, clutterCategory: P452HeightGainModelClutterCategory, nominalParam: P452HeightGainModelClutterParam) -> float:
        if isinstance(clutterCategory, P452HeightGainModelClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P452HeightGainModelClutterCategory')
        if isinstance(nominalParam, P452HeightGainModelClutterParam) == False:
            raise TypeError('parameter nominalParam: P452HeightGainModelClutterParam')
        return self._lib.GetITURP452HeightGainModelClutterValue(self._sim_ptr, clutterCategory.value, nominalParam.value)

    def SetITURP452HeightGainModelMode(self, terminal: Terminal, mode: P452HeightGainModelMode) -> None:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        if isinstance(mode, P452HeightGainModelMode) == False:
            raise TypeError('parameter mode: P452HeightGainModelMode')
        return self._lib.SetITURP452HeightGainModelMode(self._sim_ptr, terminal.value, mode.value)

    def GetITURP452HeightGainModelMode(self, terminal: Terminal) -> P452HeightGainModelMode:
        if isinstance(terminal, Terminal) == False:
            raise TypeError('parameter terminal: Terminal')
        return P452HeightGainModelMode(self._lib.GetITURP452HeightGainModelMode(self._sim_ptr, terminal.value))

    def SetITURP452RepresentativeClutterHeight(self, clutterCategory: P452ClutterCategory, reprHeight_meters: float) -> None:
        if isinstance(clutterCategory, P452ClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P452ClutterCategory')
        return self._lib.SetITURP452RepresentativeClutterHeight(self._sim_ptr, clutterCategory.value, reprHeight_meters)

    def GetITURP452RepresentativeClutterHeight(self, clutterCategory: P452ClutterCategory) -> float:
        if isinstance(clutterCategory, P452ClutterCategory) == False:
            raise TypeError('parameter clutterCategory: P452ClutterCategory')
        return self._lib.GetITURP452RepresentativeClutterHeight(self._sim_ptr, clutterCategory.value)

    def SetITURP452LandCoverMappingType(self, mappingType: P452LandCoverMappingType) -> None:
        if isinstance(mappingType, P452LandCoverMappingType) == False:
            raise TypeError('parameter mappingType: P452LandCoverMappingType')
        return self._lib.SetITURP452LandCoverMappingType(self._sim_ptr, mappingType.value)
    
    def GetITURP452LandCoverMappingType(self) -> P452LandCoverMappingType:
        return P452LandCoverMappingType(self._lib.GetITURP452LandCoverMappingType(self._sim_ptr))

    def SetITURP452SurfaceProfileMethod(self, method: P452SurfaceProfileMethod) -> None:
        if isinstance(method, P452SurfaceProfileMethod) == False:
            raise TypeError('parameter method: P452SurfaceProfileMethod')
        return self._lib.SetITURP452SurfaceProfileMethod(self._sim_ptr, method.value)

    def GetITURP452SurfaceProfileMethod(self) -> P452SurfaceProfileMethod:
        return P452SurfaceProfileMethod(self._lib.GetITURP452SurfaceProfileMethod(self._sim_ptr))


    # Extended Hata propagation model parameters

    def SetEHataClutterEnvironment(self, clutterEnvironment: EHataClutterEnvironment) -> None:
        if isinstance(clutterEnvironment, EHataClutterEnvironment) == False:
            raise TypeError('parameter clutterEnvironment: EHataClutterEnvironment')
        return self._lib.SetEHataClutterEnvironment(self._sim_ptr, clutterEnvironment.value)

    def GetEHataClutterEnvironment(self) -> EHataClutterEnvironment:
        return EHataClutterEnvironment(self._lib.GetEHataClutterEnvironment(self._sim_ptr))

    def SetEHataReliabilityPercentage(self, percent: float) -> None:
        return self._lib.SetEHataReliabilityPercentage(self._sim_ptr, percent)

    def GetEHataReliabilityPercentage(self) -> float:
        return self._lib.GetEHataReliabilityPercentage(self._sim_ptr)


    # ITU-R P.2108 statistical clutter loss model for terrestrial paths

    def SetITURP2108TerrestrialStatModelActiveState(self, active: bool) -> None:
        return self._lib.SetITURP2108TerrestrialStatModelActiveState(self._sim_ptr, active)
    
    def GetITURP2108TerrestrialStatModelActiveState(self) -> bool:
        return self._lib.GetITURP2108TerrestrialStatModelActiveState(self._sim_ptr)

    def SetITURP2108TerrestrialStatModelLocationPercentage(self, location_percent: float) -> None:
        return self._lib.SetITURP2108TerrestrialStatModelLocationPercentage(self._sim_ptr, location_percent)
    
    def GetITURP2108TerrestrialStatModelLocationPercentage(self) -> float:
        return self._lib.GetITURP2108TerrestrialStatModelLocationPercentage(self._sim_ptr)

    def GetITURP2108TerrestrialStatModelLoss(self, frequency_GHz: float, distance_km: float) -> None:
        return self._lib.GetITURP2108TerrestrialStatModelLoss(self._sim_ptr, frequency_GHz, distance_km)


    # ITU-R P.2109 building entry loss model

    def SetITURP2109ActiveState(self, active: bool) -> None:
        return self._lib.SetITURP2109ActiveState(self._sim_ptr, active)

    def GetITURP2109ActiveState(self) -> bool:
        return self._lib.GetITURP2109ActiveState(self._sim_ptr)
    
    def SetITURP2109Probability(self, probability_percent: float) -> None:
        return self._lib.SetITURP2109Probability(self._sim_ptr, probability_percent)
    
    def GetITURP2109Probability(self) -> float:
        return self._lib.GetITURP2109Probability(self._sim_ptr)

    def SetITURP2109DefaultBuildingType(self, buildingType: P2109BuildingType) -> None:
        if isinstance(buildingType, P2109BuildingType) == False:
            raise TypeError('parameter buildingType: P2109BuildingType')
        return self._lib.SetITURP2109DefaultBuildingType(self._sim_ptr, buildingType.value)

    def GetITURP2109DefaultBuildingType(self) -> int:
        return P2109BuildingType(self._lib.GetITURP2109DefaultBuildingType(self._sim_ptr))

    def GetITURP2109BuildingEntryLoss(self, frequency_GHz: float, elevAngle_degrees: float) -> float:
        return self._lib.GetITURP2109BuildingEntryLoss(self._sim_ptr, frequency_GHz, elevAngle_degrees)


    # ITU-R P.676 gaseous attenuation model for terrestrial paths

    def SetITURP676TerrPathGaseousAttenuationActiveState(self, active: bool, atmPressure_hPa: float=AUTOMATIC, temperature_C: float=AUTOMATIC, waterVapourDensity_gm3: float=AUTOMATIC) -> None:
        return self._lib.SetITURP676TerrPathGaseousAttenuationActiveState(self._sim_ptr, active, atmPressure_hPa, temperature_C, waterVapourDensity_gm3)

    def GetITURP676TerrPathGaseousAttenuationActiveState(self) -> bool:
        return self._lib.GetITURP676TerrPathGaseousAttenuationActiveState(self._sim_ptr)

    def GetITURP676GaseousAttenuation(self, frequency_GHz: float, atmPressure_hPa: float=1013.25, temperature_C: float=15, waterVapourDensity_gm3: float=7.5) -> float:
        return self._lib.GetITURP676GaseousAttenuation(self._sim_ptr, frequency_GHz, atmPressure_hPa, temperature_C, waterVapourDensity_gm3)


    # ITU digial maps

    def GetITUDigitalMapValue(self, map: ITUDigitalMap, latitude_degrees: float, longitude_degrees: float) -> float:
        if isinstance(map, ITUDigitalMap) == False:
            raise TypeError('parameter map: ITUDigitalMap')
        return self._lib.GetITUDigitalMapValue(self._sim_ptr, map.value, latitude_degrees, longitude_degrees)


    # Terrain elevation data parameters

    def SetPrimaryTerrainElevDataSource(self, terrainElevSource: TerrainElevDataSource) -> None:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return self._lib.SetPrimaryTerrainElevDataSource(self._sim_ptr, terrainElevSource.value)

    def GetPrimaryTerrainElevDataSource(self) -> TerrainElevDataSource:
        return TerrainElevDataSource(self._lib.GetPrimaryTerrainElevDataSource(self._sim_ptr))

    def SetSecondaryTerrainElevDataSource(self, terrainElevSource: TerrainElevDataSource) -> None:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return self._lib.SetSecondaryTerrainElevDataSource(self._sim_ptr, terrainElevSource.value)

    def GetSecondaryTerrainElevDataSource(self) -> TerrainElevDataSource:
        return TerrainElevDataSource(self._lib.GetSecondaryTerrainElevDataSource(self._sim_ptr))

    def SetTertiaryTerrainElevDataSource(self, terrainElevSource: TerrainElevDataSource) -> None:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return self._lib.SetTertiaryTerrainElevDataSource(self._sim_ptr, terrainElevSource.value)

    def GetTertiaryTerrainElevDataSource(self) -> TerrainElevDataSource:
        return TerrainElevDataSource(self._lib.GetTertiaryTerrainElevDataSource(self._sim_ptr))

    def SetTerrainElevDataSourceDirectory(self, terrainElevSource: TerrainElevDataSource, directory: str, useIndexFile: bool=False, overwriteIndexFile: bool=False) -> None:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return self._lib.SetTerrainElevDataSourceDirectory(self._sim_ptr, terrainElevSource.value, directory.encode(), useIndexFile, overwriteIndexFile)

    def GetTerrainElevDataSourceDirectory(self, terrainElevSource: TerrainElevDataSource) -> str:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return self._lib.GetTerrainElevDataSourceDirectory(self._sim_ptr, terrainElevSource.value).decode()

    def SetTerrainElevDataSamplingResolution(self, samplingResolution_meters: float) -> None:
        return self._lib.SetTerrainElevDataSamplingResolution(self._sim_ptr, samplingResolution_meters)

    def GetTerrainElevDataSamplingResolution(self) -> float:
        return self._lib.GetTerrainElevDataSamplingResolution(self._sim_ptr)

    def SetTerrainElevDataSourceSamplingMethod(self, terrainElevSource: TerrainElevDataSource, samplingMethod: SamplingMethod) -> None:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        if isinstance(samplingMethod, SamplingMethod) == False:
            raise TypeError('parameter samplingMethod: SamplingMethod')
        return self._lib.SetTerrainElevDataSourceSamplingMethod(self._sim_ptr, terrainElevSource.value, samplingMethod.value)

    def GetTerrainElevDataSourceSamplingMethod(self, terrainElevSource: TerrainElevDataSource) -> SamplingMethod:
        if isinstance(terrainElevSource, TerrainElevDataSource) == False:
            raise TypeError('parameter terrainElevSource: TerrainElevDataSource')
        return SamplingMethod(self._lib.GetTerrainElevDataSourceSamplingMethod(self._sim_ptr, terrainElevSource.value))

    def AddCustomTerrainElevData(self, lowerLeftCornerLat_degrees: float, lowerLeftCornerLon_degrees: float, upperRightCornerLat_degrees: float, upperRightCornerLon_degrees: float, numHorizSamples: int, numVertSamples: int, terrainElevData_meters: List[float], defineNoDataValue: bool=False, noDataValue: float=0) -> bool:
        if terrainElevData_meters is None:
            terrainElevData_meters = []
        numElevSamples = numHorizSamples * numVertSamples
        data_c = (ctypes.c_float * numElevSamples)(*terrainElevData_meters)
        return self._lib.AddCustomTerrainElevData(self._sim_ptr, lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, data_c, defineNoDataValue, noDataValue)

    def ClearCustomTerrainElevData(self) -> None:
        return self._lib.ClearCustomTerrainElevData(self._sim_ptr)

    def GetTerrainElevation(self, latitude_degrees: float, longitude_degrees: float, noDataValue: float=0) -> float:
        return self._lib.GetTerrainElevation(self._sim_ptr, latitude_degrees, longitude_degrees, noDataValue)


    # Land cover data parameters

    def SetPrimaryLandCoverDataSource(self, landCoverSource: LandCoverDataSource) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        return self._lib.SetPrimaryLandCoverDataSource(self._sim_ptr, landCoverSource.value)

    def GetPrimaryLandCoverDataSource(self) -> LandCoverDataSource:
        return LandCoverDataSource(self._lib.GetPrimaryLandCoverDataSource(self._sim_ptr))

    def SetSecondaryLandCoverDataSource(self, landCoverSource: LandCoverDataSource) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        return self._lib.SetSecondaryLandCoverDataSource(self._sim_ptr, landCoverSource.value)

    def GetSecondaryLandCoverDataSource(self) -> LandCoverDataSource:
        return LandCoverDataSource(self._lib.GetSecondaryLandCoverDataSource(self._sim_ptr))

    def SetLandCoverDataSourceDirectory(self, landCoverSource: LandCoverDataSource, directory: str, useIndexFile: bool=False, overwriteIndexFile: bool=False) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        return self._lib.SetLandCoverDataSourceDirectory(self._sim_ptr, landCoverSource.value, directory.encode(), useIndexFile, overwriteIndexFile)

    def GetLandCoverDataSourceDirectory(self, landCoverSource: LandCoverDataSource) -> str:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        return self._lib.GetLandCoverDataSourceDirectory(self._sim_ptr, landCoverSource.value).decode()

    def AddCustomLandCoverData(self, lowerLeftCornerLat_degrees: float, lowerLeftCornerLon_degrees: float, upperRightCornerLat_degrees: float, upperRightCornerLon_degrees: float, numHorizSamples: int, numVertSamples: int, landCoverData: List[int], defineNoDataValue: bool=False, noDataValue: int=0) -> bool:
        if landCoverData is None:
            landCoverData = []
        numElevSamples = numHorizSamples * numVertSamples
        data_c = (ctypes.c_short * numElevSamples)(*landCoverData)
        return self._lib.AddCustomLandCoverData(self._sim_ptr, lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, data_c, defineNoDataValue, noDataValue)

    def ClearCustomLandCoverData(self) -> None:
        return self._lib.ClearCustomLandCoverData(self._sim_ptr)

    def GetLandCoverClass(self, latitude_degrees: float, longitude_degrees: float) -> int:
        return self._lib.GetLandCoverClass(self._sim_ptr, latitude_degrees, longitude_degrees)

    def GetLandCoverClassMappedValue(self, latitude_degrees: float, longitude_degrees: float, propagationModel: PropagationModel) -> int:
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.GetLandCoverClassMappedValue(self._sim_ptr, latitude_degrees, longitude_degrees, propagationModel.value)

    def SetLandCoverClassMapping(self, landCoverSource: LandCoverDataSource, sourceClass: int, propagationModel: PropagationModel, modelValue: int) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.SetLandCoverClassMapping(self._sim_ptr, landCoverSource.value, sourceClass, propagationModel.value, modelValue)

    def GetLandCoverClassMapping(self, landCoverSource: LandCoverDataSource, sourceClass: int, propagationModel: PropagationModel) -> int:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.GetLandCoverClassMapping(self._sim_ptr, landCoverSource.value, sourceClass, propagationModel.value)

    def SetDefaultLandCoverClassMapping(self, landCoverSource: LandCoverDataSource, propagationModel: PropagationModel, modelValue: int) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.SetDefaultLandCoverClassMapping(self._sim_ptr, landCoverSource.value, propagationModel.value, modelValue)

    def GetDefaultLandCoverClassMapping(self, landCoverSource: LandCoverDataSource, propagationModel: PropagationModel) -> int:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.GetDefaultLandCoverClassMapping(self._sim_ptr, landCoverSource.value, propagationModel.value)

    def ClearLandCoverClassMappings(self, landCoverSource: LandCoverDataSource, propagationModel: PropagationModel) -> None:
        if isinstance(landCoverSource, LandCoverDataSource) == False:
            raise TypeError('parameter landCoverSource: LandCoverDataSource')
        if isinstance(propagationModel, PropagationModel) == False:
            raise TypeError('parameter propagationModel: PropagationModel')
        return self._lib.ClearLandCoverClassMappings(self._sim_ptr, landCoverSource.value, propagationModel.value)


    # Surface elevation data parameters

    def SetPrimarySurfaceElevDataSource(self, surfaceElevSource: SurfaceElevDataSource) -> None:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return self._lib.SetPrimarySurfaceElevDataSource(self._sim_ptr, surfaceElevSource.value)

    def GetPrimarySurfaceElevDataSource(self) -> SurfaceElevDataSource:
        return SurfaceElevDataSource(self._lib.GetPrimarySurfaceElevDataSource(self._sim_ptr))

    def SetSecondarySurfaceElevDataSource(self, surfaceElevSource: SurfaceElevDataSource) -> None:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return self._lib.SetSecondarySurfaceElevDataSource(self._sim_ptr, surfaceElevSource.value)

    def GetSecondarySurfaceElevDataSource(self) -> SurfaceElevDataSource:
        return SurfaceElevDataSource(self._lib.GetSecondarySurfaceElevDataSource(self._sim_ptr))

    def SetTertiarySurfaceElevDataSource(self, surfaceElevSource: SurfaceElevDataSource) -> None:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return self._lib.SetTertiarySurfaceElevDataSource(self._sim_ptr, surfaceElevSource.value)

    def GetTertiarySurfaceElevDataSource(self) -> SurfaceElevDataSource:
        return SurfaceElevDataSource(self._lib.GetTertiarySurfaceElevDataSource(self._sim_ptr))

    def SetSurfaceElevDataSourceDirectory(self, surfaceElevSource: SurfaceElevDataSource, directory: str, useIndexFile: bool=False, overwriteIndexFile: bool=False) -> None:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return self._lib.SetSurfaceElevDataSourceDirectory(self._sim_ptr, surfaceElevSource.value, directory.encode(), useIndexFile, overwriteIndexFile)

    def GetSurfaceElevDataSourceDirectory(self, surfaceElevSource: SurfaceElevDataSource) -> str:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return self._lib.GetSurfaceElevDataSourceDirectory(self._sim_ptr, surfaceElevSource.value).decode()

    def SetSurfaceAndTerrainDataSourcePairing(self, usePairing: bool) -> None:
        return self._lib.SetSurfaceAndTerrainDataSourcePairing(self._sim_ptr, usePairing)

    def GetSurfaceAndTerrainDataSourcePairing(self) -> bool:
        return self._lib.GetSurfaceAndTerrainDataSourcePairing(self._sim_ptr)

    def SetSurfaceElevDataSourceSamplingMethod(self, surfaceElevSource: SurfaceElevDataSource, samplingMethod: SamplingMethod) -> None:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        if isinstance(samplingMethod, SamplingMethod) == False:
            raise TypeError('parameter samplingMethod: SamplingMethod')
        return self._lib.SetSurfaceElevDataSourceSamplingMethod(self._sim_ptr, surfaceElevSource.value, samplingMethod.value)

    def GetSurfaceElevDataSourceSamplingMethod(self, surfaceElevSource: SurfaceElevDataSource) -> SamplingMethod:
        if isinstance(surfaceElevSource, SurfaceElevDataSource) == False:
            raise TypeError('parameter surfaceElevSource: SurfaceElevDataSource')
        return SamplingMethod(self._lib.GetSurfaceElevDataSourceSamplingMethod(self._sim_ptr, surfaceElevSource.value))

    def AddCustomSurfaceElevData(self, lowerLeftCornerLat_degrees: float, lowerLeftCornerLon_degrees: float, upperRightCornerLat_degrees: float, upperRightCornerLon_degrees: float, numHorizSamples: int, numVertSamples: int, surfaceElevData_meters: List[float], defineNoDataValue: bool=False, noDataValue: float=0) -> bool:
        if surfaceElevData_meters is None:
            surfaceElevData_meters = []
        numElevSamples = numHorizSamples * numVertSamples
        data_c = (ctypes.c_float * numElevSamples)(*surfaceElevData_meters)
        return self._lib.AddCustomSurfaceElevData(self._sim_ptr, lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, data_c, defineNoDataValue, noDataValue)

    def ClearCustomSurfaceElevData(self, lowerLeftCornerLat_degrees: float, lowerLeftCornerLon_degrees: float, upperRightCornerLat_degrees: float, upperRightCornerLon_degrees: float, numHorizSamples: int, numVertSamples: int, pathname: str, defineNoDataValue: bool=False, noDataValue: float=0) -> bool:
        return self._lib.ClearCustomSurfaceElevData(self._sim_ptr, lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees, numHorizSamples, numVertSamples, pathname.encode(), defineNoDataValue, noDataValue)

    def GetSurfaceElevation(self, latitude_degrees: float, longitude_degrees: float, noDataValue: float=0) -> float:
        return self._lib.GetSurfaceElevation(self._sim_ptr, latitude_degrees, longitude_degrees, noDataValue)


    # Reception area parameters

    def SetReceptionAreaCorners(self, lowerLeftCornerLat_degrees: float, lowerLeftCornerLon_degrees: float, upperRightCornerLat_degrees: float, upperRightCornerLon_degrees: float) -> None:
        return self._lib.SetReceptionAreaCorners(self._sim_ptr, lowerLeftCornerLat_degrees, lowerLeftCornerLon_degrees, upperRightCornerLat_degrees, upperRightCornerLon_degrees)

    def GetReceptionAreaLowerLeftCornerLatitude(self) -> float:
        return self._lib.GetReceptionAreaLowerLeftCornerLatitude(self._sim_ptr)

    def GetReceptionAreaLowerLeftCornerLongitude(self) -> float:
        return self._lib.GetReceptionAreaLowerLeftCornerLongitude(self._sim_ptr)

    def GetReceptionAreaUpperRightCornerLatitude(self) -> float:
        return self._lib.GetReceptionAreaUpperRightCornerLatitude(self._sim_ptr)

    def GetReceptionAreaUpperRightCornerLongitude(self) -> float:
        return self._lib.GetReceptionAreaUpperRightCornerLongitude(self._sim_ptr)

    def SetReceptionAreaNumHorizontalPoints(self, numPoints: int) -> None:
        return self._lib.SetReceptionAreaNumHorizontalPoints(self._sim_ptr, numPoints)

    def GetReceptionAreaNumHorizontalPoints(self) -> int:
        return self._lib.GetReceptionAreaNumHorizontalPoints(self._sim_ptr)

    def SetReceptionAreaNumVerticalPoints(self, numPoints: int) -> None:
        return self._lib.SetReceptionAreaNumVerticalPoints(self._sim_ptr, numPoints)

    def GetReceptionAreaNumVerticalPoints(self) -> int:
        return self._lib.GetReceptionAreaNumVerticalPoints(self._sim_ptr)


    # Result type parameters

    def SetResultType(self, resultType: ResultType) -> None:
        if isinstance(resultType, ResultType) == False:
            raise TypeError('parameter resultType: ResultType')
        return self._lib.SetResultType(self._sim_ptr, resultType.value)

    def GetResultType(self) -> ResultType:
        return ResultType(self._lib.GetResultType(self._sim_ptr))


    # Coverage display parameters for vector files (.mif and .kml)

    def ClearCoverageDisplayFills(self) -> None:
        return self._lib.ClearCoverageDisplayFills(self._sim_ptr)

    def AddCoverageDisplayFill(self, fromValue: float, toValue: float, rgbColor: int) -> None:
        return self._lib.AddCoverageDisplayFill(self._sim_ptr, fromValue, toValue, rgbColor)

    def GetCoverageDisplayNumFills(self) -> int:
        return self._lib.GetCoverageDisplayNumFills(self._sim_ptr)

    def GetCoverageDisplayFillFromValue(self, index: int) -> float:
        return self._lib.GetCoverageDisplayFillFromValue(self._sim_ptr, index)

    def GetCoverageDisplayFillToValue(self, index: int) -> float:
        return self._lib.GetCoverageDisplayFillToValue(self._sim_ptr, index)

    def GetCoverageDisplayFillColor(self, index: int) -> int:
        return self._lib.GetCoverageDisplayFillColor(self._sim_ptr, index)


    # Generating and accessing results

    def GenerateReceptionPointResult(self, latitude_degrees: float, longitude_degrees: float) -> None:
        return self._lib.GenerateReceptionPointResult(self._sim_ptr, latitude_degrees, longitude_degrees)
    
    def GenerateReceptionPointDetailedResult(self, latitude_degrees: float, longitude_degrees: float) -> ReceptionPointDetailedResult:
        return self._lib.GenerateReceptionPointDetailedResult(self._sim_ptr, latitude_degrees, longitude_degrees)

    # virtual double GenerateProfileReceptionPointResult(double latitude_degrees, double longitude_degrees, int numElevSamples, const double* terrainElevProfile, const int* landCoverClassMappedValueProfile=NULL, const double* surfaceElevProfile=NULL, const ITURadioClimaticZone* ituRadioClimaticZoneProfile=NULL) = 0;
    def GenerateProfileReceptionPointResult(self, latitude_degrees: float, longitude_degrees: float, numSamples: int, terrainElevProfile: List[float], landCoverClassMappedValueProfile: List[int]=None, surfaceElevProfile: List[float]=None, ituRadioClimaticZoneProfile: List[int]=None) -> float:
        if terrainElevProfile is not None:
            terrElevData_c = (ctypes.c_double * numSamples)(*terrainElevProfile)
        else:
            terrElevData_c = ctypes.POINTER(ctypes.c_double)()
        if landCoverClassMappedValueProfile is not None:
            landCoverData_c = (ctypes.c_int * numSamples)(*landCoverClassMappedValueProfile)
        else:
            landCoverData_c = ctypes.POINTER(ctypes.c_int)()
        if surfaceElevProfile is not None:
            surfElevData_c = (ctypes.c_double * numSamples)(*surfaceElevProfile)
        else:
            surfElevData_c = ctypes.POINTER(ctypes.c_double)()
        if ituRadioClimaticZoneProfile is not None:
            rczData_c = (ctypes.c_int * numSamples)(*ituRadioClimaticZoneProfile)
        else:
            rczData_c = ctypes.POINTER(ctypes.c_int)()
        return self._lib.GenerateProfileReceptionPointResult(self._sim_ptr, latitude_degrees, longitude_degrees, numSamples, terrElevData_c, landCoverData_c, surfElevData_c, rczData_c)

    def GenerateReceptionAreaResults(self) -> None:
        return self._lib.GenerateReceptionAreaResults(self._sim_ptr)

    def GetGenerateStatus(self) -> int:
        return self._lib.GetGenerateStatus(self._sim_ptr)

    def GetReceptionAreaResultValue(self, xIndex: int, yIndex: int) -> float:
        return self._lib.GetReceptionAreaResultValue(self._sim_ptr, xIndex, yIndex)

    def SetReceptionAreaResultValue(self, xIndex: int, yIndex: int, value: float) -> None:
        return self._lib.SetReceptionAreaResultValue(self._sim_ptr, xIndex, yIndex, value)

    def GetReceptionAreaResultValueAtLatLon(self, latitude_degrees: float, longitude_degrees: float) -> float:
        return self._lib.GetReceptionAreaResultValueAtLatLon(self._sim_ptr, latitude_degrees, longitude_degrees)

    def GetReceptionAreaResultLatitude(self, xIndex: int, yIndex: int) -> float:
        return self._lib.GetReceptionAreaResultLatitude(self._sim_ptr, xIndex, yIndex)

    def GetReceptionAreaResultLongitude(self, xIndex: int, yIndex: int) -> float:
        return self._lib.GetReceptionAreaResultLongitude(self._sim_ptr, xIndex, yIndex)

    def ExportReceptionAreaResultsToTextFile(self, pathname: str, resultsColumnName: str=None) -> bool:
        if resultsColumnName is not None:
            colName = resultsColumnName.encode()
        else:
            colName = ctypes.POINTER(ctypes.c_char)()
        return self._lib.ExportReceptionAreaResultsToTextFile(self._sim_ptr, pathname.encode(), colName)

    def ExportReceptionAreaResultsToMifFile(self, pathname: str, resultsUnits: str=None) -> bool:
        if resultsUnits is not None:
            units = resultsUnits.encode()
        else:
            units = ctypes.POINTER(ctypes.c_char)()
        return self._lib.ExportReceptionAreaResultsToMifFile(self._sim_ptr, pathname.encode(), units)

    def ExportReceptionAreaResultsToKmlFile(self, pathname: str, fillOpacity_percent: float=50, lineOpacity_percent: float=50, resultsUnits: str=None) -> bool:
        if resultsUnits is not None:
            units = resultsUnits.encode()
        else:
            units = ctypes.POINTER(ctypes.c_char)()
        return self._lib.ExportReceptionAreaResultsToKmlFile(self._sim_ptr, pathname.encode(), fillOpacity_percent, lineOpacity_percent, units)

    def ExportReceptionAreaResultsToBilFile(self, pathname: str) -> bool:
        return self._lib.ExportReceptionAreaResultsToBilFile(self._sim_ptr, pathname.encode())

    def ExportReceptionAreaTerrainElevationToBilFile(self, pathname: str, numHorizontalPoints: int, numVerticalPoints: int, setNoDataToZero: bool=False) -> bool:
        return self._lib.ExportReceptionAreaTerrainElevationToBilFile(self._sim_ptr, pathname.encode(), numHorizontalPoints, numVerticalPoints, setNoDataToZero)

    def ExportReceptionAreaLandCoverClassesToBilFile(self, pathname: str, numHorizontalPoints: int, numVerticalPoints: int, mapValues: bool) -> bool:
        return self._lib.ExportReceptionAreaLandCoverClassesToBilFile(self._sim_ptr, pathname.encode(), numHorizontalPoints, numVerticalPoints, mapValues)

    def ExportReceptionAreaSurfaceElevationToBilFile(self, pathname: str, numHorizontalPoints: int, numVerticalPoints: int, setNoDataToZero: bool=False) -> bool:
        return self._lib.ExportReceptionAreaSurfaceElevationToBilFile(self._sim_ptr, pathname.encode(), numHorizontalPoints, numVerticalPoints, setNoDataToZero)

    def ExportProfilesToCsvFile(self, pathname: str, latitude_degrees: float, longitude_degrees: float) -> bool:
        return self._lib.ExportProfilesToCsvFile(self._sim_ptr, pathname.encode(), latitude_degrees, longitude_degrees)


    # DEPRECATED methods, kept for compatibility with previous crc-covlib versions, may eventually be removed

    def _deprecatedMethodWarning(self, deprecMethodName: str, newMethodName: str) -> None:
        warnings.warn('Use {}() instead of {}().'.format(newMethodName, deprecMethodName), CovlibDeprecationWarning, stacklevel=3)

    # DEPRECATED method example
    #def DeprecatedMethodName(self, param1: int) -> None:
    #    self._deprecatedMethodWarning('DeprecatedMethodName', 'NewMethodName')
    #    return self.NewMethodName(param1)



def _set_args_and_return_ctypes(lib):
        lib.NewSimulation.restype = ctypes.c_void_p

        lib.Release.argtypes = [ctypes.c_void_p]
        lib.Release.restype = None

        lib.DeepCopySimulation.argtypes = [ctypes.c_void_p]
        lib.DeepCopySimulation.restype = ctypes.c_void_p

        lib.SetITUProprietaryDataDirectory.argtypes = [ctypes.c_char_p]
        lib.SetITUProprietaryDataDirectory.restype = ctypes.c_bool

        # Transmitter parameters

        lib.SetTransmitterLocation.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.SetTransmitterLocation.restype = None

        lib.GetTransmitterLatitude.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterLatitude.restype = ctypes.c_double

        lib.GetTransmitterLongitude.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterLongitude.restype = ctypes.c_double

        lib.SetTransmitterHeight.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetTransmitterHeight.restype = None

        lib.GetTransmitterHeight.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterHeight.restype = ctypes.c_double

        lib.SetTransmitterFrequency.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetTransmitterFrequency.restype = None

        lib.GetTransmitterFrequency.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterFrequency.restype = ctypes.c_double

        lib.SetTransmitterPower.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
        lib.SetTransmitterPower.restype = None

        lib.GetTransmitterPower.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetTransmitterPower.restype = ctypes.c_double

        lib.SetTransmitterLosses.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetTransmitterLosses.restype = None

        lib.GetTransmitterLosses.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterLosses.restype = ctypes.c_double

        lib.SetTransmitterPolarization.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetTransmitterPolarization.restype = None

        lib.GetTransmitterPolarization.argtypes = [ctypes.c_void_p]
        lib.GetTransmitterPolarization.restype = ctypes.c_int


        # Receiver parameters

        lib.SetReceiverHeightAboveGround.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetReceiverHeightAboveGround.restype = None

        lib.GetReceiverHeightAboveGround.argtypes = [ctypes.c_void_p]
        lib.GetReceiverHeightAboveGround.restype = ctypes.c_double

        lib.SetReceiverLosses.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetReceiverLosses.restype = None

        lib.GetReceiverLosses.argtypes = [ctypes.c_void_p]
        lib.GetReceiverLosses.restype = ctypes.c_double


        # Antenna parameters

        lib.ClearAntennaPatterns.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
        lib.ClearAntennaPatterns.restype = None

        lib.AddAntennaHorizontalPatternEntry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double ]
        lib.AddAntennaHorizontalPatternEntry.restype = None

        lib.AddAntennaVerticalPatternEntry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        lib.AddAntennaVerticalPatternEntry.restype = None

        lib.SetAntennaElectricalTilt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.SetAntennaElectricalTilt.restype = None

        lib.GetAntennaElectricalTilt.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaElectricalTilt.restype = ctypes.c_double

        lib.SetAntennaMechanicalTilt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        lib.SetAntennaMechanicalTilt.restype = None

        lib.GetAntennaMechanicalTilt.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaMechanicalTilt.restype = ctypes.c_double

        lib.GetAntennaMechanicalTiltAzimuth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaMechanicalTiltAzimuth.restype = ctypes.c_double

        lib.SetAntennaMaximumGain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.SetAntennaMaximumGain.restype = None

        lib.GetAntennaMaximumGain.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaMaximumGain.restype = ctypes.c_double

        lib.SetAntennaBearing.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.SetAntennaBearing.restype = None

        lib.GetAntennaBearingReference.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaBearingReference.restype = ctypes.c_int

        lib.GetAntennaBearing.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaBearing.restype = ctypes.c_double

        lib.NormalizeAntennaHorizontalPattern.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.NormalizeAntennaHorizontalPattern.restype = ctypes.c_double

        lib.NormalizeAntennaVerticalPattern.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.NormalizeAntennaVerticalPattern.restype = ctypes.c_double

        lib.SetAntennaPatternApproximationMethod.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.SetAntennaPatternApproximationMethod.restype = None

        lib.GetAntennaPatternApproximationMethod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetAntennaPatternApproximationMethod.restype = ctypes.c_int

        lib.GetAntennaGain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.GetAntennaGain.restype = ctypes.c_double


        # Propagation model selection

        lib.SetPropagationModel.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetPropagationModel.restype = None

        lib.GetPropagationModel.argtypes = [ctypes.c_void_p]
        lib.SetPropagationModel.restype = ctypes.c_int


        # Longley-Rice propagation model parameters

        lib.SetLongleyRiceSurfaceRefractivity.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceSurfaceRefractivity.restype = None

        lib.GetLongleyRiceSurfaceRefractivity.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceSurfaceRefractivity.restype = ctypes.c_double

        lib.SetLongleyRiceGroundDielectricConst.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceGroundDielectricConst.restype = None

        lib.GetLongleyRiceGroundDielectricConst.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceGroundDielectricConst.restype = ctypes.c_double
        
        lib.SetLongleyRiceGroundConductivity.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceGroundConductivity.restype = None

        lib.GetLongleyRiceGroundConductivity.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceGroundConductivity.restype = ctypes.c_double

        lib.SetLongleyRiceClimaticZone.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetLongleyRiceClimaticZone.restype = None

        lib.GetLongleyRiceClimaticZone.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceClimaticZone.restype = ctypes.c_int

        lib.SetLongleyRiceActivePercentageSet.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetLongleyRiceActivePercentageSet.restype = None

        lib.GetLongleyRiceActivePercentageSet.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceActivePercentageSet.restype = ctypes.c_int

        lib.SetLongleyRiceTimePercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceTimePercentage.restype = None

        lib.GetLongleyRiceTimePercentage.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceTimePercentage.restype = ctypes.c_double

        lib.SetLongleyRiceLocationPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceLocationPercentage.restype = None

        lib.GetLongleyRiceLocationPercentage.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceLocationPercentage.restype = ctypes.c_double

        lib.SetLongleyRiceSituationPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceSituationPercentage.restype = None

        lib.GetLongleyRiceSituationPercentage.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceSituationPercentage.restype = ctypes.c_double

        lib.SetLongleyRiceConfidencePercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceConfidencePercentage.restype = None

        lib.GetLongleyRiceConfidencePercentage.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceConfidencePercentage.restype = ctypes.c_double

        lib.SetLongleyRiceReliabilityPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetLongleyRiceReliabilityPercentage.restype = None

        lib.GetLongleyRiceReliabilityPercentage.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceReliabilityPercentage.restype = ctypes.c_double

        lib.SetLongleyRiceModeOfVariability.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetLongleyRiceModeOfVariability.restype = None

        lib.GetLongleyRiceModeOfVariability.argtypes = [ctypes.c_void_p]
        lib.GetLongleyRiceModeOfVariability.restype = ctypes.c_int


        # ITU-R P.1812 propagation model parameters

        lib.SetITURP1812TimePercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP1812TimePercentage.restype = None

        lib.GetITURP1812TimePercentage.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812TimePercentage.restype = ctypes.c_double

        lib.SetITURP1812LocationPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP1812LocationPercentage.restype = None

        lib.GetITURP1812LocationPercentage.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812LocationPercentage.restype = ctypes.c_double

        lib.SetITURP1812AverageRadioRefractivityLapseRate.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP1812AverageRadioRefractivityLapseRate.restype = None

        lib.GetITURP1812AverageRadioRefractivityLapseRate.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812AverageRadioRefractivityLapseRate.restype = ctypes.c_double

        lib.SetITURP1812SeaLevelSurfaceRefractivity.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP1812SeaLevelSurfaceRefractivity.restype = None

        lib.GetITURP1812SeaLevelSurfaceRefractivity.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812SeaLevelSurfaceRefractivity.restype = ctypes.c_double

        lib.SetITURP1812PredictionResolution.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP1812PredictionResolution.restype = None

        lib.GetITURP1812PredictionResolution.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812PredictionResolution.restype = ctypes.c_double

        lib.SetITURP1812RepresentativeClutterHeight.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.SetITURP1812RepresentativeClutterHeight.restype = None

        lib.GetITURP1812RepresentativeClutterHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetITURP1812RepresentativeClutterHeight.restype = ctypes.c_double

        lib.SetITURP1812RadioClimaticZonesFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.SetITURP1812RadioClimaticZonesFile.restype = None

        lib.GetITURP1812RadioClimaticZonesFile.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812RadioClimaticZonesFile.restype = ctypes.c_char_p

        lib.SetITURP1812LandCoverMappingType.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP1812LandCoverMappingType.restype = None

        lib.GetITURP1812LandCoverMappingType.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812LandCoverMappingType.restype = ctypes.c_int

        lib.SetITURP1812SurfaceProfileMethod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP1812SurfaceProfileMethod.restype = None

        lib.GetITURP1812SurfaceProfileMethod.argtypes = [ctypes.c_void_p]
        lib.GetITURP1812SurfaceProfileMethod.restype = ctypes.c_int


        # ITU-R P.452 propagation model parameters

        lib.SetITURP452TimePercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP452TimePercentage.restype = None

        lib.GetITURP452TimePercentage.argtypes = [ctypes.c_void_p]
        lib.GetITURP452TimePercentage.restype = ctypes.c_double

        lib.SetITURP452PredictionType.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP452PredictionType.restype = None

        lib.GetITURP452PredictionType.argtypes = [ctypes.c_void_p]
        lib.GetITURP452PredictionType.restype = ctypes.c_int

        lib.SetITURP452AverageRadioRefractivityLapseRate.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP452AverageRadioRefractivityLapseRate.restype = None

        lib.GetITURP452AverageRadioRefractivityLapseRate.argtypes = [ctypes.c_void_p]
        lib.GetITURP452AverageRadioRefractivityLapseRate.restype = ctypes.c_double

        lib.SetITURP452SeaLevelSurfaceRefractivity.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP452SeaLevelSurfaceRefractivity.restype = None

        lib.GetITURP452SeaLevelSurfaceRefractivity.argtypes = [ctypes.c_void_p]
        lib.GetITURP452SeaLevelSurfaceRefractivity.restype = ctypes.c_double

        lib.SetITURP452AirTemperature.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP452AirTemperature.restype = None

        lib.GetITURP452AirTemperature.argtypes = [ctypes.c_void_p]
        lib.GetITURP452AirTemperature.restype = ctypes.c_double

        lib.SetITURP452AirPressure.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP452AirPressure.restype = None

        lib.GetITURP452AirPressure.argtypes = [ctypes.c_void_p]
        lib.GetITURP452AirPressure.restype = ctypes.c_double       

        lib.SetITURP452RadioClimaticZonesFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.SetITURP452RadioClimaticZonesFile.restype = None

        lib.GetITURP452RadioClimaticZonesFile.argtypes = [ctypes.c_void_p]
        lib.GetITURP452RadioClimaticZonesFile.restype = ctypes.c_char_p

        lib.SetITURP452HeightGainModelClutterValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.SetITURP452HeightGainModelClutterValue.restype = None

        lib.GetITURP452HeightGainModelClutterValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.GetITURP452HeightGainModelClutterValue.restype = ctypes.c_double

        lib.SetITURP452HeightGainModelMode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.SetITURP452HeightGainModelMode.restype = None

        lib.GetITURP452HeightGainModelMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetITURP452HeightGainModelMode.restype = ctypes.c_int

        lib.SetITURP452RepresentativeClutterHeight.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.SetITURP452RepresentativeClutterHeight.restype = None

        lib.GetITURP452RepresentativeClutterHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetITURP452RepresentativeClutterHeight.restype = ctypes.c_double

        lib.SetITURP452LandCoverMappingType.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP452LandCoverMappingType.restype = None

        lib.GetITURP452LandCoverMappingType.argtypes = [ctypes.c_void_p]
        lib.GetITURP452LandCoverMappingType.restype = ctypes.c_int

        lib.SetITURP452SurfaceProfileMethod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP452SurfaceProfileMethod.restype = None

        lib.GetITURP452SurfaceProfileMethod.argtypes = [ctypes.c_void_p]
        lib.GetITURP452SurfaceProfileMethod.restype = ctypes.c_int


        # Extended Hata propagation model parameters

        lib.SetEHataClutterEnvironment.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetEHataClutterEnvironment.restype = None

        lib.GetEHataClutterEnvironment.argtypes = [ctypes.c_void_p]
        lib.GetEHataClutterEnvironment.restype = ctypes.c_int

        lib.SetEHataReliabilityPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetEHataReliabilityPercentage.restype = None

        lib.GetEHataReliabilityPercentage.argtypes = [ctypes.c_void_p]
        lib.GetEHataReliabilityPercentage.restype = ctypes.c_double


        # ITU-R P.2108 statistical clutter loss model for terrestrial paths

        lib.SetITURP2108TerrestrialStatModelActiveState.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.SetITURP2108TerrestrialStatModelActiveState.restype = None

        lib.GetITURP2108TerrestrialStatModelActiveState.argtypes = [ctypes.c_void_p]
        lib.GetITURP2108TerrestrialStatModelActiveState.restype = ctypes.c_bool

        lib.SetITURP2108TerrestrialStatModelLocationPercentage.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP2108TerrestrialStatModelLocationPercentage.restype = None

        lib.GetITURP2108TerrestrialStatModelLocationPercentage.argtypes = [ctypes.c_void_p]
        lib.GetITURP2108TerrestrialStatModelLocationPercentage.restype = ctypes.c_double

        lib.GetITURP2108TerrestrialStatModelLoss.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GetITURP2108TerrestrialStatModelLoss.restype = ctypes.c_double


        # ITU-R P.2109 building entry loss model

        lib.SetITURP2109ActiveState.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.SetITURP2109ActiveState.restype = None
        
        lib.GetITURP2109ActiveState.argtypes = [ctypes.c_void_p]
        lib.GetITURP2109ActiveState.restype = ctypes.c_bool

        lib.SetITURP2109Probability.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetITURP2109Probability.restype = None
        
        lib.GetITURP2109Probability.argtypes = [ctypes.c_void_p]
        lib.GetITURP2109Probability.restype = ctypes.c_double

        lib.SetITURP2109DefaultBuildingType.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetITURP2109DefaultBuildingType.restype = None

        lib.GetITURP2109DefaultBuildingType.argtypes = [ctypes.c_void_p]
        lib.GetITURP2109DefaultBuildingType.restype = ctypes.c_int

        lib.GetITURP2109BuildingEntryLoss.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GetITURP2109BuildingEntryLoss.restype = ctypes.c_double


        # ITU-R P.676 gaseous attenuation model for terrestrial paths

        lib.SetITURP676TerrPathGaseousAttenuationActiveState.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.SetITURP676TerrPathGaseousAttenuationActiveState.restype = None

        lib.GetITURP676TerrPathGaseousAttenuationActiveState.argtypes = [ctypes.c_void_p]
        lib.GetITURP676TerrPathGaseousAttenuationActiveState.restype = ctypes.c_bool

        lib.GetITURP676GaseousAttenuation.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.GetITURP676GaseousAttenuation.restype = ctypes.c_double


        # ITU digial maps

        lib.GetITUDigitalMapValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        lib.GetITUDigitalMapValue.restype = ctypes.c_double


        # Terrain elevation data parameters

        lib.SetPrimaryTerrainElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetPrimaryTerrainElevDataSource.restype = None

        lib.GetPrimaryTerrainElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetPrimaryTerrainElevDataSource.restype = ctypes.c_int

        lib.SetSecondaryTerrainElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetSecondaryTerrainElevDataSource.restype = None

        lib.GetSecondaryTerrainElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetSecondaryTerrainElevDataSource.restype = ctypes.c_int

        lib.SetTertiaryTerrainElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetTertiaryTerrainElevDataSource.restype = None

        lib.GetTertiaryTerrainElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetTertiaryTerrainElevDataSource.restype = ctypes.c_int

        lib.SetTerrainElevDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]
        lib.SetTerrainElevDataSourceDirectory.restype = None

        lib.GetTerrainElevDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetTerrainElevDataSourceDirectory.restype = ctypes.c_char_p

        lib.SetTerrainElevDataSamplingResolution.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.SetTerrainElevDataSamplingResolution.restype = None

        lib.GetTerrainElevDataSamplingResolution.argtypes = [ctypes.c_void_p]
        lib.GetTerrainElevDataSamplingResolution.restype = ctypes.c_double

        lib.SetTerrainElevDataSourceSamplingMethod.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.SetTerrainElevDataSourceSamplingMethod.restype = None

        lib.GetTerrainElevDataSourceSamplingMethod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetTerrainElevDataSourceSamplingMethod.restype = ctypes.c_int

        lib.AddCustomTerrainElevData.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_bool, ctypes.c_float]
        lib.AddCustomTerrainElevData.restype = ctypes.c_bool

        lib.ClearCustomTerrainElevData.argtypes = [ctypes.c_void_p]
        lib.ClearCustomTerrainElevData.restype = None

        lib.GetTerrainElevation.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.GetTerrainElevation.restype = ctypes.c_double


        # Land cover data parameters

        lib.SetPrimaryLandCoverDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetPrimaryLandCoverDataSource.restype = None

        lib.GetPrimaryLandCoverDataSource.argtypes = [ctypes.c_void_p]
        lib.GetPrimaryLandCoverDataSource.restype = ctypes.c_int

        lib.SetSecondaryLandCoverDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetSecondaryLandCoverDataSource.restype = None

        lib.GetSecondaryLandCoverDataSource.argtypes = [ctypes.c_void_p]
        lib.GetSecondaryLandCoverDataSource.restype = ctypes.c_int

        lib.SetLandCoverDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]
        lib.SetLandCoverDataSourceDirectory.restype = None

        lib.GetLandCoverDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetLandCoverDataSourceDirectory.restype = ctypes.c_char_p

        lib.AddCustomLandCoverData.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_bool, ctypes.c_short]
        lib.AddCustomLandCoverData.restype = ctypes.c_bool

        lib.ClearCustomLandCoverData.argtypes = [ctypes.c_void_p]
        lib.ClearCustomLandCoverData.restype = None

        lib.GetLandCoverClass.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GetLandCoverClass.restype = ctypes.c_int

        lib.GetLandCoverClassMappedValue.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        lib.GetLandCoverClassMappedValue.restype = ctypes.c_int

        lib.SetLandCoverClassMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.SetLandCoverClassMapping.restype = None

        lib.GetLandCoverClassMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.GetLandCoverClassMapping.restype = ctypes.c_int

        lib.SetDefaultLandCoverClassMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.SetDefaultLandCoverClassMapping.restype = None

        lib.GetDefaultLandCoverClassMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.GetDefaultLandCoverClassMapping.restype = ctypes.c_int

        lib.ClearLandCoverClassMappings.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.ClearLandCoverClassMappings.restype = None


        # Surface elevation data parameters

        lib.SetPrimarySurfaceElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetPrimarySurfaceElevDataSource.restype = None

        lib.GetPrimarySurfaceElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetPrimarySurfaceElevDataSource.restype = ctypes.c_int

        lib.SetSecondarySurfaceElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetSecondarySurfaceElevDataSource.restype = None

        lib.GetSecondarySurfaceElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetSecondarySurfaceElevDataSource.restype = ctypes.c_int

        lib.SetTertiarySurfaceElevDataSource.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetTertiarySurfaceElevDataSource.restype = None

        lib.GetTertiarySurfaceElevDataSource.argtypes = [ctypes.c_void_p]
        lib.GetTertiarySurfaceElevDataSource.restype = ctypes.c_int

        lib.SetSurfaceElevDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]
        lib.SetSurfaceElevDataSourceDirectory.restype = None

        lib.GetSurfaceElevDataSourceDirectory.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetSurfaceElevDataSourceDirectory.restype = ctypes.c_char_p

        lib.SetSurfaceAndTerrainDataSourcePairing.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.SetSurfaceAndTerrainDataSourcePairing.restype = None

        lib.GetSurfaceAndTerrainDataSourcePairing.argtypes = [ctypes.c_void_p]
        lib.GetSurfaceAndTerrainDataSourcePairing.restype = ctypes.c_bool

        lib.SetSurfaceElevDataSourceSamplingMethod.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.SetSurfaceElevDataSourceSamplingMethod.restype = None

        lib.GetSurfaceElevDataSourceSamplingMethod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetSurfaceElevDataSourceSamplingMethod.restype = ctypes.c_int

        lib.AddCustomSurfaceElevData.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_bool, ctypes.c_float]
        lib.AddCustomSurfaceElevData.restype = ctypes.c_bool

        lib.ClearCustomSurfaceElevData.argtypes = [ctypes.c_void_p]
        lib.ClearCustomSurfaceElevData.restype = None

        lib.GetSurfaceElevation.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.GetSurfaceElevation.restype = ctypes.c_double


        # Reception area parameters

        lib.SetReceptionAreaCorners.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.SetReceptionAreaCorners.restype = None

        lib.GetReceptionAreaLowerLeftCornerLatitude.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaLowerLeftCornerLatitude.restype = ctypes.c_double

        lib.GetReceptionAreaLowerLeftCornerLongitude.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaLowerLeftCornerLongitude.restype = ctypes.c_double

        lib.GetReceptionAreaUpperRightCornerLatitude.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaUpperRightCornerLatitude.restype = ctypes.c_double

        lib.GetReceptionAreaUpperRightCornerLongitude.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaUpperRightCornerLongitude.restype = ctypes.c_double

        lib.SetReceptionAreaNumHorizontalPoints.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetReceptionAreaNumHorizontalPoints.restype = None

        lib.GetReceptionAreaNumHorizontalPoints.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaNumHorizontalPoints.restype = ctypes.c_int

        lib.SetReceptionAreaNumVerticalPoints.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetReceptionAreaNumVerticalPoints.restype = None

        lib.GetReceptionAreaNumVerticalPoints.argtypes = [ctypes.c_void_p]
        lib.GetReceptionAreaNumVerticalPoints.restype = ctypes.c_int


        # Result type parameters

        lib.SetResultType.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetResultType.restype = None

        lib.GetResultType.argtypes = [ctypes.c_void_p]
        lib.GetResultType.restype = ctypes.c_int


        # Coverage display parameters for vector files (.mif and .kml)

        lib.ClearCoverageDisplayFills.argtypes = [ctypes.c_void_p]
        lib.ClearCoverageDisplayFills.restype = None

        lib.AddCoverageDisplayFill.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        lib.AddCoverageDisplayFill.restype = None

        lib.GetCoverageDisplayNumFills.argtypes = [ctypes.c_void_p]
        lib.GetCoverageDisplayNumFills.restype = ctypes.c_int

        lib.GetCoverageDisplayFillFromValue.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetCoverageDisplayFillFromValue.restype = ctypes.c_double

        lib.GetCoverageDisplayFillToValue.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetCoverageDisplayFillToValue.restype = ctypes.c_double

        lib.GetCoverageDisplayFillColor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GetCoverageDisplayFillColor.restype = ctypes.c_int


        # Generating and accessing results

        lib.GenerateReceptionPointResult.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GenerateReceptionPointResult.restype = ctypes.c_double

        lib.GenerateReceptionPointDetailedResult.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GenerateReceptionPointDetailedResult.restype = ReceptionPointDetailedResult

        lib.GenerateProfileReceptionPointResult.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        lib.GenerateProfileReceptionPointResult.restype = ctypes.c_double

        lib.GenerateReceptionAreaResults.argtypes = [ctypes.c_void_p]
        lib.GenerateReceptionAreaResults.restype = None

        lib.GetGenerateStatus.argtypes = [ctypes.c_void_p]
        lib.GetGenerateStatus.restype = ctypes.c_int

        lib.GetReceptionAreaResultValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.GetReceptionAreaResultValue.restype = ctypes.c_double

        lib.SetReceptionAreaResultValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.SetReceptionAreaResultValue.restype = None

        lib.GetReceptionAreaResultValueAtLatLon.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        lib.GetReceptionAreaResultValueAtLatLon.restype = ctypes.c_double

        lib.GetReceptionAreaResultLatitude.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.GetReceptionAreaResultLatitude.restype = ctypes.c_double

        lib.GetReceptionAreaResultLongitude.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.GetReceptionAreaResultLongitude.restype = ctypes.c_double

        lib.ExportReceptionAreaResultsToTextFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ExportReceptionAreaResultsToTextFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaResultsToMifFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ExportReceptionAreaResultsToMifFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaResultsToKmlFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
        lib.ExportReceptionAreaResultsToKmlFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaResultsToBilFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.ExportReceptionAreaResultsToBilFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaTerrainElevationToBilFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        lib.ExportReceptionAreaTerrainElevationToBilFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaLandCoverClassesToBilFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        lib.ExportReceptionAreaLandCoverClassesToBilFile.restype = ctypes.c_bool

        lib.ExportReceptionAreaSurfaceElevationToBilFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        lib.ExportReceptionAreaSurfaceElevationToBilFile.restype = ctypes.c_bool

        lib.ExportProfilesToCsvFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
        lib.ExportProfilesToCsvFile.restype = ctypes.c_bool



_covlib_cdll = _load_covlib_library()
