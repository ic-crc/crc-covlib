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

    print('\ncrc-covlib - Topography Exports')

    sim = covlib.Simulation()

    # Set elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)

    # Export terrain elevation data covering the reception area (1000 x 1000 points).
    # The output is a .bil file that can be visualized with a GIS application like QGIS.
    # This allows to verify that crc-covlib was able to read the terrain elevation data and to see whether any 
    # section of the terrain is missing.
    # Set the last parameter to True to use a value of 0 (zero) when no data is available (0 meter is the default 
    # elevation value used by crc-covlib in terrain profiles when no data is avaiblable).
    sim.ExportReceptionAreaTerrainElevationToBilFile(os.path.join(script_dir, 'cdem.bil'), 1000, 1000, True)

    # Set land cover data parameters
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    # Export land cover data covering the reception area.
    # Notice the last parameter (mapValues) is set to False. This instructs crc-covlib to export land cover
    # classes as they appear from the source (here ESA WorldCover file(s)), before any mapping is applied.
    sim.ExportReceptionAreaLandCoverClassesToBilFile(os.path.join(script_dir, 'worldcover-unmapped.bil'), 1000, 1000, False)

    # Select the ITU-R P.1812 propagation model and define the mapping between ESA WorldCover's classes
    # and P.1812's clutter categories. 
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)
    CLUT = covlib.P1812ClutterCategory
    sim.ClearLandCoverClassMappings(WORLDCOVER, P1812) # delete existing default mapping
    sim.SetLandCoverClassMapping(WORLDCOVER, 80, P1812, CLUT.P1812_WATER_SEA) # map 'Permanent water bodies' (80) to 'water/sea' (1)
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P1812, CLUT.P1812_OPEN_RURAL) # map all other ESA WorldCover classes to 'open/rural' (2)

    # The mapValues parameter is now set to True. This instructs crc-covlib to apply any defined mappings between
    # the land cover data and the currently selected propagation model before the export takes place.
    sim.ExportReceptionAreaLandCoverClassesToBilFile(os.path.join(script_dir, 'worldcover-mapped-to-P1812-clutter.bil'), 1000, 1000, True)

    # With P.1812, we can alternately map land cover classes the representative clutter heights instead of clutter
    # categories.
    sim.SetITURP1812LandCoverMappingType(covlib.P1812LandCoverMappingType.P1812_MAP_TO_REPR_CLUTTER_HEIGHT)
    sim.ClearLandCoverClassMappings(WORLDCOVER, P1812) # delete existing mapping
    sim.SetLandCoverClassMapping(WORLDCOVER, 10, P1812, 15) # map 'Tree cover' (10) to a representative clutter height of 15m
    sim.SetLandCoverClassMapping(WORLDCOVER, 50, P1812, 15) # map 'Built-up' (50) to a representative clutter height of 15m
    sim.SetDefaultLandCoverClassMapping(WORLDCOVER, P1812, 0) # map all other ESA WorldCover classes to a representative clutter height of 0m

    # The exported data will now contain the representative clutter heights.
    sim.ExportReceptionAreaLandCoverClassesToBilFile(os.path.join(script_dir, 'worldcover-mapped-to-repr-heights.bil'), 1000, 1000, True)

    # Set surface elevation data parameters
    CDSM = covlib.SurfaceElevDataSource.SURF_ELEV_NRCAN_CDSM
    sim.SetPrimarySurfaceElevDataSource(CDSM)
    sim.SetSurfaceElevDataSourceDirectory(CDSM, os.path.join(script_dir, '../../../data/surface-elev-samples/NRCAN_CDSM'))

    # Export surface elevation data covering the reception area (1000 x 1000 points).
    # Setting the last parameter to true will use a value of 0 (zero) when no data is available.
    # 0m surface height is the default value used by crc-covlib in terrain profiles when no data is avaiblable.
    sim.ExportReceptionAreaSurfaceElevationToBilFile(os.path.join(script_dir, 'cdsm.bil'), 1000, 1000, False)

    print('Topography exports completed\n')
