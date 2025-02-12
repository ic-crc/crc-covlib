[Français](./README_FR.md)

# Python examples

## Requirements

Some third-party python packages are required to run some of the examples. You may install these packages from the _requirements.txt_ file.

```bash
py -m pip install -r requirements.txt        # on Windows

python3 -m pip install -r requirements.txt   # on Linux
```

## Description of examples

### antenna-pattern
Example showing usage of an antenna pattern file and other antenna parameters.

### area-results-comparison
Example comparing results from two different simulations. The reception area of a third Simulation object is used to store and export the difference between the two simulations.

### hello-covlib
Simplest example showing how to instantiate the Simulation class and call one of its methods.

### iturm2101
Example showing how to generate and save a beamforming antenna pattern (ITU-R M.2101-0 recommendation, section 5) using crc-covlib's helper sub-package.

### iturp1411
Various examples using propagation models from ITU-R P.1411 along with street and building data from OpenStreetMap.

### iturp1812-landcover
Example using the ITU-R P.1812 propagation model to generate a coverage. It uses both terrain elevation (HRDEM DTM) and land cover (ESA WorldCover) data.

### iturp1812-surface
Another example using the ITU-R P.1812 propagation model to generate a coverage, but this one uses surface elevation data (following the method described in Annex 1, section 3.2.2 of ITU-R P.1812-7) instead of land cover data.

### iturp452v17
Example using the ITU-R P.452-17 propagation model to generate a coverage. It uses both terrain elevation (CDEM) and land cover (ESA WorldCover) data.

### iturp452v18
Example using the ITU-R P.452-18 propagation model to generate a coverage. It uses both terrain elevation (CDEM) and land cover (ESA WorldCover) data.

### line-of-sight
Evaluate whether propagation paths are line-of-sight or non-line-of-sight using crc-covlib's helper sub-package.

### local-ray
Elapsed time comparison between running a few coverage simulations sequentially and in parallel using [Ray](https://www.ray.io/) on the local machine.

### numba-jit-compiling
Some functionalities from the helper sub-package may be translated into optimized machine code using [Numba](https://numba.pydata.org/). Numba's just in time (JIT) compiling is disabled by default in crc-covlib but this example shows how to enable it.

### overview
Example using most of the Simulation object's methods. Shows how each method can be called (input parameters, returned values).

### profiles
Examples exporting calculated results and terrain profiles to a .csv file and plotting those profiles using matplotlib. Also shows how to calculate a result (path loss, field strength, etc.) using custom terrain elevation, land cover or surface elevation profiles.

### secondary-terrain-elev-source
Example demonstrating the use of a secondary terrain elevation data source when generating a coverage. When terrain elevation data is missing from the primary source, the secondary source is automatically used as a backup mechanism.

### terrain-elev-sources
Set of simple examples generating Longley-Rice coverages using various terrain elevation data sources:
- Canadian Digital Elevation Model (CDEM).
- crc-covlib's custom format.
- High Resolution Digital Elevation Model (HRDEM) from Natural Resources Canada.
- NASA's Shuttle Radar Topography Mission (SRTM) data.
- Other data in GeoTIFF file format.

### topography-exports
Verify that crc-covlib is reading terrain elevation, land cover and surface elevation correctly by exporting read data to raster files (.bil format).

### topography-helper
Examples showing the use of the topography module from crc-covlib's helper sub-package. This module uses the third-party packages rasterio and numpy to support the reading of a large variety of raster file formats. It shows how various terrain elevation, land cover and surface elevation data files may be read and used for simulations by passing their content to crc-covlib as custom data.
