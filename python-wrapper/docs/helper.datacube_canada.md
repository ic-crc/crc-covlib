# datacube_canada helper module
Facilitate downloads of terrain, surface and land cover data from the Canada Centre for Mapping and
Earth Observation Data Cube Platform.

Publisher - Current Organization Name: Natural Resources Canada \
Licence: [Open Government Licence - Canada](https://open.canada.ca/en/open-government-licence-canada)

Additional links: \
https://datacube.services.geo.ca/en/index.html  \
https://datacube.services.geo.ca/stac/api/

```python
from crc_covlib.helper import datacube_canada 
```

- [LatLonBox (class)](#latlonbox) 
- [DownloadCdem](#downloadcdem)
- [DownloadCdsm](#downloadcdsm)
- [DownloadMrdemDtm](#downloadmrdemdtm)
- [DownloadMrdemDsm](#downloadmrdemdsm)
- [DownloadHrdemDtm1m](#downloadhrdemdtm1m)
- [DownloadHrdemDsm1m](#downloadhrdemdsm1m)
- [DownloadHrdemDtm2m](#downloadhrdemdtm2m)
- [DownloadHrdemDsm2m](#downloadhrdemdsm2m)
- [DownloadLandcover2020](#downloadlandcover2020)

***

### LatLonBox
#### crc_covlib.helper.datacube_canada.LatLonBox
```python
class LatLonBox(minLat:float, minLon:float, maxLat:float, maxLon:float)
```
Bounding box in geographical coordinates (WGS84 or EPSG 4326 assumed).
    
Attributes:
- __minLat__ (float): minimum latitude of bounding box (degrees).
- __minLon__ (float): minimum longitude of bounding box (degrees).
- __maxLat__ (float): maximum latitude of bounding box (degrees).
- __maxLon__ (float): maximum longutide of bounding box (degrees).

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadCdem
#### crc_covlib.helper.datacube_canada.DownloadCdem
```python
def DownloadCdem(bounds: LatLonBox, outputPathname: str) -> None
```
Downloads an extract from the Canadian Digital Elevation Model (CDEM). \
_**NOTE: CDEM is a legacy product, a better option is to use [DownloadMrdemDtm()](#downloadmrdemdtm) instead.**_

For more details on this product: \
https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadCdsm
#### crc_covlib.helper.datacube_canada.DownloadCdsm
```python
def DownloadCdsm(bounds: LatLonBox, outputPathname: str) -> None
```
Downloads an extract from the Canadian Digital Surface Model (CDSM). \
_**NOTE: CDSM is a legacy product, a better option is to use [DownloadMrdemDsm()](#downloadmrdemdsm) instead.**_

For more details on this product: \
https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadMrdemDtm
#### crc_covlib.helper.datacube_canada.DownloadMrdemDtm
```python
def DownloadMrdemDtm(bounds: LatLonBox, outputPathname: str) -> None
```
Downloads an extract from the Medium Resolution Digital Elevation Model (MRDEM) / Digital Terrain Model (DTM).

For more details on this product: \
https://open.canada.ca/data/en/dataset/18752265-bda3-498c-a4ba-9dfe68cb98da

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadMrdemDsm
#### crc_covlib.helper.datacube_canada.DownloadMrdemDsm
```python
def DownloadMrdemDsm(bounds: LatLonBox, outputPathname: str) -> None
```
Downloads an extract from the Medium Resolution Digital Elevation Model (MRDEM) / Digital Surface Model (DSM).

For more details on this product: \
https://open.canada.ca/data/en/dataset/18752265-bda3-498c-a4ba-9dfe68cb98da

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadHrdemDtm1m
#### crc_covlib.helper.datacube_canada.DownloadHrdemDtm1m
```python
def DownloadHrdemDtm1m(bounds: LatLonBox, outputPathname: str) -> list[str]
```
Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Terrain Model (DTM) at 1-meter resolution.

For more details on this product: \
https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

Returns:
- (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain one file (i.e. outputPathname). However when the specified bounds overlap more than one tile of the HRDEM mosaic product, one file for each overlapping tile will be created. Created files may contain some or even only "no data" values depending on the availability of data for the requested area.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadHrdemDsm1m
#### crc_covlib.helper.datacube_canada.DownloadHrdemDsm1m
```python
def DownloadHrdemDsm1m(bounds: LatLonBox, outputPathname: str) -> list[str]
```
Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Surface Model (DSM) at 1-meter resolution.

For more details on this product: \
https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

Returns:
- (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain one file (i.e. outputPathname). However when the specified bounds overlap more than one tile of the HRDEM mosaic product, one file for each overlapping tile will be created. Created files may contain some or even only "no data" values depending on the availability of data for the requested area.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadHrdemDtm2m
#### crc_covlib.helper.datacube_canada.DownloadHrdemDtm2m
```python
def DownloadHrdemDtm2m(bounds: LatLonBox, outputPathname: str) -> list[str]
```
Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Terrain Model (DTM) at 2-meter resolution.

For more details on this product: \
https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

Returns:
- (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain one file (i.e. outputPathname). However when the specified bounds overlap more than one tile of the HRDEM mosaic product, one file for each overlapping tile will be created. Created files may contain some or even only "no data" values depending on the availability of data for the requested area.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadHrdemDsm2m
#### crc_covlib.helper.datacube_canada.DownloadHrdemDsm2m
```python
def DownloadHrdemDsm2m(bounds: LatLonBox, outputPathname: str) -> list[str]
```
Downloads an extract from the High Resolution Digital Elevation Model (HRDEM) / Digital Surface Model (DSM) at 2-meter resolution.

For more details on this product: \
https://open.canada.ca/data/en/dataset/0fe65119-e96e-4a57-8bfe-9d9245fba06b

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

Returns:
- (list[str]): List of created file(s) (full path(s)). In most cases, the list will contain one file (i.e. outputPathname). However when the specified bounds overlap more than one tile of the HRDEM mosaic product, one file for each overlapping tile will be created. Created files may contain some or even only "no data" values depending on the availability of data for the requested area.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadLandcover2020
#### crc_covlib.helper.datacube_canada.DownloadLandcover2020
```python
def DownloadLandcover2020(bounds: LatLonBox, outputPathname: str) -> None
```
Downloads an extract from the 2020 Land Cover of Canada map.

For more details on this product: \
https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47

Args:
- __bounds__ (crc_covlib.helper.datacube_canada.LatLonBox): Geographical area to be extracted.
- __outputPathname__ (str): Destination file for the extract. The file and any missing directory will be created if non-existent, otherwise the file will be overwritten. The extract is saved in the GeoTIFF (.tif) format.

[Back to top](#datacube_canada-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***