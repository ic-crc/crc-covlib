# itur_data helper module
This module accesses the International Telecommunication Union (ITU) website to download data files for the user's personal use.

```python
from crc_covlib.helper import itur_data
```

- [DownloadCoreDigitalMaps](#downloadcoredigitalmaps)
- [DownloadITURP453Data](#downloaditurp453data)
- [DownloadITURP530Data](#downloaditurp530data)
- [DownloadITURP676Data](#downloaditurp676data)
- [DownloadITURP837Data](#downloaditurp837data)
- [DownloadITURP839Data](#downloaditurp839data)
- [DownloadITURP840AnnualData](#downloaditurp840annualdata)
- [DownloadITURP840SingleMonthData](#downloaditurp840singlemonthdata)
- [DownloadITURP840MonthtlyData](#downloaditurp840singlemonthdata)
- [DownloadITURP1511Data](#downloaditurp1511data)
- [DownloadITURP2001Data](#downloaditurp2001data)
- [DownloadITURP2145AnnualData](#downloaditurp2145annualdata)
- [DownloadITURP2145SingleMonthData](#downloaditurp2145singlemonthdata)
- [DownloadITURP2145MonthtlyData](#downloaditurp2145monthtlydata)

***

### DownloadCoreDigitalMaps
#### crc_covlib.helper.itur_data.DownloadCoreDigitalMaps
```python
def DownloadCoreDigitalMaps(directory: str=None) -> None
```
Download ITU digital maps used by the simulation module.

Args:
- __directory__ (str|None): The digital maps files are installed in the specified direcory or in 'crc_covlib/data/itu_proprietary' if set to None.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP453Data
#### crc_covlib.helper.itur_data.DownloadITURP453Data
```python
def DownloadITURP453Data() -> None
```
Download and install ITU data files that are used by the itur_p453.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p453/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP530Data
#### crc_covlib.helper.itur_data.DownloadITURP530Data
```python
def DownloadITURP530Data() -> None
```
Download and install ITU data files that are used by the itur_p530.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p530/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP676Data
#### crc_covlib.helper.itur_data.DownloadITURP676Data
```python
def DownloadITURP676Data() -> None
```
Download and install ITU data files that are used by the itur_p676.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p676/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP837Data
#### crc_covlib.helper.itur_data.DownloadITURP837Data
```python
def DownloadITURP837Data() -> None
```
Download and install ITU data files that are used by the itur_p837.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p837/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP839Data
#### crc_covlib.helper.itur_data.DownloadITURP839Data
```python
def DownloadITURP839Data() -> None
```
Download and install ITU data files that are used by the itur_p839.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p839/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP840AnnualData
#### crc_covlib.helper.itur_data.DownloadITURP840AnnualData
```python
def DownloadITURP840AnnualData() -> None
```
Download and install the ITU annual statistics files that are used by the itur_p840.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p840/annual/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP840SingleMonthData
#### crc_covlib.helper.itur_data.DownloadITURP840SingleMonthData
```python
def DownloadITURP840SingleMonthData(month: int) -> None
```
Download and install the ITU monthly statistics files (for the specified month only) that are used by the itur_p840.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p840/monthly/'.

Args:
- __month__ (int): Month, from 1 (January) to 12 (December).

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP840MonthtlyData
#### crc_covlib.helper.itur_data.DownloadITURP840MonthtlyData
```python
def DownloadITURP840MonthtlyData() -> None
```
Download and install the ITU monthly statistics files that are used by the itur_p840.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p840/monthly/'. This requires **1.46 GB** of disk space.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP1511Data
#### crc_covlib.helper.itur_data.DownloadITURP1511Data
```python
def DownloadITURP1511Data() -> None
```
Download and install ITU data files that are used by the itur_p1511.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p1511/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP2001Data
#### crc_covlib.helper.itur_data.DownloadITURP2001Data
```python
def DownloadITURP2001Data() -> None
```
Download and install ITU data files that are used by the itur_p2001.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p2001/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP2145AnnualData
#### crc_covlib.helper.itur_data.DownloadITURP2145AnnualData
```python
def DownloadITURP2145AnnualData() -> None
```
Download and install the ITU annual statistics files that are used by the itur_p2145.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p2145/'.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP2145SingleMonthData
#### crc_covlib.helper.itur_data.DownloadITURP2145SingleMonthData
```python
def DownloadITURP2145SingleMonthData(month: int) -> None
```
Download and install the ITU monthly statistics files (for the specified month only) that are used by the itur_p2145.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p2145/'.

Args:
- __month__ (int): Month, from 1 (January) to 12 (December).

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***

### DownloadITURP2145MonthtlyData
#### crc_covlib.helper.itur_data.DownloadITURP2145MonthtlyData
```python
def DownloadITURP2145MonthtlyData() -> None
```
Download and install the ITU monthly statistics files that are used by the itur_p2145.py module. Files are installed in 'crc_covlib/helper/data/itu_proprietary/p2145/'. This requires **7.56 GB** of disk space.

[Back to top](#itur_data-helper-module) | [Back to main index](./readme.md#helper-sub-package-api-documentation)

***