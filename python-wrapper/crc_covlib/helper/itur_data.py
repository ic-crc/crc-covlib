"""
NOTICE: This script accesses the International Telecommunication Union (ITU)
website to download data files for the user's personal use.

see https://www.itu.int/en/Pages/copyright.aspx
"""

import os
import sys
from typing import Union
import urllib.request
from zipfile import ZipFile 


__all__ = ['DownloadCoreDigitalMaps',
           'DownloadITURP453Data',
           'DownloadITURP530Data',
           'DownloadITURP676Data',
           'DownloadITURP837Data',
           'DownloadITURP839Data',
           'DownloadITURP840AnnualData',
           'DownloadITURP840SingleMonthData',
           'DownloadITURP840MonthtlyData',
           'DownloadITURP1511Data',
           'DownloadITURP2001Data',
           'DownloadITURP2145AnnualData',
           'DownloadITURP2145SingleMonthData',
           'DownloadITURP2145MonthtlyData']


_scriptDir = os.path.dirname(os.path.abspath(__file__))
_defaultInstallDir = os.path.join(_scriptDir, 'data', 'itu_proprietary')
        

def DownloadCoreDigitalMaps(directory: str=None) -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1812-7-202308-I!!ZIP-E.zip'
    if directory is None:
        directory = os.path.join(_scriptDir, '..', 'data', 'itu_proprietary')
    zipPathname = _Download(url, directory)
    _ExtractSpecific(zipPathname, ['DN50.TXT', 'N050.TXT'], deleteZipArchive=True)

    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1510-1-201706-I!!ZIP-E.zip'
    zipPathname = _Download(url, directory)
    _ExtractSpecific(zipPathname, ['T_Annual.TXT'], deleteZipArchive=True)

    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2001-5-202308-I!!ZIP-E.zip'
    zipPathname = _Download(url, directory)
    _ExtractSpecific(zipPathname, ['surfwv_50_fixed.txt'], deleteZipArchive=True)


def DownloadITURP453Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-I!!ZIP-E.zip'
    zipPathname1 = _Download(url, os.path.join(_defaultInstallDir, 'p453'))
    zipPathname2 = _ExtractSpecific(zipPathname1, ['P.453_NWET_Maps.zip'], deleteZipArchive=True)[0]
    zipPathname3 = _ExtractSpecific(zipPathname2, ['P.453_NWET_Maps_Annual.zip'], deleteZipArchive=True)[0]
    _ExtractSpecific(zipPathname3, ['NWET_Annual_50.TXT'], deleteZipArchive=True)[0]


def DownloadITURP530Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p530'))
    _ExtractSpecific(zipPathname, ['dN75.csv', 'LogK.csv'], deleteZipArchive=True)


def DownloadITURP676Data() -> None:
    url = 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020001TXTM.txt'
    _Download(url, os.path.join(_defaultInstallDir, 'p676'))
    url = 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020002TXTM.txt'
    _Download(url, os.path.join(_defaultInstallDir, 'p676'))


def DownloadITURP837Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.837-7-201706-I!!ZIP-E.zip'
    zipPathname1 = _Download(url, os.path.join(_defaultInstallDir, 'p837'))
    zipPathname2 = _ExtractSpecific(zipPathname1, ['R-REC-P.837-7-Maps.zip'], deleteZipArchive=True)[0]
    zipPathname3 = _ExtractSpecific(zipPathname2, ['P.837_R001_Maps.zip'], deleteZipArchive=True)[0]
    _ExtractSpecific(zipPathname3, ['R001.TXT'], deleteZipArchive=True)[0]


def DownloadITURP839Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p839'))
    _ExtractSpecific(zipPathname, ['h0.txt'], True)


def DownloadITURP840AnnualData() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part01-0-202308-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p840', 'annual'))
    _ExtractAll(zipPathname, deleteZipArchive=True)

    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part14-0-202308-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p840', 'annual'))
    _ExtractAll(zipPathname, deleteZipArchive=True)


def DownloadITURP840SingleMonthData(month: int) -> None:
    """
    month from 1 to 12
    """
    monthNumStr = '{:02d}'.format(month)
    partNumStr = '{:02d}'.format(month+1)
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part{}-0-202308-I!!ZIP-E.zip'.format(partNumStr)
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p840', 'monthly', monthNumStr))
    _ExtractAll(zipPathname, deleteZipArchive=True)


def DownloadITURP840MonthtlyData() -> None:
    for month in range(1, 12+1):
        DownloadITURP840SingleMonthData(month)


def DownloadITURP1511Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1511-3-202408-I!!ZIP-E.zip'
    zipPathname1 = _Download(url, os.path.join(_defaultInstallDir, 'p1511'))
    zipPathname2 = _ExtractSpecific(zipPathname1, ['R-REC-P1511-3-1.zip'], True)[0]
    _ExtractSpecific(zipPathname2, ['TOPO.dat'], deleteZipArchive=True)[0]


def DownloadITURP2001Data() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2001-5-202308-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p2001'))
    _ExtractSpecific(zipPathname, ['DN_Median.txt'], deleteZipArchive=True)


def DownloadITURP2145AnnualData() -> None:
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part01-0-202208-I!!ZIP-E.zip'
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p2145'))
    _ExtractSpecific(zipPathname, ['Attribution&disclaimer.txt'], deleteZipArchive=False)
    zipPathnameList = _ExtractSpecific(zipPathname,
                                       ['P_Annual.zip', 'RHO_Annual.zip', 'T_Annual.zip', 'V_Annual.zip'],
                                       deleteZipArchive=True)
    for zipPathname in zipPathnameList:
        _ExtractAll(zipPathname, deleteZipArchive=True, createNewDir=True)

    # Weibull annual data
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part14-0-202208-I!!ZIP-E.zip'
    zipPathname1 = _Download(url, os.path.join(_defaultInstallDir, 'p2145'))
    zipPathname2 = _ExtractSpecific(zipPathname1, ['Weibull_Annual.zip'], deleteZipArchive=True)[0]
    _ExtractAll(zipPathname2, deleteZipArchive=True, createNewDir=True)


def DownloadITURP2145SingleMonthData(month: int) -> None:
    """
    month from 1 to 12
    """
    monthNumStr = '{:02d}'.format(month)
    partNumStr = '{:02d}'.format(month+1)
    url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part{}-0-202208-I!!ZIP-E.zip'.format(partNumStr)
    zipPathname = _Download(url, os.path.join(_defaultInstallDir, 'p2145'))
    _ExtractSpecific(zipPathname, ['Attribution&disclaimer.txt'], deleteZipArchive=False)
    zipPathnameList = _ExtractSpecific(zipPathname,
                                       ['P_Month{}.zip'.format(monthNumStr),
                                        'RHO_Month{}.zip'.format(monthNumStr),
                                        'T_Month{}.zip'.format(monthNumStr),
                                        'V_Month{}.zip'.format(monthNumStr)],
                                        deleteZipArchive=True)
    for zipPathname in zipPathnameList:
        _ExtractAll(zipPathname, deleteZipArchive=True, createNewDir=True)

    
def DownloadITURP2145MonthtlyData() -> None:
    for month in range(1, 12+1):
        DownloadITURP2145SingleMonthData(month)   


def _HandleProgress(blocknum: int, blocksize: int, totalsize: int) -> Union[object, None]:
    barLength = 33
    progress = 0
    if totalsize > 0:
        progress = min(1, blocknum*blocksize/totalsize)
    block = int(barLength*progress)
    text = '\r[{}] {:.1f}% of {:.2f} MB  '.format( '#'*block + '-'*(barLength-block), progress*100, totalsize/1E6)
    sys.stdout.write(text)
    sys.stdout.flush()


def _Download(url: str, directory: str) -> Union[str, None]:
    os.makedirs(directory, exist_ok=True)
    filename = os.path.basename(url)
    pathname = os.path.join(directory, filename)
    print('\ndownloading {}'.format(url))
    _HandleProgress(0, 0, 0)
    try:
        newPathname, _ = urllib.request.urlretrieve(url, pathname, _HandleProgress)
    except:
        # recommendation version may have been superseded
        if url.find('-I!!ZIP') != -1:
            url = url.replace('-I!!ZIP', '-S!!ZIP')

            filename = os.path.basename(url)
            pathname = os.path.join(directory, filename)
            print('\ndownloading {}'.format(url))
            _HandleProgress(0, 0, 0)
            newPathname, _ = urllib.request.urlretrieve(url, pathname, _HandleProgress)
    print('')
    return newPathname

    
def _ExtractAll(zipPathname: str, deleteZipArchive: bool, createNewDir: bool=False) -> None:
    """
    If createNewDir is set to True, the zip archive is extracted into a new direcotry named after
    the zip archive's filename.
    """
    zipFilename = os.path.basename(zipPathname)
    outputDir = os.path.dirname(zipPathname)
    if createNewDir == True:
        zipFilenameNoExt, _ = os.path.splitext(zipFilename)
        outputDir = os.path.join(outputDir, zipFilenameNoExt)
        os.makedirs(outputDir, exist_ok=True)
    print('extracting all from {} into {}'.format(zipFilename, os.path.realpath(outputDir)))
    with ZipFile(zipPathname, 'r') as zObject: 
        zObject.extractall(path=outputDir)
    if deleteZipArchive:
        print('deleting {}'.format(zipPathname))
        os.remove(zipPathname)


def _ExtractSpecific(zipPathname: str, filenames: list[str], deleteZipArchive: bool) -> list[str]:
    extractedPathnames = []
    zipFilename = os.path.basename(zipPathname)
    dir = os.path.dirname(zipPathname)
    with ZipFile(zipPathname, 'r') as zObject:
        for filename in filenames:
            print('extracting {} from {} into {}'.format(filename, zipFilename, os.path.realpath(dir)))
            extractedPathnames.append(zObject.extract(member=filename, path=dir))
    if deleteZipArchive:
        print('deleting {}'.format(os.path.realpath(zipPathname)))
        os.remove(zipPathname)
    return extractedPathnames
