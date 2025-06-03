# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

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
    if directory is None:
        directory = os.path.join(_scriptDir, '..', 'data', 'itu_proprietary')

    filenames = ['DN50.TXT', 'N050.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1812-7-202308-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)

    filenames = ['T_Annual.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1510-1-201706-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)

    filenames = ['surfwv_50_fixed.txt']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2001-5-202308-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP453Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p453')
    filenames = ['NWET_Annual_50.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-I!!ZIP-E.zip'
        zipPathname1 = _Download(url, directory)
        zipPathname2 = _ExtractSpecific(zipPathname1, ['P.453_NWET_Maps.zip'], deleteZipArchive=True)[0]
        zipPathname3 = _ExtractSpecific(zipPathname2, ['P.453_NWET_Maps_Annual.zip'], deleteZipArchive=True)[0]
        _ExtractSpecific(zipPathname3, filenames, deleteZipArchive=True)[0]
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP530Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p530')
    filenames = ['dN75.csv', 'LogK.csv']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP676Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p676')
    filenames = ['R11010000020001TXTM.txt', 'R11010000020002TXTM.txt']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020001TXTM.txt'
        _Download(url, directory)
        url = 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020002TXTM.txt'
        _Download(url, directory)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP837Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p837')
    filenames = ['R001.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.837-7-201706-I!!ZIP-E.zip'
        zipPathname1 = _Download(url, directory)
        zipPathname2 = _ExtractSpecific(zipPathname1, ['R-REC-P.837-7-Maps.zip'], deleteZipArchive=True)[0]
        zipPathname3 = _ExtractSpecific(zipPathname2, ['P.837_R001_Maps.zip'], deleteZipArchive=True)[0]
        _ExtractSpecific(zipPathname3, filenames, deleteZipArchive=True)[0]
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP839Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p839')
    filenames = ['h0.txt']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP840AnnualData() -> None:
    directory = os.path.join(_defaultInstallDir, 'p840', 'annual')
    filenames = ['L_001.TXT', 'L_002.TXT', 'L_003.TXT', 'L_005.TXT', 'L_01.TXT', 'L_02.TXT', 'L_03.TXT',
                 'L_05.TXT', 'L_1.TXT', 'L_2.TXT', 'L_3.TXT', 'L_5.TXT', 'L_10.TXT', 'L_20.TXT', 'L_30.TXT',
                 'L_50.TXT', 'L_60.TXT', 'L_70.TXT', 'L_80.TXT', 'L_90.TXT', 'L_95.TXT', 'L_99.TXT',
                 'L_100.TXT', 'L_mean.TXT', 'L_std.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part01-0-202308-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractAll(zipPathname, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)

    filenames = ['mL.TXT', 'PL.TXT', 'sL.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part14-0-202308-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractAll(zipPathname, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP840SingleMonthData(month: int) -> None:
    """
    month from 1 to 12
    """
    monthNumStr = '{:02d}'.format(month)
    partNumStr = '{:02d}'.format(month+1)
    directory = os.path.join(_defaultInstallDir, 'p840', 'monthly', monthNumStr)
    filenames = ['L_01.TXT', 'L_02.TXT', 'L_03.TXT', 'L_05.TXT', 'L_1.TXT', 'L_2.TXT', 'L_3.TXT',
                 'L_5.TXT', 'L_10.TXT', 'L_20.TXT', 'L_30.TXT', 'L_50.TXT', 'L_60.TXT', 'L_70.TXT',
                 'L_80.TXT', 'L_90.TXT', 'L_95.TXT', 'L_99.TXT', 'L_100.TXT', 'L_mean.TXT', 'L_std.TXT']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840Part{}-0-202308-I!!ZIP-E.zip'.format(partNumStr)
        zipPathname = _Download(url, directory)
        _ExtractAll(zipPathname, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP840MonthtlyData() -> None:
    for month in range(1, 12+1):
        DownloadITURP840SingleMonthData(month)


def DownloadITURP1511Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p1511')
    filenames = ['TOPO.dat']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1511-3-202408-I!!ZIP-E.zip'
        zipPathname1 = _Download(url, directory)
        zipPathname2 = _ExtractSpecific(zipPathname1, ['R-REC-P1511-3-1.zip'], True)[0]
        _ExtractSpecific(zipPathname2, filenames, deleteZipArchive=True)[0]
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP2001Data() -> None:
    directory = os.path.join(_defaultInstallDir, 'p2001')
    filenames = ['DN_Median.txt']
    if _FilesAllExist(directory, filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2001-5-202308-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, filenames, deleteZipArchive=True)
    else:
        _PrintAlreadyInstalled(directory, filenames)


def DownloadITURP2145AnnualData() -> None:
    directory = os.path.join(_defaultInstallDir, 'p2145')
    percents = ['001', '002', '003', '005', '01', '02', '03', '05', '1', '2', '3', '5', '10', '20',
                '30', '50', '60', '70', '80', '90', '95', '99']
    P_filenames = ['P_{}.TXT'.format(p) for p in percents] + ['P_mean.TXT', 'P_std.TXT', 'PSCH.TXT', 'Z_ground.TXT']
    RHO_filenames = ['RHO_{}.TXT'.format(p) for p in percents] + ['RHO_mean.TXT', 'RHO_std.TXT', 'VSCH.TXT', 'Z_ground.TXT']
    T_filenames = ['T_{}.TXT'.format(p) for p in percents] + ['T_mean.TXT', 'T_std.TXT', 'TSCH.TXT', 'Z_ground.TXT']
    V_filenames = ['V_{}.TXT'.format(p) for p in percents] + ['V_mean.TXT', 'V_std.TXT', 'VSCH.TXT', 'Z_ground.TXT']
    P_dir = os.path.join(directory, 'P_Annual')
    RHO_dir = os.path.join(directory, 'RHO_Annual')
    T_dir = os.path.join(directory, 'T_Annual')
    V_dir = os.path.join(directory, 'V_Annual')

    if _FilesAllExist(P_dir, P_filenames) == False or _FilesAllExist(RHO_dir, RHO_filenames) == False or \
       _FilesAllExist(T_dir, T_filenames) == False or _FilesAllExist(V_dir, V_filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part01-0-202208-I!!ZIP-E.zip'
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, ['Attribution&disclaimer.txt'], deleteZipArchive=False)
        zipPathnameList = _ExtractSpecific(zipPathname,
                                           ['P_Annual.zip', 'RHO_Annual.zip', 'T_Annual.zip', 'V_Annual.zip'],
                                           deleteZipArchive=True)
        for zipPathname in zipPathnameList:
            _ExtractAll(zipPathname, deleteZipArchive=True, createNewDir=True)
    else:
        _PrintAlreadyInstalled(P_dir, P_filenames)
        _PrintAlreadyInstalled(RHO_dir, RHO_filenames)
        _PrintAlreadyInstalled(T_dir, T_filenames)
        _PrintAlreadyInstalled(V_dir, V_filenames)

    # Weibull annual data
    Weibull_filenames = ['kV.TXT', 'lambdaV.TXT', 'VSCH.TXT', 'Z_ground.TXT']
    Weibull_dir = os.path.join(directory, 'Weibull_Annual')
    if _FilesAllExist(Weibull_dir, Weibull_filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part14-0-202208-I!!ZIP-E.zip'
        zipPathname1 = _Download(url, directory)
        zipPathname2 = _ExtractSpecific(zipPathname1, ['Weibull_Annual.zip'], deleteZipArchive=True)[0]
        _ExtractAll(zipPathname2, deleteZipArchive=True, createNewDir=True)
    else:
        _PrintAlreadyInstalled(Weibull_dir, Weibull_filenames)


def DownloadITURP2145SingleMonthData(month: int) -> None:
    """
    month from 1 to 12
    """
    directory = os.path.join(_defaultInstallDir, 'p2145')
    percents = ['01', '02', '03', '05', '1', '2', '3', '5', '10', '20', '30', '50', '60', '70',
                '80', '90', '95', '99']
    monthNumStr = '{:02d}'.format(month)
    partNumStr = '{:02d}'.format(month+1)
    P_filenames = ['P_{}.TXT'.format(p) for p in percents] + ['P_mean.TXT', 'P_std.TXT', 'PSCH.TXT', 'Z_ground.TXT']
    RHO_filenames = ['RHO_{}.TXT'.format(p) for p in percents] + ['RHO_mean.TXT', 'RHO_std.TXT', 'VSCH.TXT', 'Z_ground.TXT']
    T_filenames = ['T_{}.TXT'.format(p) for p in percents] + ['T_mean.TXT', 'T_std.TXT', 'TSCH.TXT', 'Z_ground.TXT']
    V_filenames = ['V_{}.TXT'.format(p) for p in percents] + ['V_mean.TXT', 'V_std.TXT', 'VSCH.TXT', 'Z_ground.TXT']
    P_dir = os.path.join(directory, 'P_Month{}'.format(monthNumStr))
    RHO_dir = os.path.join(directory, 'RHO_Month{}'.format(monthNumStr))
    T_dir = os.path.join(directory, 'T_Month{}'.format(monthNumStr))
    V_dir = os.path.join(directory, 'V_Month{}'.format(monthNumStr))

    if _FilesAllExist(P_dir, P_filenames) == False or _FilesAllExist(RHO_dir, RHO_filenames) == False or \
       _FilesAllExist(T_dir, T_filenames) == False or _FilesAllExist(V_dir, V_filenames) == False:
        url = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2145Part{}-0-202208-I!!ZIP-E.zip'.format(partNumStr)
        zipPathname = _Download(url, directory)
        _ExtractSpecific(zipPathname, ['Attribution&disclaimer.txt'], deleteZipArchive=False)
        zipPathnameList = _ExtractSpecific(zipPathname,
                                           ['P_Month{}.zip'.format(monthNumStr),
                                            'RHO_Month{}.zip'.format(monthNumStr),
                                            'T_Month{}.zip'.format(monthNumStr),
                                            'V_Month{}.zip'.format(monthNumStr)],
                                           deleteZipArchive=True)
        for zipPathname in zipPathnameList:
            _ExtractAll(zipPathname, deleteZipArchive=True, createNewDir=True)
    else:
        _PrintAlreadyInstalled(P_dir, P_filenames)
        _PrintAlreadyInstalled(RHO_dir, RHO_filenames)
        _PrintAlreadyInstalled(T_dir, T_filenames)
        _PrintAlreadyInstalled(V_dir, V_filenames)

    
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


def _FilesAllExist(directory: str, filenames: list[str]) -> bool:
    for filename in filenames:
        pathname = os.path.join(directory, filename)
        if os.path.isfile(pathname) == False:
            return False
    return True


def _PrintAlreadyInstalled(directory: str, filenames: list[str]) -> None:
    for filename in filenames:
        pathname = os.path.join(directory, filename)
        pathname = os.path.realpath(pathname)
        print('already installed: {}'.format(pathname))
