/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoTIFFReader.h"
#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>

#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))


GeoTIFFReader::GeoTIFFFileInfo::GeoTIFFFileInfo()
{
	m_tiffPtr = nullptr;
	m_readBuf = nullptr;
	Clear();
}

GeoTIFFReader::GeoTIFFFileInfo::~GeoTIFFFileInfo()
{
	Close();
}

GeoTIFFReader::GeoTIFFFileInfo::GeoTIFFFileInfo(const GeoTIFFFileInfo& original)
{
	m_tiffPtr = nullptr;
	m_readBuf = nullptr;
	*this = original;
}

const GeoTIFFReader::GeoTIFFFileInfo& GeoTIFFReader::GeoTIFFFileInfo::operator=(const GeoTIFFFileInfo& original)
{
	if (&original == this)
		return *this;

	GeoRasterFileInfo::operator=(original);

	m_compression = original.m_compression;
	m_rowsPerStrip = original.m_rowsPerStrip;
	m_bitsPerSample = original.m_bitsPerSample;
	m_samplesPerPixel = original.m_samplesPerPixel;
	m_sampleFormat = original.m_sampleFormat;
	m_tileHeight = original.m_tileHeight;
	m_tileWidth = original.m_tileWidth;

	m_bytesPerStrip = original.m_bytesPerStrip;
	m_bytesPerTile = original.m_bytesPerTile;

	m_noDataValue = original.m_noDataValue;
	m_noDataValuePresent = original.m_noDataValuePresent;

	memcpy(m_ModelPixelScale, original.m_ModelPixelScale, 3*sizeof(double));
	memcpy(m_ModelTiepoint, original.m_ModelTiepoint, 6*sizeof(double));

	m_GTModelTypeGeoKey = original.m_GTModelTypeGeoKey;
	m_GTRasterTypeGeoKey = original.m_GTRasterTypeGeoKey;
	m_GeogAngularUnitsGeoKey = original.m_GeogAngularUnitsGeoKey;
	m_ProjectedCSTypeGeoKey = original.m_ProjectedCSTypeGeoKey;
	m_ProjLinearUnitsGeoKey = original.m_ProjLinearUnitsGeoKey;
	m_GeographicTypeGeoKey = original.m_GeographicTypeGeoKey;
	m_GeogTOWGS84GeoKey = original.m_GeogTOWGS84GeoKey;
	m_GTCitationGeoKey = original.m_GTCitationGeoKey;
	m_GeogCitationGeoKey = original.m_GeogCitationGeoKey;
	m_GeogSemiMajorAxisGeoKey = original.m_GeogSemiMajorAxisGeoKey;
	m_GeogInvFlatteningGeoKey = original.m_GeogInvFlatteningGeoKey;

	Close(); // do not copy file pointer and buffer
	m_cache = original.m_cache;

	return *this;
}

void GeoTIFFReader::GeoTIFFFileInfo::Close()
{
	if (m_readBuf != nullptr)
	{
		_TIFFfree(m_readBuf);
		m_readBuf = nullptr;
	}
	if (m_tiffPtr != nullptr)
	{
		TIFFClose(m_tiffPtr);
		m_tiffPtr = nullptr;
	}
}

void GeoTIFFReader::GeoTIFFFileInfo::Clear()
{
	GeoRasterFileInfo::Clear();

	m_compression = 0;
	m_noDataValue = INT16_MIN;
	m_noDataValuePresent = false;
	m_rowsPerStrip = 0;
	m_bitsPerSample = 0;
	m_samplesPerPixel = 0;
	m_sampleFormat = 0;
	m_tileHeight = 0;
	m_tileWidth = 0;

	m_bytesPerStrip = 0;
	m_bytesPerTile = 0;

	memset(m_ModelPixelScale, 0, 3 * sizeof(double));
	memset(m_ModelTiepoint, 0, 6 * sizeof(double));

	m_GTModelTypeGeoKey = 0;
	m_GTRasterTypeGeoKey = 0;
	m_GeogAngularUnitsGeoKey = 0;
	m_ProjectedCSTypeGeoKey = 0;
	m_ProjLinearUnitsGeoKey = 0;
	m_GeographicTypeGeoKey = 0;
	m_GeogTOWGS84GeoKey.resize(0);
	m_GTCitationGeoKey = "";
	m_GeogCitationGeoKey = "";
	m_GeogSemiMajorAxisGeoKey = 0;
	m_GeogInvFlatteningGeoKey = 0;

	Close();
	m_cache.Clear();
}

// returns false if reading the file is not currently supported by the GeoTIFFReader
bool GeoTIFFReader::GeoTIFFFileInfo::ValidateAndSynch()
{
	//Print();

	if(m_GeographicTypeGeoKey == 4326 || // WGS 84
	   m_GeographicTypeGeoKey == 4269 || // NAD83
	   m_GTCitationGeoKey.find("WGS 84 / UTM zone") == 0 ||
	   m_GTCitationGeoKey.find("NAD83 / UTM zone") == 0 )
	{
		m_applyDatumTransform = false;
	}
	else if(m_ProjectedCSTypeGeoKey == 3979 || // NAD83(CSRS) / Canada Atlas Lambert
	        m_GeographicTypeGeoKey == 4617 || // NAD83(CSRS) (ex. product: NRCAN CDEM)
	        m_GeographicTypeGeoKey == 4140 || // NAD83(CSRS98)
	        m_GTCitationGeoKey.find("NAD83(CSRS) / UTM zone") == 0 ) // (ex. product: NRCAN HRDEM)
	{
		m_applyDatumTransform = true;
		m_toWgs84HelmertParams.clear();
		m_toWgs84HelmertParams.insert(m_toWgs84HelmertParams.end(),
			// see https://epsg.io/3979 under TOWGS84[] from "OGC WKT" file
		    {-0.991, 1.9072, 0.5129, -1.25033E-07, -4.6785E-08, -5.6529E-08, 0});
	}
	else
		return false;

	if( m_samplesPerPixel != 1 )
		return false;

	if(m_bitsPerSample != 8 && m_bitsPerSample != 16 && m_bitsPerSample != 32 )
		return false;

	// 1 = unsigned integer, 2 = signed integer, 3 = floating point
	if( m_sampleFormat != 1 && m_sampleFormat != 2 && m_sampleFormat != 3)
		return false;

	if( m_GTModelTypeGeoKey == 1 ) // 1 = ModelTypeProjected
	{
		if( m_ProjLinearUnitsGeoKey != 9001 ) // 9001 = Linear_Meter
			return false;

		if( m_ProjectedCSTypeGeoKey == 3979 )
			m_coordSystem = EPSG_3979;
		else
		{
			m_coordSystem = UTM;

			std::size_t found = m_GTCitationGeoKey.find("UTM zone ");
			bool zoneAndHemFound = false;
			if (found != std::string::npos)
			{
				char c;
				found += 9; // length of "UTM zone "
				for(std::size_t i=0, curPos=found ; i<3 && curPos<m_GTCitationGeoKey.size() ; i++, curPos++)
				{
					c = m_GTCitationGeoKey[curPos];
					if( c == 'N' || c == 'S')
					{
						m_northp = (c=='N') ? true : false;
						m_zone = atoi(m_GTCitationGeoKey.substr(found).c_str());
						zoneAndHemFound = true;
						break;
					}
				}
			}
			if( zoneAndHemFound == false )
				return false;
		}
	}
	else if( m_GTModelTypeGeoKey == 2 ) // 2 = ModelTypeGeographic
	{
		m_coordSystem = GEOGRAPHIC;

		if ( m_GeogAngularUnitsGeoKey != 9102 ) // 9102 = Angular_Degree
			return false;
	}
	else
		return false;

	if( m_GTRasterTypeGeoKey == 1 ) // 1 = RasterPixelIsArea
	{
		m_pixelHeight = m_ModelPixelScale[1];
		m_pixelWidth = m_ModelPixelScale[0];

		// Note: the center of the top left "pixel" is at 
		//  lat = m_ModelTiepoint[1][1] - (m_ModelPixelScale[1]/2.0)
		//  lon = m_ModelTiepoint[1][0] + (m_ModelPixelScale[0]/2.0)
		m_topLimit = m_ModelTiepoint[1][1];
		m_bottomLimit = m_topLimit - (m_rasterHeight*m_pixelHeight);
		m_leftLimit = m_ModelTiepoint[1][0];
		m_rightLimit = m_leftLimit + (m_rasterWidth*m_pixelWidth);

		// help verifying with values from "listgeo -d -proj4 <filename>"
		/*
		if( m_coordSystem == UTM )
			std::cout << std::fixed << std::setprecision(3);
		else
			std::cout << std::fixed << std::setprecision(7);
		std::cout << "Upper Left  (" << m_leftLimit << "," << m_topLimit << ")" << std::endl;
		std::cout << "Lower Left  (" << m_leftLimit << "," << m_bottomLimit << ")" << std::endl;
		std::cout << "Upper right (" << m_rightLimit << "," << m_topLimit << ")" << std::endl;
		std::cout << "Lower Right (" << m_rightLimit << "," << m_bottomLimit << ")" << std::endl << std::endl;
		*/
	}
	else if( m_GTRasterTypeGeoKey == 2 ) // 2 = RasterPixelIsPoint
	{
		m_pixelHeight = m_ModelPixelScale[1];
		m_pixelWidth = m_ModelPixelScale[0];

		// Note: the center of the top left "pixel" is at 
		//  lat = m_ModelTiepoint[1][1]
		//  lon = m_ModelTiepoint[1][0]
		m_topLimit = m_ModelTiepoint[1][1] + (m_ModelPixelScale[1]/2.0);
		m_bottomLimit = m_topLimit - (m_rasterHeight*m_pixelHeight);
		m_leftLimit = m_ModelTiepoint[1][0] - (m_ModelPixelScale[0]/2.0);
		m_rightLimit = m_leftLimit + (m_rasterWidth*m_pixelWidth);
	}
	else
		return false;

	// Prevent reading any file that uses the JBIG compression (TAG values from https://en.wikipedia.org/wiki/TIFF).
	// This is done to ensure having libjbig (JBIG-KIT) usage fall under a "dependency licence" (see https://www.cl.cam.ac.uk/~mgk25/jbigkit/).
	// We have to link to libjbig (either statically or dynamically) as a requirement for libtiff. Doing so wihtout
	// falling under the "dependency licence" would force us the put the project under the GNU General Public License
	// if we want to distribute any binaries (.dll, .so).
	// Disclaimer: comment above comes from a developer, not from a lawyer.
	if( m_compression==0x9 || m_compression==0xA || m_compression==0x8765 || m_compression==0x879B )
		return false;

	return true;
}

void GeoTIFFReader::GeoTIFFFileInfo::Print()
{
	std::cout << "pathname: " << m_pathname << std::endl;
	std::cout << "No data value defined: " << ((m_noDataValuePresent) ? "yes" : "no") << std::endl;
	if(m_noDataValuePresent)
		std::cout << "No data value: " << m_noDataValue << std::endl;
	std::cout << "GTModelTypeGeoKey: " << m_GTModelTypeGeoKey;
	if( m_GTModelTypeGeoKey == 1 ) std::cout << " (ModelTypeProjected)" << std::endl;
	else if( m_GTModelTypeGeoKey == 2 ) std::cout << " (ModelTypeGeographic)" << std::endl;
	else if( m_GTModelTypeGeoKey == 2 ) std::cout << " (ModelTypeGeocentric)" << std::endl;
	else std::cout << " (?)" << std::endl;
	std::cout << "GeographicTypeGeoKey: " << m_GeographicTypeGeoKey << std::endl;
	std::cout << "ProjectedCSTypeGeoKey: " << m_ProjectedCSTypeGeoKey << std::endl;
	std::cout << "GTCitationGeoKey: " << m_GTCitationGeoKey << std::endl;
	std::cout << "GeogCitationGeoKey: " << m_GeogCitationGeoKey << std::endl << std::endl;
}



GeoTIFFReader::GeoTIFFReader()
{
	pDir = "";
	pFile = "";
	pLastTiffUsed = nullptr;

	//pCacheHitCount = 0;
	//pCacheMissCount = 0;
}

GeoTIFFReader::~GeoTIFFReader()
{
	//std::cout << pCacheHitCount << " cache hits" << std::endl;
	//std::cout << pCacheMissCount << " cache misses" << std::endl;
}

GeoTIFFReader::GeoTIFFReader(const GeoTIFFReader& original)
{
	*this = original;
}

const GeoTIFFReader& GeoTIFFReader::operator=(const GeoTIFFReader& original)
{
	if (&original == this)
		return *this;

	pDir = original.pDir;
	pFile = original.pFile;
	pGeoTiffs = original.pGeoTiffs;
	pUpdateRTree(); // needs to rebuild the R-Tree (and not copy it) since it stores pointers from pGeoTiffs
	pLastTiffUsed = nullptr;

	return *this;
}

void GeoTIFFReader::SetDirectory(const char* directory, bool useIndexFile/*=false*/, bool overwriteIndexFile/*=false*/)
{
	pDir = directory;
	pFile = "";

	if( useIndexFile == false )
		pUpdateFilesInfo(directory);
	else
	{
		if( overwriteIndexFile == true )
		{
			pUpdateFilesInfo(directory);
			pCreateIndexFile(true);
		}
		else
		{
			if( pReadIndexFile() == false )
			{
				pUpdateFilesInfo(directory);
				// Do not overwrite the index file if it exists, as failure to read it may have been
				// caused by too many opened files at the OS level, and other processes may be at
				// reading it.
				pCreateIndexFile(false);
			}
		}
	}

	pUpdateFilesCacheSettings();
	pUpdateRTree();
}

const char* GeoTIFFReader::GetDirectory() const
{
	return pDir.c_str();
}

void GeoTIFFReader::SetFile(const char* pathname)
{
GeoTIFFFileInfo tiffInfo;

	pFile = pathname;
	pDir = "";

	pGeoTiffs.clear();
	pLastTiffUsed = nullptr;
	if (pReadTagsAndKeys(pathname, tiffInfo) == true)
	{
		if( tiffInfo.ValidateAndSynch() == true )
			pGeoTiffs.push_back(tiffInfo);
	}
	pUpdateFilesCacheSettings();
	pUpdateRTree();
}

const char* GeoTIFFReader::GetFile() const
{
	return pFile.c_str();
}

bool GeoTIFFReader::pCreateIndexFile(bool overwriteIfExists)
{
std::string indexPathname = pDir + "/crc_covlib_geotiff_index";

	if( overwriteIfExists == false && fs::exists(indexPathname) == true )
		return true;

	if( pGeoTiffs.size() == 0 ) // if directory does not contain any supported geotiff
		return false;

	std::ofstream outfile;
	bool success = false;
	outfile.open(indexPathname.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	if (outfile)
	{
	size_t num = pGeoTiffs.size();
	int32_t indexVersion = GEOTIFF_INDEX_VERSION;

		outfile.write(reinterpret_cast<char*>(&indexVersion), sizeof(indexVersion));
		outfile.write(reinterpret_cast<char*>(&num), (std::streamsize) sizeof(num));
		for(size_t i=0 ; i<num ; i++)
			pSerializeTiffInfoFile(outfile, pGeoTiffs[i]);
		success = true;
	}
	outfile.close();
	return success;
}

bool GeoTIFFReader::pReadIndexFile()
{
std::ifstream infile;
bool success = false;
std::string indexPathname = pDir + "/crc_covlib_geotiff_index";
int geotiffIndexVersion = -1;

	infile.open(indexPathname.c_str(), std::ios::in | std::ios::binary);
	if(infile)
	{
		infile.read(reinterpret_cast<char*>(&geotiffIndexVersion), sizeof(geotiffIndexVersion));

		if( geotiffIndexVersion == GEOTIFF_INDEX_VERSION )
		{
			pGeoTiffs.clear();
			pLastTiffUsed = nullptr;
			size_t num;
			infile.read(reinterpret_cast<char*>(&num), sizeof(num));
			pGeoTiffs.resize(num);
			for(size_t i=0 ; i<num ; i++)
				pDeserializeTiffInfoFile(infile, pGeoTiffs[i]);
			success = true;
		}
	}
	infile.close();
	return success;
}

bool GeoTIFFReader::pReadTagsAndKeys(const char* pathname, GeoTIFFFileInfo& tiffInfo)
{
TIFF* tif = nullptr;
bool readOK = true;

	tiffInfo.Clear();
	tiffInfo.m_pathname = pathname;

	TIFFSetWarningHandler(nullptr);
	tif = TIFFOpen(pathname, "r");
	if( tif != nullptr )
	{
	uint16_t count = 0;
	char* dataStr = nullptr;
	uint16_t* data = nullptr;

		readOK &= (TIFFGetField(tif, TIFFTAG_COMPRESSION, &(tiffInfo.m_compression)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &(tiffInfo.m_rasterWidth)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &(tiffInfo.m_rasterHeight)) == 1);
		TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &(tiffInfo.m_rowsPerStrip)); // will not be present in tile-oriented tiffs
		readOK &= (TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &(tiffInfo.m_sampleFormat)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &(tiffInfo.m_samplesPerPixel)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &(tiffInfo.m_bitsPerSample)) == 1);
		if (TIFFGetField(tif, 42113, &count, &dataStr) == 1 && dataStr != nullptr)
		{
			try {
				tiffInfo.m_noDataValue = static_cast<int32_t>(std::stoi(dataStr)); // NOTE : dataStr may contain a float
				tiffInfo.m_noDataValuePresent = true;
			}
			catch (const std::exception& e) {
				// handle invalid string
				tiffInfo.m_noDataValue = INT16_MIN;
				tiffInfo.m_noDataValuePresent = false;
			}
		}
		TIFFGetField(tif, TIFFTAG_TILELENGTH, &(tiffInfo.m_tileHeight)); // will not be present in strip-oriented tiffs
		TIFFGetField(tif, TIFFTAG_TILEWIDTH, &(tiffInfo.m_tileWidth)); // will not be present in strip-oriented tiffs

		int64_t bytesPerStrip = TIFFStripSize(tif);
		if( bytesPerStrip <= INT32_MAX )
			tiffInfo.m_bytesPerStrip = static_cast<int32_t>(bytesPerStrip); // NOTE : could eventually support bigger strip size
		else
			readOK = false;
		int64_t bytesPerTile = TIFFTileSize(tif);
		if( bytesPerTile <= INT32_MAX )
			tiffInfo.m_bytesPerTile = static_cast<int32_t>(bytesPerTile); // NOTE : could eventually support bigger tile size
		else
			readOK = false;

		readOK &= pGetGeoTagDoubleArrayValue(tif, 33550, &(tiffInfo.m_ModelPixelScale[0]), 3);
		readOK &= pGetGeoTagDoubleArrayValue(tif, 33922, &(tiffInfo.m_ModelTiepoint[0][0]), 6);

		readOK &= (TIFFGetField(tif, 34735, &count, &data) == 1); // 34735 = GeoKeyDirectoryTag
		if( count % 4 == 0 )
		{
		uint16_t keyID = 0, tiffTagLocation = 0, numValues = 0, valueOffset = 0;

			for(uint32_t i=0 ; i<count ; i+=4)
			{
				keyID = data[i];
				tiffTagLocation = data[i+1];
				numValues = data[i+2];
				valueOffset = data[i+3];

				// see http://geotiff.maptools.org/spec/geotiff6.html
				if( tiffTagLocation == 0 )
				{
					switch(keyID)
					{
						case 1024:
							tiffInfo.m_GTModelTypeGeoKey = valueOffset;
							break;
						case 1025:
							tiffInfo.m_GTRasterTypeGeoKey = valueOffset;
							break;
						case 2054:
							tiffInfo.m_GeogAngularUnitsGeoKey = valueOffset;
							break;
						case 2048:
							tiffInfo.m_GeographicTypeGeoKey = valueOffset;
							break;
						case 3072:
							tiffInfo.m_ProjectedCSTypeGeoKey = valueOffset;
							break;
						case 3076:
							tiffInfo.m_ProjLinearUnitsGeoKey = valueOffset;
							break;
						default:
							break;
					}
				}
				else
				{
					switch(keyID)
					{
						case 1026:
							readOK &= pGetGeoKeyStringValue(tif, tiffTagLocation, numValues, valueOffset, tiffInfo.m_GTCitationGeoKey);
							break;
						case 2049:
							readOK &= pGetGeoKeyStringValue(tif, tiffTagLocation, numValues, valueOffset, tiffInfo.m_GeogCitationGeoKey);
							break;
						case 2062:
							readOK &= pGetGeoKeyDoubleArrayValue(tif, tiffTagLocation, numValues, valueOffset, tiffInfo.m_GeogTOWGS84GeoKey);
							break;
						case 2057:
							readOK &= pGetGeoKeyDoubleValue(tif, tiffTagLocation, numValues, valueOffset, tiffInfo.m_GeogSemiMajorAxisGeoKey);
							break;
						case 2059:
							readOK &= pGetGeoKeyDoubleValue(tif, tiffTagLocation, numValues, valueOffset, tiffInfo.m_GeogInvFlatteningGeoKey);
							break;
						default:
							break;
					}
				}
			}
		}
		else
			readOK = false;

		TIFFClose(tif);
	}
	else
		readOK = false;

	return readOK;
}

bool GeoTIFFReader::pGetGeoTagDoubleArrayValue(TIFF* tif, uint32_t tag, double* dst, uint32_t dstSize)
{
    uint16_t count = 0;  // libtiff uses uint16_t for count in most array tags
    double* data = nullptr;

    if (TIFFGetField(tif, tag, &count, &data) == 1 
        && data != nullptr 
        && count == dstSize)
    {
        memcpy(dst, data, dstSize * sizeof(double));
        return true;
    }
    return false;
}

bool GeoTIFFReader::pGetGeoKeyStringValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::string& dst)
{
    if (numValues == 0)
        return false;

    uint16_t count = 0;
    char* data = nullptr;

    if (TIFFGetField(tif, tiffTagLocation, &count, &data) == 1
        && data != nullptr
        && valueOffset + numValues <= count)
    {
        dst.assign(data + valueOffset, numValues - 1); // -1 so not to include null terminator
        return true;
    }
    return false;
}

bool GeoTIFFReader::pGetGeoKeyDoubleArrayValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::vector<double>& dst)
{
    if (numValues == 0)
        return false;

    uint16_t count = 0;
    double* data = nullptr;

    if (TIFFGetField(tif, tiffTagLocation, &count, &data) == 1
        && data != nullptr
        && valueOffset + numValues <= count)
    {
        dst.resize(numValues);
        memcpy(dst.data(), data + valueOffset, numValues*sizeof(double));
        return true;
    }
    return false;
}

bool GeoTIFFReader::pGetGeoKeyDoubleValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, double& dst)
{
    if (numValues != 1)
        return false;

    uint16_t count = 0;
    double* data = nullptr;

    if (TIFFGetField(tif, tiffTagLocation, &count, &data) == 1
        && data != nullptr
        && valueOffset < count)
    {
        dst = data[valueOffset];
        return true;
    }
    return false;
}

void GeoTIFFReader::pUpdateFilesInfo(const char* directory)
{
std::vector<std::string> tifPathnames = pGetPathnameList(directory, ".tif");
GeoTIFFFileInfo tiffInfo;

	pGeoTiffs.clear();
	pLastTiffUsed = nullptr;
	for (size_t i=0; i < tifPathnames.size(); i++)
	{
		if (pReadTagsAndKeys(tifPathnames[i].c_str(), tiffInfo) == true)
		{
			if( tiffInfo.ValidateAndSynch() == true )
				pGeoTiffs.push_back(tiffInfo);
		}
	}
}

void GeoTIFFReader::pUpdateFilesCacheSettings()
{
GeoTIFFReader::GeoTIFFFileInfo* tiffInfo;
uint16_t bytesPerSample;
int32_t stripOrTileSizeInBytes;

	for (size_t i=0; i < pGeoTiffs.size(); i++)
	{
		tiffInfo = &(pGeoTiffs[i]);

		bytesPerSample = tiffInfo->m_bitsPerSample/8;

		if( tiffInfo->m_rowsPerStrip > 0 )
			stripOrTileSizeInBytes = tiffInfo->m_bytesPerStrip;
		else
			stripOrTileSizeInBytes = tiffInfo->m_bytesPerTile;
		
		if (bytesPerSample <= UINT8_MAX && stripOrTileSizeInBytes >= 0)
		{
			tiffInfo->m_cache.SetCacheEntrySize(static_cast<uint8_t>(bytesPerSample),
			                                    static_cast<uint32_t>(stripOrTileSizeInBytes));
			tiffInfo->m_cache.SetTotalSizeLimit(UINT32_MAX);
		}
		else
			 throw std::runtime_error("Invalid value from GeoTIFF file.");
	}
}

void GeoTIFFReader::pUpdateRTree()
{
GeoTIFFReader::GeoTIFFFileInfo* tiffInfo;
double minLat, minLon, maxLat, maxLon;
double minNativeCoords[2];
double maxNativeCoords[2];

	pGeoTiffsRTree.RemoveAll();
	for (size_t i=0; i < pGeoTiffs.size(); i++)
	{
		tiffInfo = &(pGeoTiffs[i]);
		tiffInfo->GetWgs84BoundingBox(&minLat, &minLon, &maxLat, &maxLon);
		minNativeCoords[0] = minLon;
		minNativeCoords[1] = minLat;
		maxNativeCoords[0] = maxLon;
		maxNativeCoords[1] = maxLat;
		pGeoTiffsRTree.Insert(minNativeCoords, maxNativeCoords, tiffInfo);
	}
}

void GeoTIFFReader::CloseAllFiles(bool clearCaches)
{
	for (size_t i = 0; i < pGeoTiffs.size(); i++)
	{
		pGeoTiffs[i].Close();
		if(clearCaches)
			pGeoTiffs[i].m_cache.Clear();
	}
}

std::vector<std::string> GeoTIFFReader::pGetPathnameList(const char* directory, const char* fileExtension)
{
std::vector<std::string> result;
std::string requestedExt = fileExtension;

	pToLowercase(requestedExt);
	if( requestedExt[0] != '.' )
		requestedExt = '.' + requestedExt;
	try
	{
		for (const auto& p : fs::recursive_directory_iterator(directory))
		{
			if (!fs::is_directory(p))
			{
				std::string ext(p.path().extension().string());
				pToLowercase(ext);
				if (ext == requestedExt)
					result.push_back(p.path().string());
			}
		}
	}
	catch(const std::exception& e)
	{
	}
	return result;
}

void GeoTIFFReader::pToLowercase(std::string& s)
{
	for (size_t i = 0; i < s.length(); i++)
		s[i] = tolower(s[i]);
}

std::string GeoTIFFReader::pGetRelativePath(const char* baseDir, const char* pathname)
{
#ifdef _WIN32
    const char sep[] = "/\\:";
#else
    const char sep[] = "/";
#endif
std::string baseDirCopy = baseDir;
std::string pathnameCopy = pathname;
std::vector<std::string> tokens;
char* token = nullptr;
char* saveptr = nullptr; 
size_t searchFrom = 0;
size_t findResult;
std::string result = pathname;

#ifdef _WIN32
	// pathnames are case sensitive in Linux, but not in Windows
	pToLowercase(baseDirCopy);
	pToLowercase(pathnameCopy);

	#define strtok_r strtok_s
#endif

	token = strtok_r(baseDirCopy.data(), sep, &saveptr);
	while (token != nullptr)
	{
		tokens.push_back(token);
		token = strtok_r(nullptr, sep, &saveptr);
	}

	for (size_t i = 0; i < tokens.size(); i++)
	{
		findResult = pathnameCopy.find(tokens[i], searchFrom);
		if (findResult != std::string::npos)
			searchFrom = findResult + tokens[i].size() + 1;
	}

	result = result.substr(searchFrom);
	
#ifdef _WIN32
	std::replace(result.begin(), result.end(), '\\', '/');
#endif
	return result;
}

bool GeoTIFFReader::pCompareTiffInfoOnResolution(const GeoTIFFFileInfo* tiffInfo1, const GeoTIFFFileInfo* tiffInfo2)
{
	return (tiffInfo1->ResolutionInMeters() < tiffInfo2->ResolutionInMeters());
}

// Get list of GeoTIFFFileInfos that contain point (lat, lon), ordered by resolution (most precise to less precise).
std::vector<GeoTIFFReader::GeoTIFFFileInfo*> GeoTIFFReader::pGetGeoTiffFileInfoList(double lat, double lon)
{
std::vector<GeoTIFFFileInfo*> result;
double pt[2] = {lon, lat};

	auto SearchCallback = [lat, lon, &result] (GeoTIFFFileInfo* fileInfo) -> bool
	{
		// good to check with IsIn() since the WGS84 lat/lon box used in the R-Tree may encompass zones
		// that are not actually part of the file (if the file is in UTM coordinates for example)
		if( fileInfo->IsIn(lat, lon) == true )
			result.push_back(fileInfo);
		return true; // true to continue searching (in order to get all files containing the point)
	};
	pGeoTiffsRTree.Search(pt, pt, SearchCallback);
	sort(result.begin(), result.end(), pCompareTiffInfoOnResolution);
	return result;
}

bool GeoTIFFReader::GetClosestValue(double lat, double lon, void* value, double* closestPtLat/*=nullptr*/, double* closestPtLon/*=nullptr*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}

bool GeoTIFFReader::GetClosestIntValue(double lat, double lon, int* value, double* closestPtLat/*=nullptr*/, double* closestPtLon/*=nullptr*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestIntValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}

bool GeoTIFFReader::GetClosestFltValue(double lat, double lon, float* value, double* closestPtLat/*=nullptr*/, double* closestPtLon/*=nullptr*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestFltValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}
	
bool GeoTIFFReader::GetInterplValue(double lat, double lon, float* value)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetInterplFltValue;

	return pGetValue(lat, lon, value, nullptr, nullptr, f);
}

bool GeoTIFFReader::pGetValue(double lat, double lon, void* value,
							  double* closestPtLat, double* closestPtLon,
							  GetValueMemberFunc getValueFunc)
{
	if( pLastTiffUsed != nullptr && pLastTiffUsed->IsIn(lat, lon) == true )
		if( CALL_MEMBER_FN(*this, getValueFunc)(pLastTiffUsed, lat, lon, value, closestPtLat, closestPtLon) == true )
			return true;

	std::vector<GeoTIFFFileInfo*> tiffInfoList = pGetGeoTiffFileInfoList(lat, lon);
	GeoTIFFFileInfo* tiffInfo;
	for(size_t i=0 ; i<tiffInfoList.size() ; i++)
	{
		tiffInfo = tiffInfoList[i];
		if( tiffInfo != pLastTiffUsed )
		{
			if( CALL_MEMBER_FN(*this, getValueFunc)(tiffInfo, lat, lon, value, closestPtLat, closestPtLon) == true )
			{
				pLastTiffUsed = tiffInfo;
				return true;
			}
		}
	}

	pLastTiffUsed = nullptr;
	return false;
}

bool GeoTIFFReader::pGetClosestValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
									 double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != nullptr && closestPtLon != nullptr)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelValue(tiffInfo, x, y, value);
}

bool GeoTIFFReader::pGetClosestIntValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
										double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != nullptr && closestPtLon != nullptr)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelIntValue(tiffInfo, x, y, static_cast<int*>(value));
}

bool GeoTIFFReader::pGetClosestFltValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
										double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != nullptr && closestPtLon != nullptr)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelFltValue(tiffInfo, x, y, static_cast<float*>(value));
}

bool GeoTIFFReader::pGetInterplFltValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value, 
										[[maybe_unused]]double* closestPtLat, [[maybe_unused]]double* closestPtLon)
{
uint32_t x1, x2, y1, y2;
double xDbl, yDbl;
float val11, val12, val21, val22;
bool success;

	tiffInfo->GetSurroundingPixelIndexes(lat, lon, &x1, &x2, &y1, &y2, &xDbl, &yDbl);

	success = true;
	success &= pGetPixelFltValue(tiffInfo, x1, y1, &val11);
	success &= pGetPixelFltValue(tiffInfo, x1, y2, &val12);
	success &= pGetPixelFltValue(tiffInfo, x2, y1, &val21);
	success &= pGetPixelFltValue(tiffInfo, x2, y2, &val22);
	if( success )
	{
	double result;

		tiffInfo->BilinearInterpl(x1, x2, y1, y2, val11, val12, val21, val22, xDbl, yDbl, &result);
		*(static_cast<float*>(value)) = result;
		return true;
	}

	// try closest pixel value at last resort
	return pGetClosestFltValue(tiffInfo, lat, lon, value, nullptr, nullptr);
}

// Return false if value could not be read or if it is the "no data" value.
// Value must be able to contain tiffInfo->m_bitsPerSample/8 bytes.
bool GeoTIFFReader::pGetPixelValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, void* value)
{
	if(tiffInfo->m_tiffPtr == nullptr)
	{
		TIFFSetWarningHandler(nullptr);

		tiffInfo->m_tiffPtr = TIFFOpen(tiffInfo->m_pathname.c_str(), "r");

		// In case opening the file failed because too many files were already opened...
		if(tiffInfo->m_tiffPtr == nullptr)
		{
			CloseAllFiles(false);
			tiffInfo->m_tiffPtr = TIFFOpen(tiffInfo->m_pathname.c_str(), "r");
		}
	}

	if(tiffInfo->m_tiffPtr)
	{
		if( tiffInfo->m_rowsPerStrip > 0 )
		{ // strip-oriented tiff
		tmsize_t numBytesRead;
		uint32_t stripIndex = y / tiffInfo->m_rowsPerStrip;
		uint32_t rowIndexWithinStrip = y % tiffInfo->m_rowsPerStrip;
		uint16_t bytesPerSample = tiffInfo->m_bitsPerSample / 8;
		tmsize_t bytesOffsetWithinStrip;
		void* valueLocationWithinStrip;

			// try to get value from cache first
			bytesOffsetWithinStrip = (rowIndexWithinStrip*tiffInfo->m_rasterWidth + x)*bytesPerSample;
			if( tiffInfo->m_cache.GetValue(stripIndex, bytesOffsetWithinStrip, value) == true )
			{
				//pCacheHitCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}

			if(tiffInfo->m_readBuf == nullptr)
				tiffInfo->m_readBuf = _TIFFmalloc(tiffInfo->m_bytesPerStrip);
			numBytesRead = TIFFReadEncodedStrip(tiffInfo->m_tiffPtr, stripIndex, tiffInfo->m_readBuf, static_cast<tmsize_t>(tiffInfo->m_bytesPerStrip));
			if( bytesOffsetWithinStrip < numBytesRead)
			{
				valueLocationWithinStrip = (static_cast<uint8_t*>(tiffInfo->m_readBuf)) + bytesOffsetWithinStrip;
				memcpy(value, valueLocationWithinStrip, bytesPerSample);
				tiffInfo->m_cache.CacheStripData(stripIndex, tiffInfo->m_readBuf, numBytesRead);
				//pCacheMissCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}
		}
		else
		{ // tile-oriented tiff
		tmsize_t numBytesRead;
        ttile_t tileIndex;
		uint16_t bytesPerSample = tiffInfo->m_bitsPerSample / 8;
		uint32_t xWithinTile = x % tiffInfo->m_tileWidth;
		uint32_t yWithinTile = y % tiffInfo->m_tileHeight;
		tmsize_t byteOffsetWithinTile;
		void* valueLocationWithinTile;

			// try to get value from cache first
			tileIndex = TIFFComputeTile(tiffInfo->m_tiffPtr, x, y, 0, 0);
			byteOffsetWithinTile = (yWithinTile*tiffInfo->m_tileWidth + xWithinTile)*bytesPerSample;

			if( tiffInfo->m_cache.GetValue(tileIndex, byteOffsetWithinTile, value) == true )
			{
				//pCacheHitCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}

			if(tiffInfo->m_readBuf == nullptr)
				tiffInfo->m_readBuf = _TIFFmalloc(tiffInfo->m_bytesPerTile);
			numBytesRead = TIFFReadEncodedTile(tiffInfo->m_tiffPtr, tileIndex, tiffInfo->m_readBuf, static_cast<tmsize_t>(tiffInfo->m_bytesPerTile));
			if( byteOffsetWithinTile < numBytesRead)
			{
				valueLocationWithinTile = (static_cast<uint8_t*>(tiffInfo->m_readBuf)) + byteOffsetWithinTile;
				memcpy(value, valueLocationWithinTile, bytesPerSample);
				tiffInfo->m_cache.CacheTileData(tileIndex, tiffInfo->m_readBuf, numBytesRead);
				//pCacheMissCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}
		}
    }

	return false;
}

bool GeoTIFFReader::pGetPixelFltValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, float* value)
{
uint8_t buf[4];

	if( pGetPixelValue(tiffInfo, x, y, buf) == true )
	{
	int tiffDataType = (tiffInfo->m_bitsPerSample << 16) + tiffInfo->m_sampleFormat;

		switch(tiffDataType)
		{
			case TIFF_UINT8:
				*value = static_cast<float>(*reinterpret_cast<const uint8_t*>(buf));
				return true;
			case TIFF_INT8:
				*value = static_cast<float>(*reinterpret_cast<const int8_t*>(buf));
				return true;
			case TIFF_UINT16:
				*value = static_cast<float>(*reinterpret_cast<const uint16_t*>(buf));
				return true;
			case TIFF_INT16:
				*value = static_cast<float>(*reinterpret_cast<const int16_t*>(buf));
				return true;
			case TIFF_UINT32:
				*value = static_cast<float>(*reinterpret_cast<const uint32_t*>(buf));
				return true;
			case TIFF_INT32:
				*value = static_cast<float>(*reinterpret_cast<const int32_t*>(buf));
				return true;
			case TIFF_FLOAT32:
				*value = *reinterpret_cast<const float*>(buf);
				return true;
		}
	}

	return false;
}

bool GeoTIFFReader::pGetPixelIntValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, int* value)
{
uint8_t buf[4];

	if( pGetPixelValue(tiffInfo, x, y, buf) == true )
	{
	int tiffDataType = (tiffInfo->m_bitsPerSample << 16) + tiffInfo->m_sampleFormat;

		switch(tiffDataType)
		{
			case TIFF_UINT8:
				*value = static_cast<int>(*reinterpret_cast<const uint8_t*>(buf));
				return true;
			case TIFF_INT8:
				*value = static_cast<int>(*reinterpret_cast<const int8_t*>(buf));
				return true;
			case TIFF_UINT16:
				*value = static_cast<int>(*reinterpret_cast<const uint16_t*>(buf));
				return true;
			case TIFF_INT16:
				*value = static_cast<int>(*reinterpret_cast<const int16_t*>(buf));
				return true;
			case TIFF_UINT32:
				*value = static_cast<int>(*reinterpret_cast<const uint32_t*>(buf));
				return true;
			case TIFF_INT32:
				*value = static_cast<int>(*reinterpret_cast<const int32_t*>(buf));
				return true;
			case TIFF_FLOAT32:
				*value = static_cast<int>(*reinterpret_cast<const float*>(buf));
				return true;
		}
	}

	return false;
}

bool GeoTIFFReader::pIsNoDataValue(const GeoTIFFFileInfo* tiffInfo, const void* value)
{
	if( tiffInfo->m_noDataValuePresent == false )
		return false;

	int tiffDataType = ((static_cast<int>(tiffInfo->m_bitsPerSample)) << 16) + tiffInfo->m_sampleFormat;
	switch(tiffDataType)
	{
		case TIFF_UINT8:
			return tiffInfo->m_noDataValue == *(static_cast<const uint8_t*>(value));
		case TIFF_INT8:
			return tiffInfo->m_noDataValue == *(static_cast<const int8_t*>(value));
		case TIFF_UINT16:
			return tiffInfo->m_noDataValue == *(static_cast<const uint16_t*>(value));
		case TIFF_INT16:
			return tiffInfo->m_noDataValue == *(static_cast<const int16_t*>(value));
		case TIFF_UINT32:
			return tiffInfo->m_noDataValue == static_cast<int32_t>(*(static_cast<const uint32_t*>(value)));
		case TIFF_INT32:
			return tiffInfo->m_noDataValue == *(static_cast<const int32_t*>(value));
		case TIFF_FLOAT32:
			return tiffInfo->m_noDataValue == *(static_cast<const float*>(value));
	}

	return false;
}

void GeoTIFFReader::pSerializeTiffInfoFile(std::ostream& os, GeoTIFFFileInfo& tiffInfo)
{
	// NOTE: Increment GeoTIFFReader::GEOTIFF_INDEX_VERSION each time pSerializeTiffInfoFile()
	//       and pDeserializeTiffInfoFile() are updated.

    auto writeField = [&os](auto& field)
    {
        os.write(reinterpret_cast<char*>(&field), sizeof(field));
    };

	writeField(tiffInfo.m_coordSystem);
	writeField(tiffInfo.m_rasterHeight);
	writeField(tiffInfo.m_rasterWidth);
	writeField(tiffInfo.m_topLimit);
	writeField(tiffInfo.m_bottomLimit);
	writeField(tiffInfo.m_leftLimit);
	writeField(tiffInfo.m_rightLimit);
	writeField(tiffInfo.m_pixelHeight);
	writeField(tiffInfo.m_pixelWidth);
	writeField(tiffInfo.m_zone);
	writeField(tiffInfo.m_northp);
	std::string relPath = pGetRelativePath(pDir.c_str(), tiffInfo.m_pathname.c_str());
	pSerializeString(os, relPath);
	writeField(tiffInfo.m_applyDatumTransform);
	pSerializeDoubleVector(os, tiffInfo.m_toWgs84HelmertParams);

	writeField(tiffInfo.m_compression);
	writeField(tiffInfo.m_rowsPerStrip);
	writeField(tiffInfo.m_bitsPerSample);
	writeField(tiffInfo.m_samplesPerPixel);
	writeField(tiffInfo.m_sampleFormat);
	writeField(tiffInfo.m_tileHeight);
	writeField(tiffInfo.m_tileWidth);
	writeField(tiffInfo.m_bytesPerStrip);
	writeField(tiffInfo.m_bytesPerTile);
	writeField(tiffInfo.m_noDataValue);
	writeField(tiffInfo.m_noDataValuePresent);

	os.write(reinterpret_cast<char*>(&tiffInfo.m_ModelPixelScale), 3*sizeof(double));
	os.write(reinterpret_cast<char*>(&tiffInfo.m_ModelTiepoint), 6*sizeof(double));

	writeField(tiffInfo.m_GTModelTypeGeoKey);
	writeField(tiffInfo.m_GTRasterTypeGeoKey);
	writeField(tiffInfo.m_GeogAngularUnitsGeoKey);
	writeField(tiffInfo.m_ProjectedCSTypeGeoKey);
	writeField(tiffInfo.m_ProjLinearUnitsGeoKey);
	writeField(tiffInfo.m_GeographicTypeGeoKey);
	pSerializeDoubleVector(os, tiffInfo.m_GeogTOWGS84GeoKey);
	pSerializeString(os, tiffInfo.m_GTCitationGeoKey);
	pSerializeString(os, tiffInfo.m_GeogCitationGeoKey);
	writeField(tiffInfo.m_GeogSemiMajorAxisGeoKey);
	writeField(tiffInfo.m_GeogInvFlatteningGeoKey);
}

void GeoTIFFReader::pDeserializeTiffInfoFile(std::istream& is, GeoTIFFFileInfo& tiffInfo)
{
	// NOTE: Increment GeoTIFFReader::GEOTIFF_INDEX_VERSION each time pSerializeTiffInfoFile()
	//       and pDeserializeTiffInfoFile() are updated.

    auto readField = [&is](auto& field)
    {
        is.read(reinterpret_cast<char*>(&field), sizeof(field));
    };

	readField(tiffInfo.m_coordSystem);
	readField(tiffInfo.m_rasterHeight);
	readField(tiffInfo.m_rasterWidth);
	readField(tiffInfo.m_topLimit);
	readField(tiffInfo.m_bottomLimit);
	readField(tiffInfo.m_leftLimit);
	readField(tiffInfo.m_rightLimit);
	readField(tiffInfo.m_pixelHeight);
	readField(tiffInfo.m_pixelWidth);
	readField(tiffInfo.m_zone);
	readField(tiffInfo.m_northp);
	pDeserializeString(is, tiffInfo.m_pathname);
	tiffInfo.m_pathname = pDir + "/" + tiffInfo.m_pathname;
	readField(tiffInfo.m_applyDatumTransform);
	pDeserializeDoubleVector(is, tiffInfo.m_toWgs84HelmertParams);

	readField(tiffInfo.m_compression);
	readField(tiffInfo.m_rowsPerStrip);
	readField(tiffInfo.m_bitsPerSample);
	readField(tiffInfo.m_samplesPerPixel);
	readField(tiffInfo.m_sampleFormat);
	readField(tiffInfo.m_tileHeight);
	readField(tiffInfo.m_tileWidth);
	readField(tiffInfo.m_bytesPerStrip);
	readField(tiffInfo.m_bytesPerTile);
	readField(tiffInfo.m_noDataValue);
	readField(tiffInfo.m_noDataValuePresent);

	is.read(reinterpret_cast<char*>(&tiffInfo.m_ModelPixelScale), 3*sizeof(double));
	is.read(reinterpret_cast<char*>(&tiffInfo.m_ModelTiepoint), 6*sizeof(double));

	readField(tiffInfo.m_GTModelTypeGeoKey);
	readField(tiffInfo.m_GTRasterTypeGeoKey);
	readField(tiffInfo.m_GeogAngularUnitsGeoKey);
	readField(tiffInfo.m_ProjectedCSTypeGeoKey);
	readField(tiffInfo.m_ProjLinearUnitsGeoKey);
	readField(tiffInfo.m_GeographicTypeGeoKey);
	pDeserializeDoubleVector(is, tiffInfo.m_GeogTOWGS84GeoKey);
	pDeserializeString(is, tiffInfo.m_GTCitationGeoKey);
	pDeserializeString(is, tiffInfo.m_GeogCitationGeoKey);
	readField(tiffInfo.m_GeogSemiMajorAxisGeoKey);
	readField(tiffInfo.m_GeogInvFlatteningGeoKey);

	tiffInfo.Close();
}

void GeoTIFFReader::pSerializeString(std::ostream& os, const std::string& str)
{
	size_t numChars = str.size();
	os.write(reinterpret_cast<char*>(&numChars), sizeof(numChars));
	if(numChars > 0)
		os.write(str.data(), static_cast<std::streamsize>(numChars*sizeof(char)));
}

void GeoTIFFReader::pDeserializeString(std::istream& is, std::string& str)
{
	size_t numChars = 0;
	is.read(reinterpret_cast<char*>(&numChars), sizeof(numChars));
	str.resize(numChars);
	if(numChars > 0)
		is.read(str.data(), static_cast<std::streamsize>(numChars*sizeof(char)));
}

void GeoTIFFReader::pSerializeDoubleVector(std::ostream& os, const std::vector<double>& v)
{
	size_t numItems = v.size();
	os.write(reinterpret_cast<const char*>(&numItems), sizeof(numItems));
	if(numItems > 0)
		os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(numItems*sizeof(double)));
}

void GeoTIFFReader::pDeserializeDoubleVector(std::istream& is, std::vector<double>& v)
{
	size_t numItems = 0;
	is.read(reinterpret_cast<char*>(&numItems), sizeof(numItems));
	v.resize(numItems);
	if(numItems > 0)
		is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(numItems*sizeof(double)));
}
