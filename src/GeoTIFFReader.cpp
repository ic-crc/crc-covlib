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
	m_tiffPtr = NULL;
	m_readBuf = NULL;
	Clear();
}

GeoTIFFReader::GeoTIFFFileInfo::~GeoTIFFFileInfo()
{
	Close();
}

GeoTIFFReader::GeoTIFFFileInfo::GeoTIFFFileInfo(const GeoTIFFFileInfo& original)
{
	m_tiffPtr = NULL;
	m_readBuf = NULL;
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
	if (m_readBuf != NULL)
	{
		_TIFFfree(m_readBuf);
		m_readBuf = NULL;
	}
	if (m_tiffPtr != NULL)
	{
		TIFFClose(m_tiffPtr);
		m_tiffPtr = NULL;
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
	pLastTiffUsed = NULL;

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
	pLastTiffUsed = NULL;

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
	pLastTiffUsed = NULL;
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

		outfile.write((char*) &indexVersion, sizeof(indexVersion));
		outfile.write((char*) &num, (std::streamsize) sizeof(num));
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
		infile.read((char*) &geotiffIndexVersion, sizeof(geotiffIndexVersion));

		if( geotiffIndexVersion == GEOTIFF_INDEX_VERSION )
		{
			pGeoTiffs.clear();
			pLastTiffUsed = NULL;
			size_t num;
			infile.read((char*) &num, sizeof(num));
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
TIFF* tif = NULL;
bool readOK = true;

	tiffInfo.Clear();
	tiffInfo.m_pathname = pathname;

	TIFFSetWarningHandler(NULL);
	tif = TIFFOpen(pathname, "r");
	if( tif != NULL )
	{
	uint32_t count = 0;
	void* data = NULL;

		readOK &= (TIFFGetField(tif, TIFFTAG_COMPRESSION, &(tiffInfo.m_compression)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &(tiffInfo.m_rasterWidth)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &(tiffInfo.m_rasterHeight)) == 1);
		TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &(tiffInfo.m_rowsPerStrip)); // will not be present in tile-oriented tiffs
		readOK &= (TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &(tiffInfo.m_sampleFormat)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &(tiffInfo.m_samplesPerPixel)) == 1);
		readOK &= (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &(tiffInfo.m_bitsPerSample)) == 1);
		if(TIFFGetField(tif, 42113, &count, &data) == 1)
		{
			tiffInfo.m_noDataValue = atoi((char*)data);
			tiffInfo.m_noDataValuePresent = true;
		}
		TIFFGetField(tif, TIFFTAG_TILELENGTH, &(tiffInfo.m_tileHeight)); // will not be present in strip-oriented tiffs
		TIFFGetField(tif, TIFFTAG_TILEWIDTH, &(tiffInfo.m_tileWidth)); // will not be present in strip-oriented tiffs

		tiffInfo.m_bytesPerStrip = TIFFStripSize(tif);
		tiffInfo.m_bytesPerTile = TIFFTileSize(tif); 

		readOK &= pGetGeoTagDoubleArrayValue(tif, 33550, tiffInfo.m_ModelPixelScale, 3);
		readOK &= pGetGeoTagDoubleArrayValue(tif, 33922, tiffInfo.m_ModelTiepoint[0], 6);

		readOK &= (TIFFGetField(tif, 34735, &count, &data) == 1); // 34735 = GeoKeyDirectoryTag
		if( count % 4 == 0 )
		{
		uint16_t keyID = 0, tiffTagLocation = 0, numValues = 0, valueOffset = 0;

			for(uint32_t i=0 ; i<count ; i+=4)
			{
				keyID = ((uint16_t*)data)[i];
				tiffTagLocation = ((uint16_t*)data)[i+1];
				numValues = ((uint16_t*)data)[i+2];
				valueOffset = ((uint16_t*)data)[i+3];

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
uint32_t count = 0;
void* data = NULL;

	if(TIFFGetField(tif, tag, &count, &data) == 1 && count == dstSize)
	{
		memcpy(dst, data, dstSize*sizeof(double));
		return true;
	}
	return false;
}

bool GeoTIFFReader::pGetGeoKeyStringValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::string& dst)
{
uint32_t count = 0;
void* data = NULL;

	if(TIFFGetField(tif, tiffTagLocation, &count, &data) == 1)
	{
		if( valueOffset+numValues <= count && numValues > 0 )
		{
			dst = ((char*)data) + valueOffset;
			dst.resize((size_t)numValues-1);
			return true;
		}
	}
	return false;
}

bool GeoTIFFReader::pGetGeoKeyDoubleArrayValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::vector<double>& dst)
{
uint32_t count = 0;
void* data = NULL;

	if(TIFFGetField(tif, tiffTagLocation, &count, &data) == 1)
	{
		if( valueOffset+numValues <= count )
		{
			dst.resize(numValues);
			memcpy(&(dst[0]), ((double*)data) + valueOffset, numValues*sizeof(double));
			return true;
		}
	}
	return false;
}

bool GeoTIFFReader::pGetGeoKeyDoubleValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, double& dst)
{
uint32_t count = 0;
void* data = NULL;

	if(TIFFGetField(tif, tiffTagLocation, &count, &data) == 1 && numValues == 1)
	{
		dst = *(((double*)data) + valueOffset);
		return true;
	}
	return false;
}

void GeoTIFFReader::pUpdateFilesInfo(const char* directory)
{
std::vector<std::string> tifPathnames = pGetPathnameList(directory, ".tif");
GeoTIFFFileInfo tiffInfo;

	pGeoTiffs.clear();
	pLastTiffUsed = NULL;
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
uint8_t bytesPerSample;
uint32_t stripOrTileSizeInBytes;

	for (size_t i=0; i < pGeoTiffs.size(); i++)
	{
		tiffInfo = &(pGeoTiffs[i]);

		bytesPerSample = tiffInfo-> m_bitsPerSample/8;

		if( tiffInfo->m_rowsPerStrip > 0 )
			stripOrTileSizeInBytes = (uint32_t) tiffInfo->m_bytesPerStrip;
		else
			stripOrTileSizeInBytes = (uint32_t) tiffInfo->m_bytesPerTile;
		
		tiffInfo->m_cache.SetCacheEntrySize(bytesPerSample, stripOrTileSizeInBytes);
		tiffInfo->m_cache.SetTotalSizeLimit(UINT32_MAX);
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
	const char sep[3] = { '/', '\\', ':' };
#else
	const char sep[1] = { '/' };
#endif
std::string baseDirCopy = baseDir;
std::string pathnameCopy = pathname;
char* token;
std::vector<char*> tokens;
size_t searchFrom = 0;
size_t findResult;
std::string result = pathname;

#ifdef _WIN32 // pathnames are case sensitive in Linux, but not in Windows
	pToLowercase(baseDirCopy);
	pToLowercase(pathnameCopy);
#endif

	token = strtok(const_cast<char*>(baseDirCopy.c_str()), sep);
	while (token != NULL)
	{
		tokens.push_back(token);
		token = strtok(NULL, sep);
	}

	for (size_t i = 0; i < tokens.size(); i++)
	{
		findResult = pathnameCopy.find(tokens[i], searchFrom);
		if (findResult != std::string::npos)
			searchFrom = findResult + strlen(tokens[i]) + 1;
	}

	result = result.substr(searchFrom);
	
#ifdef _WIN32
	std::replace(result.begin(), result.end(), '\\', '/');
#endif
	return result;
}

bool GeoTIFFReader::pCompareTiffInfoOnResolution(GeoTIFFFileInfo* tiffInfo1, GeoTIFFFileInfo* tiffInfo2)
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

bool GeoTIFFReader::GetClosestValue(double lat, double lon, void* value, double* closestPtLat/*=NULL*/, double* closestPtLon/*=NULL*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}

bool GeoTIFFReader::GetClosestIntValue(double lat, double lon, int* value, double* closestPtLat/*=NULL*/, double* closestPtLon/*=NULL*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestIntValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}

bool GeoTIFFReader::GetClosestFltValue(double lat, double lon, float* value, double* closestPtLat/*=NULL*/, double* closestPtLon/*=NULL*/)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetClosestFltValue;

	return pGetValue(lat, lon, value, closestPtLat, closestPtLon, f);
}
	
bool GeoTIFFReader::GetInterplValue(double lat, double lon, float* value)
{
GetValueMemberFunc f = &GeoTIFFReader::pGetInterplFltValue;

	return pGetValue(lat, lon, value, NULL, NULL, f);
}

bool GeoTIFFReader::pGetValue(double lat, double lon, void* value,
							  double* closestPtLat, double* closestPtLon,
							  GetValueMemberFunc getValueFunc)
{
	if( pLastTiffUsed != NULL && pLastTiffUsed->IsIn(lat, lon) == true )
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

	pLastTiffUsed = NULL;
	return false;
}

bool GeoTIFFReader::pGetClosestValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
									 double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != NULL && closestPtLon != NULL)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelValue(tiffInfo, x, y, value);
}

bool GeoTIFFReader::pGetClosestIntValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
										double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != NULL && closestPtLon != NULL)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelIntValue(tiffInfo, x, y, (int*)value);
}

bool GeoTIFFReader::pGetClosestFltValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value,
										double* closestPtLat, double* closestPtLon)
{
uint32_t x, y;

	tiffInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != NULL && closestPtLon != NULL)
		tiffInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pGetPixelFltValue(tiffInfo, x, y, (float*)value);
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
		*((float*)value) = result;
		return true;
	}

	// try closest pixel value at last resort
	return pGetClosestFltValue(tiffInfo, lat, lon, value, NULL, NULL);
}

// Return false if value could not be read or if it is the "no data" value.
// Value must be able to contain tiffInfo->m_bitsPerSample/8 bytes.
bool GeoTIFFReader::pGetPixelValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, void* value)
{
	if(tiffInfo->m_tiffPtr == NULL)
	{
		TIFFSetWarningHandler(NULL);

		tiffInfo->m_tiffPtr = TIFFOpen(tiffInfo->m_pathname.c_str(), "r");

		// In case opening the file failed because too many files were already opened...
		if(tiffInfo->m_tiffPtr == NULL)
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
		uint32_t bytesOffsetWithinStrip;
		void* valueLocationWithinStrip;

			// try to get value from cache first
			bytesOffsetWithinStrip = (rowIndexWithinStrip*tiffInfo->m_rasterWidth + x)*bytesPerSample;
			if( tiffInfo->m_cache.GetValue(stripIndex, bytesOffsetWithinStrip, value) == true )
			{
				//pCacheHitCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}

			if(tiffInfo->m_readBuf == NULL)
				tiffInfo->m_readBuf = _TIFFmalloc(tiffInfo->m_bytesPerStrip);
			numBytesRead = TIFFReadEncodedStrip(tiffInfo->m_tiffPtr, stripIndex, tiffInfo->m_readBuf, (tmsize_t) tiffInfo->m_bytesPerStrip);
			if( (tmsize_t)bytesOffsetWithinStrip < numBytesRead)
			{
				valueLocationWithinStrip = ((uint8_t*)tiffInfo->m_readBuf) + bytesOffsetWithinStrip;
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
		uint32_t byteOffsetWithinTile;
		void* valueLocationWithinTile;

			// try to get value from cache first
			tileIndex = TIFFComputeTile(tiffInfo->m_tiffPtr, x, y, 0, 0);
			byteOffsetWithinTile = (yWithinTile*tiffInfo->m_tileWidth + xWithinTile)*bytesPerSample;

			if( tiffInfo->m_cache.GetValue(tileIndex, byteOffsetWithinTile, value) == true )
			{
				//pCacheHitCount++;
				return !pIsNoDataValue(tiffInfo, value);
			}

			if(tiffInfo->m_readBuf == NULL)
				tiffInfo->m_readBuf = _TIFFmalloc(tiffInfo->m_bytesPerTile);
			numBytesRead = TIFFReadEncodedTile(tiffInfo->m_tiffPtr, tileIndex, tiffInfo->m_readBuf, (tmsize_t) tiffInfo->m_bytesPerTile);
			if( (tmsize_t)byteOffsetWithinTile < numBytesRead)
			{
				valueLocationWithinTile = ((uint8_t*)tiffInfo->m_readBuf) + byteOffsetWithinTile;
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
				*value = (float) *((uint8_t*)buf);
				return true;
			case TIFF_INT8:
				*value = (float) *((int8_t*)buf);
				return true;
			case TIFF_UINT16:
				*value = (float) *((uint16_t*)buf);
				return true;
			case TIFF_INT16:
				*value = (float) *((int16_t*)buf);
				return true;
			case TIFF_UINT32:
				*value = (float) *((uint32_t*)buf);
				return true;
			case TIFF_INT32:
				*value = (float) *((int32_t*)buf);
				return true;
			case TIFF_FLOAT32:
				*value = *((float*)buf);
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
				*value = (int) *((uint8_t*)buf);
				return true;
			case TIFF_INT8:
				*value = (int) *((int8_t*)buf);
				return true;
			case TIFF_UINT16:
				*value = (int) *((uint16_t*)buf);
				return true;
			case TIFF_INT16:
				*value = (int) *((int16_t*)buf);
				return true;
			case TIFF_UINT32:
				*value = (int) *((uint32_t*)buf);
				return true;
			case TIFF_INT32:
				*value = (int) *((int32_t*)buf);
				return true;
			case TIFF_FLOAT32:
				*value = (int) *((float*)buf);
				return true;
		}
	}

	return false;
}

bool GeoTIFFReader::pIsNoDataValue(GeoTIFFFileInfo* tiffInfo, void* value)
{
	if( tiffInfo->m_noDataValuePresent == false )
		return false;

	int tiffDataType = (((int)(tiffInfo->m_bitsPerSample)) << 16) + tiffInfo->m_sampleFormat;
	switch(tiffDataType)
	{
		case TIFF_UINT8:
			return (tiffInfo->m_noDataValue == *((uint8_t*)value));
		case TIFF_INT8:
			return (tiffInfo->m_noDataValue == *((int8_t*)value));
		case TIFF_UINT16:
			return (tiffInfo->m_noDataValue == *((uint16_t*)value));
		case TIFF_INT16:
			return (tiffInfo->m_noDataValue == *((int16_t*)value));
		case TIFF_UINT32:
			return (tiffInfo->m_noDataValue == (int32_t) *((uint32_t*)value));
		case TIFF_INT32:
			return (tiffInfo->m_noDataValue == *((int32_t*)value));
		case TIFF_FLOAT32:
			return (tiffInfo->m_noDataValue == *((float*)value));
	}

	return false;
}

void GeoTIFFReader::pSerializeTiffInfoFile(std::ostream& os, GeoTIFFFileInfo& tiffInfo)
{
	// NOTE: Increment GeoTIFFReader::GEOTIFF_INDEX_VERSION each time pSerializeTiffInfoFile()
	//       and pDeserializeTiffInfoFile() are updated.

	os.write((char*) &tiffInfo.m_coordSystem, sizeof(tiffInfo.m_coordSystem));
	os.write((char*) &tiffInfo.m_rasterHeight, sizeof(tiffInfo.m_rasterHeight));
	os.write((char*) &tiffInfo.m_rasterWidth, sizeof(tiffInfo.m_rasterWidth));
	os.write((char*) &tiffInfo.m_topLimit, sizeof(tiffInfo.m_topLimit));
	os.write((char*) &tiffInfo.m_bottomLimit, sizeof(tiffInfo.m_bottomLimit));
	os.write((char*) &tiffInfo.m_leftLimit, sizeof(tiffInfo.m_leftLimit));
	os.write((char*) &tiffInfo.m_rightLimit, sizeof(tiffInfo.m_rightLimit));
	os.write((char*) &tiffInfo.m_pixelHeight, sizeof(tiffInfo.m_pixelHeight));
	os.write((char*) &tiffInfo.m_pixelWidth, sizeof(tiffInfo.m_pixelWidth));
	os.write((char*) &tiffInfo.m_zone, sizeof(tiffInfo.m_zone));
	os.write((char*) &tiffInfo.m_northp, sizeof(tiffInfo.m_northp));
	std::string relPath = pGetRelativePath(pDir.c_str(), tiffInfo.m_pathname.c_str());
	pSerializeString(os, relPath);
	os.write((char*) &tiffInfo.m_applyDatumTransform, sizeof(tiffInfo.m_applyDatumTransform));
	pSerializeDoubleVector(os, tiffInfo.m_toWgs84HelmertParams);

	os.write((char*) &tiffInfo.m_compression, sizeof(tiffInfo.m_compression));
	os.write((char*) &tiffInfo.m_rowsPerStrip, sizeof(tiffInfo.m_rowsPerStrip));
	os.write((char*) &tiffInfo.m_bitsPerSample, sizeof(tiffInfo.m_bitsPerSample));
	os.write((char*) &tiffInfo.m_samplesPerPixel, sizeof(tiffInfo.m_samplesPerPixel));
	os.write((char*) &tiffInfo.m_sampleFormat, sizeof(tiffInfo.m_sampleFormat));
	os.write((char*) &tiffInfo.m_tileHeight, sizeof(tiffInfo.m_tileHeight));
	os.write((char*) &tiffInfo.m_tileWidth, sizeof(tiffInfo.m_tileWidth));
	os.write((char*) &tiffInfo.m_bytesPerStrip, sizeof(tiffInfo.m_bytesPerStrip));
	os.write((char*) &tiffInfo.m_bytesPerTile, sizeof(tiffInfo.m_bytesPerTile));
	os.write((char*) &tiffInfo.m_noDataValue, sizeof(tiffInfo.m_noDataValue));
	os.write((char*) &tiffInfo.m_noDataValuePresent, sizeof(tiffInfo.m_noDataValuePresent));

	os.write((char*) &tiffInfo.m_ModelPixelScale, 3*sizeof(double));
	os.write((char*) &tiffInfo.m_ModelTiepoint, 6*sizeof(double));

	os.write((char*) &tiffInfo.m_GTModelTypeGeoKey, sizeof(tiffInfo.m_GTModelTypeGeoKey));
	os.write((char*) &tiffInfo.m_GTRasterTypeGeoKey, sizeof(tiffInfo.m_GTRasterTypeGeoKey));
	os.write((char*) &tiffInfo.m_GeogAngularUnitsGeoKey, sizeof(tiffInfo.m_GeogAngularUnitsGeoKey));
	os.write((char*) &tiffInfo.m_ProjectedCSTypeGeoKey, sizeof(tiffInfo.m_ProjectedCSTypeGeoKey));
	os.write((char*) &tiffInfo.m_ProjLinearUnitsGeoKey, sizeof(tiffInfo.m_ProjLinearUnitsGeoKey));
	os.write((char*) &tiffInfo.m_GeographicTypeGeoKey, sizeof(tiffInfo.m_GeographicTypeGeoKey));
	pSerializeDoubleVector(os, tiffInfo.m_GeogTOWGS84GeoKey);
	pSerializeString(os, tiffInfo.m_GTCitationGeoKey);
	pSerializeString(os, tiffInfo.m_GeogCitationGeoKey);
	os.write((char*) &tiffInfo.m_GeogSemiMajorAxisGeoKey, sizeof(tiffInfo.m_GeogSemiMajorAxisGeoKey));
	os.write((char*) &tiffInfo.m_GeogInvFlatteningGeoKey, sizeof(tiffInfo.m_GeogInvFlatteningGeoKey));
}

void GeoTIFFReader::pDeserializeTiffInfoFile(std::istream& is, GeoTIFFFileInfo& tiffInfo)
{
	// NOTE: Increment GeoTIFFReader::GEOTIFF_INDEX_VERSION each time pSerializeTiffInfoFile()
	//       and pDeserializeTiffInfoFile() are updated.

	is.read((char*) &tiffInfo.m_coordSystem, sizeof(tiffInfo.m_coordSystem));
	is.read((char*) &tiffInfo.m_rasterHeight, sizeof(tiffInfo.m_rasterHeight));
	is.read((char*) &tiffInfo.m_rasterWidth, sizeof(tiffInfo.m_rasterWidth));
	is.read((char*) &tiffInfo.m_topLimit, sizeof(tiffInfo.m_topLimit));
	is.read((char*) &tiffInfo.m_bottomLimit, sizeof(tiffInfo.m_bottomLimit));
	is.read((char*) &tiffInfo.m_leftLimit, sizeof(tiffInfo.m_leftLimit));
	is.read((char*) &tiffInfo.m_rightLimit, sizeof(tiffInfo.m_rightLimit));
	is.read((char*) &tiffInfo.m_pixelHeight, sizeof(tiffInfo.m_pixelHeight));
	is.read((char*) &tiffInfo.m_pixelWidth, sizeof(tiffInfo.m_pixelWidth));
	is.read((char*) &tiffInfo.m_zone, sizeof(tiffInfo.m_zone));
	is.read((char*) &tiffInfo.m_northp, sizeof(tiffInfo.m_northp));
	pDeserializeString(is, tiffInfo.m_pathname);
	tiffInfo.m_pathname = pDir + "/" + tiffInfo.m_pathname;
	is.read((char*) &tiffInfo.m_applyDatumTransform, sizeof(tiffInfo.m_applyDatumTransform));
	pDeserializeDoubleVector(is, tiffInfo.m_toWgs84HelmertParams);

	is.read((char*) &tiffInfo.m_compression, sizeof(tiffInfo.m_compression));
	is.read((char*) &tiffInfo.m_rowsPerStrip, sizeof(tiffInfo.m_rowsPerStrip));
	is.read((char*) &tiffInfo.m_bitsPerSample, sizeof(tiffInfo.m_bitsPerSample));
	is.read((char*) &tiffInfo.m_samplesPerPixel, sizeof(tiffInfo.m_samplesPerPixel));
	is.read((char*) &tiffInfo.m_sampleFormat, sizeof(tiffInfo.m_sampleFormat));
	is.read((char*) &tiffInfo.m_tileHeight, sizeof(tiffInfo.m_tileHeight));
	is.read((char*) &tiffInfo.m_tileWidth, sizeof(tiffInfo.m_tileWidth));
	is.read((char*) &tiffInfo.m_bytesPerStrip, sizeof(tiffInfo.m_bytesPerStrip));
	is.read((char*) &tiffInfo.m_bytesPerTile, sizeof(tiffInfo.m_bytesPerTile));
	is.read((char*) &tiffInfo.m_noDataValue, sizeof(tiffInfo.m_noDataValue));
	is.read((char*) &tiffInfo.m_noDataValuePresent, sizeof(tiffInfo.m_noDataValuePresent));

	is.read((char*) &tiffInfo.m_ModelPixelScale, 3*sizeof(double));
	is.read((char*) &tiffInfo.m_ModelTiepoint, 6*sizeof(double));

	is.read((char*) &tiffInfo.m_GTModelTypeGeoKey, sizeof(tiffInfo.m_GTModelTypeGeoKey));
	is.read((char*) &tiffInfo.m_GTRasterTypeGeoKey, sizeof(tiffInfo.m_GTRasterTypeGeoKey));
	is.read((char*) &tiffInfo.m_GeogAngularUnitsGeoKey, sizeof(tiffInfo.m_GeogAngularUnitsGeoKey));
	is.read((char*) &tiffInfo.m_ProjectedCSTypeGeoKey, sizeof(tiffInfo.m_ProjectedCSTypeGeoKey));
	is.read((char*) &tiffInfo.m_ProjLinearUnitsGeoKey, sizeof(tiffInfo.m_ProjLinearUnitsGeoKey));
	is.read((char*) &tiffInfo.m_GeographicTypeGeoKey, sizeof(tiffInfo.m_GeographicTypeGeoKey));
	pDeserializeDoubleVector(is, tiffInfo.m_GeogTOWGS84GeoKey);
	pDeserializeString(is, tiffInfo.m_GTCitationGeoKey);
	pDeserializeString(is, tiffInfo.m_GeogCitationGeoKey);
	is.read((char*) &tiffInfo.m_GeogSemiMajorAxisGeoKey, sizeof(tiffInfo.m_GeogSemiMajorAxisGeoKey));
	is.read((char*) &tiffInfo.m_GeogInvFlatteningGeoKey, sizeof(tiffInfo.m_GeogInvFlatteningGeoKey));

	tiffInfo.Close();
}

void GeoTIFFReader::pSerializeString(std::ostream& os, std::string& str)
{
	size_t numChars = str.size();
	os.write((char*) &numChars, sizeof(numChars));
	if(numChars > 0)
		os.write(&(str[0]), (std::streamsize)(numChars*sizeof(char)));
}

void GeoTIFFReader::pDeserializeString(std::istream& is, std::string& str)
{
	size_t numChars = 0;
	is.read((char*) &numChars, sizeof(numChars));
	str.resize(numChars);
	if(numChars > 0)
		is.read(&(str[0]), (std::streamsize)(numChars*sizeof(char)));
}

void GeoTIFFReader::pSerializeDoubleVector(std::ostream& os, std::vector<double>& v)
{
	size_t numItems = v.size();
	os.write((char*) &numItems, sizeof(numItems));
	if(numItems > 0)
		os.write((char*) &(v[0]), (std::streamsize)(numItems*sizeof(double)));
}

void GeoTIFFReader::pDeserializeDoubleVector(std::istream& is, std::vector<double>& v)
{
	size_t numItems = 0;
	is.read((char*) &numItems, sizeof(numItems));
	v.resize(numItems);
	if(numItems > 0)
		is.read((char*) &(v[0]), (std::streamsize)(numItems*sizeof(double)));
}
