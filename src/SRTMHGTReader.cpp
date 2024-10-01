#include "SRTMHGTReader.h"
#ifdef _MSC_VER
	#define _CRT_SECURE_NO_DEPRECATE
#endif
#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif


// SRTMHGTFileInfo Class //////////////////////////////////////////////////////////////////////////

SRTMHGTReader::SRTMHGTFileInfo::SRTMHGTFileInfo(std::string& pathname, uint32_t rasterHeight, uint32_t rasterWidth,
                                                double minLat, double maxLat, double minLon, double maxLon,
                                                double pixelHeightDeg, double pixelWidthDeg, int16_t noDataValue)
{
	m_coordSystem = GEOGRAPHIC;
	m_rasterHeight = rasterHeight;
	m_rasterWidth = rasterWidth;
	m_topLimit = maxLat; 
	m_bottomLimit = minLat;
	m_leftLimit = minLon;
	m_rightLimit = maxLon;
	m_pixelHeight = pixelHeightDeg;
	m_pixelWidth = pixelWidthDeg;
	m_pathname = pathname;
	m_noDataValue = noDataValue;
	m_filePtr = nullptr;
}

SRTMHGTReader::SRTMHGTFileInfo::SRTMHGTFileInfo(const SRTMHGTFileInfo& original)
{
	m_filePtr = nullptr;
	*this = original;
}

SRTMHGTReader::SRTMHGTFileInfo::~SRTMHGTFileInfo()
{
	Close(true);
}

const SRTMHGTReader::SRTMHGTFileInfo& SRTMHGTReader::SRTMHGTFileInfo::operator=(const SRTMHGTFileInfo& original)
{
	if (&original == this)
		return *this;

	GeoRasterFileInfo::operator=(original);

	m_noDataValue = original.m_noDataValue;
	Close(true); // do not copy file pointer (m_filePtr) and cached data

	return *this;
}

void SRTMHGTReader::SRTMHGTFileInfo::Close(bool clearCache)
{
	if (m_filePtr != nullptr)
	{
		fclose(m_filePtr);
		m_filePtr = nullptr;
	}
	if( clearCache )
		m_cache.clear();
}

void SRTMHGTReader::SRTMHGTFileInfo::Cache(uint32_t x, uint32_t y, int16_t value)
{
int xIndexCacheElem, yIndexCacheElem;
uint32_t key;
std::map<uint32_t, CacheElement>::iterator it;

	// Use the 5 least significant bits for indexing within a cache element,
	// other bits will be used for the map's key.
	xIndexCacheElem = x & 0x1F; // take 5 bits (2^5=32, the x size of a cache element)
	yIndexCacheElem = y & 0x1F; // take 5 bits (2^5=32, the y size of a cache element)
	x = x >> 5;
	y = y >> 5;
	key = (x << 16) + y;
	it = m_cache.find(key);
	if (it != m_cache.end())
	{
		it->second.data[xIndexCacheElem][yIndexCacheElem] = value;
	}
	else
	{
	CacheElement elem;
	int16_t uninitializedValue = m_noDataValue + 1;

		for(int i=0 ; i<32 ; i++)
			for(int j=0 ; j<32 ; j++)
				elem.data[i][j] = uninitializedValue;

		elem.data[xIndexCacheElem][yIndexCacheElem] = value;
		m_cache[key] = elem;
	}

}

bool SRTMHGTReader::SRTMHGTFileInfo::GetFromCache(uint32_t x, uint32_t y, int16_t* value)
{
int xIndexCacheElem, yIndexCacheElem;
uint32_t key;
std::map<uint32_t, CacheElement>::iterator it;

	xIndexCacheElem = x & 0x1F;
	yIndexCacheElem = y & 0x1F;
	x = x >> 5;
	y = y >> 5;
	key = (x << 16) + y;
	it = m_cache.find(key);
	if (it != m_cache.end())
	{
		*value = it->second.data[xIndexCacheElem][yIndexCacheElem];
		if( *value != m_noDataValue + 1 ) // if not the "uninitialized" value
			return true;
	}
	return false;
}



// SRTMHGTReader Class ///////////////////////////////////////////////////////////////////

SRTMHGTReader::SRTMHGTReader()
{
	pLastUsedFileInfo = nullptr;
}

SRTMHGTReader::SRTMHGTReader(const SRTMHGTReader& original)
{
	*this = original;
}

SRTMHGTReader::~SRTMHGTReader()
{

}

const SRTMHGTReader& SRTMHGTReader::operator=(const SRTMHGTReader& original)
{
	if (&original == this)
		return *this;

	pDirectory = original.pDirectory;
	pSRTMHGTFiles = original.pSRTMHGTFiles;
	pUpdateRTree(); // needs to rebuild the R-Tree (and not copy it) since it stores pointers from pSRTMHGTFiles
	pLastUsedFileInfo = nullptr;

	return *this;
}

void SRTMHGTReader::SetDirectory(const char* directory)
{
	pDirectory = directory;
	pUpdateFilesInfo(directory);
}
	
const char* SRTMHGTReader::GetDirectory() const
{
	return pDirectory.c_str();
}

bool SRTMHGTReader::GetValue(double lat, double lon, bool interpolate, float* value)
{
	if( pLastUsedFileInfo != nullptr && pLastUsedFileInfo->IsIn(lat, lon) == true )
		if( pGetValue(pLastUsedFileInfo, lat, lon, interpolate, value) == true )
			return true;

	std::vector<SRTMHGTFileInfo*> fileInfoList = pGetFileInfoList(lat, lon);
	SRTMHGTFileInfo* fileInfo;
	for(size_t i=0 ; i<fileInfoList.size() ; i++)
	{
		fileInfo = fileInfoList[i];
		if( fileInfo != pLastUsedFileInfo )
		{
			if( pGetValue(fileInfo, lat, lon, interpolate, value) == true )
			{
				pLastUsedFileInfo = fileInfo;
				return true;
			}
		}
	}

	pLastUsedFileInfo = nullptr;
	return false;
}
	
void SRTMHGTReader::CloseAllFiles(bool clearCaches)
{
	for(size_t i=0 ; i<pSRTMHGTFiles.size() ; i++)
		pSRTMHGTFiles[i].Close(clearCaches);
}

void SRTMHGTReader::pUpdateFilesInfo(const char* directory)
{
std::vector<SRTMHGTReader::Path> pathList = pGetPathList(directory);
double lat, lon;
std::uintmax_t sizeInBytes;
std::error_code sizeError;
double pixelSizeDeg;

	pLastUsedFileInfo = nullptr;
	pSRTMHGTFiles.clear();

	for(size_t i=0 ; i<pathList.size() ; i++)
	{
		if( pIsFilenameSupported(pathList[i].filename, &lat, &lon) == true )
		{
			sizeInBytes = fs::file_size(fs::path(pathList[i].pathname), sizeError);
			if( sizeInBytes == 25934402 ) // 3601 x 3601 x 2 bytes
			{
				// SRTM1 elevation files (only the .hgt files are required) possible sources at:
				// https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11
				// https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/
				// https://dwtkns.com/srtm30m/
				// https://step.esa.int/auxdata/dem/SRTMGL1/
				pixelSizeDeg = 1.0/3600.0;
				pSRTMHGTFiles.push_back(
					SRTMHGTFileInfo(pathList[i].pathname, 3601, 3601, lat-(pixelSizeDeg/2), (lat+1)+(pixelSizeDeg/2),
					                lon-(pixelSizeDeg/2), (lon+1)+(pixelSizeDeg/2), pixelSizeDeg, pixelSizeDeg, -32768)
				);
			}
			else if( sizeInBytes == 2884802 )
			{
				// SRTM3 elevation files (only the .hgt files are required) possible sources at:
				// https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11
				pixelSizeDeg = 3.0/3600.0;
				pSRTMHGTFiles.push_back(
					SRTMHGTFileInfo(pathList[i].pathname, 1201, 1201, lat-(pixelSizeDeg/2), (lat+1)+(pixelSizeDeg/2),
					                lon-(pixelSizeDeg/2), (lon+1)+(pixelSizeDeg/2), pixelSizeDeg, pixelSizeDeg, -32768)
				);
			}
			else if( sizeInBytes == 57600000 )
			{
				// SRTM30 elevation files (only the .dem files are required) possible sources at:
				// https://lpdaac.usgs.gov/products/srtmgl30v021/
				// https://web.archive.org/web/20170124235811/https://dds.cr.usgs.gov/srtm/version2_1/SRTM30/
				// http://www.webgis.com/terr_world.html
				pSRTMHGTFiles.push_back(
					SRTMHGTFileInfo(pathList[i].pathname, 6000, 4800, lat-50, lat, lon, lon+40, 30.0/3600.0, 30.0/3600.0, -9999)
				);
			}
			else if( sizeInBytes >= 8 )
			{
				// Not an official format, but allow here to use a custom resolution as long as
				// the raster's height and width are equal.
				unsigned int size = sqrt(sizeInBytes/2.0);
				if( size*size*2 == sizeInBytes )
				{
					pixelSizeDeg = 1.0/(size-1);
					pSRTMHGTFiles.push_back(
						SRTMHGTFileInfo(pathList[i].pathname, size, size, lat-(pixelSizeDeg/2), (lat+1)+(pixelSizeDeg/2),
										lon-(pixelSizeDeg/2), (lon+1)+(pixelSizeDeg/2), pixelSizeDeg, pixelSizeDeg, -32768)
					);
				}
			}
		}
	}

	pUpdateRTree();
}

void SRTMHGTReader::pUpdateRTree()
{
SRTMHGTFileInfo* fileInfo;
double minLat, minLon, maxLat, maxLon;
double minNativeCoords[2];
double maxNativeCoords[2];

	pRTree.RemoveAll();
	for (size_t i=0; i < pSRTMHGTFiles.size(); i++)
	{
		fileInfo = &(pSRTMHGTFiles[i]);
		fileInfo->GetWgs84BoundingBox(&minLat, &minLon, &maxLat, &maxLon);
		minNativeCoords[0] = minLon;
		minNativeCoords[1] = minLat;
		maxNativeCoords[0] = maxLon;
		maxNativeCoords[1] = maxLat;
		pRTree.Insert(minNativeCoords, maxNativeCoords, fileInfo);
	}
}

std::vector<SRTMHGTReader::Path> SRTMHGTReader::pGetPathList(const char* directory)
{
std::vector<SRTMHGTReader::Path> result;
	
	try
	{
		for (const auto& p : fs::recursive_directory_iterator(directory))
		{
			if (!fs::is_directory(p))
			{
			SRTMHGTReader::Path path;
			std::error_code sizeError;

				path.pathname = p.path().string();
				path.filename = p.path().filename().string();
				result.push_back(path);
			}
		}
	}
	catch(const std::exception& e)
	{
	}

	return result;
}

bool SRTMHGTReader::pIsFilenameSupported(std::string& filename, double* lat, double* lon)
{
	if( filename.length() < 7 )
		return false;

	char char0 = tolower(filename[0]);
	char char3 = tolower(filename[3]);
	char char4 = tolower(filename[4]);
	char charNS, charEW, charLat0, charLat1, charLon0, charLon1, charLon2;

	if( (char0=='n' || char0=='s') && (char3=='w' || char3=='e') )
	{
		charNS = char0;
		charEW = char3;
		charLat0 = filename[1];
		charLat1 = filename[2];
		charLon0 = filename[4];
		charLon1 = filename[5];
		charLon2 = filename[6];
	}
	else if( (char0=='w' || char0=='e') && (char4=='n' || char4=='s') )
	{
		charNS = char4;
		charEW = char0;
		charLat0 = filename[5];
		charLat1 = filename[6];
		charLon0 = filename[1];
		charLon1 = filename[2];
		charLon2 = filename[3];
	}
	else
		return false;

	if( isdigit(charLat0)==0 || isdigit(charLat1)==0 )
		return false;
	if( isdigit(charLon0)==0 || isdigit(charLon1)==0 || isdigit(charLon2)==0 )
		return false;

	*lat = (charLat0-48)*10 + (charLat1-48);
	if( charNS == 's')
		*lat = -(*lat);
	if( *lat < -90 || *lat > 90 )
		return false;

	*lon = (charLon0-48)*100 + (charLon1-48)*10 + (charLon2-48);
	if( charEW == 'w')
		*lon = -(*lon);
	if( *lon < -180 || *lon > 180 )
		return false;

	return true;
}

bool SRTMHGTReader::pGetValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, bool interpolate, float* value)
{
	if( interpolate == false )
	{
	int16_t tmpValue;
	bool success = pGetClosestValue(fileInfo, lat, lon, &tmpValue);
		*value = (float)tmpValue;
		return success;
	}
	else
		return pGetInterplValue(fileInfo, lat, lon, value);
}

bool SRTMHGTReader::pGetInterplValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, float* value)
{
uint32_t x1, x2, y1, y2;
double xDbl, yDbl;
int16_t val11, val12, val21, val22;
bool success;

	fileInfo->GetSurroundingPixelIndexes(lat, lon, &x1, &x2, &y1, &y2, &xDbl, &yDbl);

	success = true;
	success &= pReadFromFile(fileInfo, x1, y1, &val11);
	success &= pReadFromFile(fileInfo, x1, y2, &val12);
	success &= pReadFromFile(fileInfo, x2, y1, &val21);
	success &= pReadFromFile(fileInfo, x2, y2, &val22);
	if( success )
	{
	double result;

		fileInfo->BilinearInterpl(x1, x2, y1, y2, val11, val12, val21, val22, xDbl, yDbl, &result);
		*value = (float)result;
		return true;
	}

	// try to get closest value at last resort
	int16_t tmpValue;
	success = pGetClosestValue(fileInfo, lat, lon, &tmpValue);
	*value = (float) tmpValue;
	return success;
}

bool SRTMHGTReader::pGetClosestValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, int16_t* value,
									 double* closestPtLat/*=nullptr*/, double* closestPtLon/*=nullptr*/)
{
uint32_t x, y;

	fileInfo->GetPixelIndex(lat, lon, &x, &y);
	if( closestPtLat != nullptr && closestPtLon != nullptr)
		fileInfo->GetPixelWgs84Coord(x, y, closestPtLat, closestPtLon);
	return pReadFromFile(fileInfo, x, y, value);
}

bool SRTMHGTReader::pCompareFileInfoOnResolution(SRTMHGTFileInfo* fileInfo1, SRTMHGTFileInfo* fileInfo2)
{
	return (fileInfo1->ResolutionInMeters() < fileInfo2->ResolutionInMeters());
}

std::vector<SRTMHGTReader::SRTMHGTFileInfo*> SRTMHGTReader::pGetFileInfoList(double lat, double lon)
{
std::vector<SRTMHGTFileInfo*> result;
double pt[2] = {lon, lat};

	auto SearchCallback = [lat, lon, &result] (SRTMHGTFileInfo* fileInfo) -> bool
	{
		if( fileInfo->IsIn(lat, lon) == true )
			result.push_back(fileInfo);
		return true; // true to continue searching (in order to get all files containing the point)
	};
	pRTree.Search(pt, pt, SearchCallback);
	sort(result.begin(), result.end(), pCompareFileInfoOnResolution);
	return result;
}

bool SRTMHGTReader::pReadFromFile(SRTMHGTFileInfo* fileInfo, uint32_t x, uint32_t y, int16_t* value)
{
int16_t tmpValue;
uint32_t pixelSize = 2; // in bytes

	if( fileInfo->GetFromCache(x, y, value) == true)
	{
		if( *value == fileInfo->m_noDataValue )
			return false;
		return true;
	}

	if( fileInfo->m_filePtr == nullptr )
	{
		fileInfo->m_filePtr = fopen(fileInfo->m_pathname.c_str(), "rb");
		if(fileInfo->m_filePtr == nullptr)
			return false;
	}

	if( fseek(fileInfo->m_filePtr, (long int) ((y*fileInfo->m_rasterWidth*pixelSize) + (x*pixelSize)), SEEK_SET) != 0 )
		return false;
		
	if( fread(&tmpValue, pixelSize, 1, fileInfo->m_filePtr) != 1 )
		return false;

	*value = (tmpValue & 0xFF00) >> 8;
	*value += (tmpValue & 0xFF) << 8;

	fileInfo->Cache(x, y, *value);

	if( *value == fileInfo->m_noDataValue )
		return false;
	
	return true;
}
