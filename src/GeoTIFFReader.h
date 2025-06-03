/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "GeoRasterFileInfo.h"
#include "GeoTIFFFileCache.h"
#include <string>
#include <vector>
#include <tiffio.h>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "RTree.h"
#pragma GCC diagnostic pop


class GeoTIFFReader
{
public:
	GeoTIFFReader();
	GeoTIFFReader(const GeoTIFFReader& original);
	virtual ~GeoTIFFReader();
	const GeoTIFFReader& operator=(const GeoTIFFReader& original);

	void SetDirectory(const char* directory, bool useIndexFile=false, bool overwriteIndexFile=false);
	const char* GetDirectory() const;

	void SetFile(const char* pathname);
	const char* GetFile() const;

	bool GetClosestValue(double lat, double lon, void* value, double* closestPtLat=NULL, double* closestPtLon=NULL);
	bool GetClosestIntValue(double lat, double lon, int* value, double* closestPtLat=NULL, double* closestPtLon=NULL);
	bool GetClosestFltValue(double lat, double lon, float* value, double* closestPtLat=NULL, double* closestPtLon=NULL);
	bool GetInterplValue(double lat, double lon, float* value);

	void CloseAllFiles(bool clearCaches);

protected:
	class GeoTIFFFileInfo : public GeoRasterFileInfo
	{
	public:
		GeoTIFFFileInfo();
		GeoTIFFFileInfo(const GeoTIFFFileInfo& original);
		virtual ~GeoTIFFFileInfo();
		const GeoTIFFFileInfo& operator=(const GeoTIFFFileInfo& original);

		void Clear();
		void Close();
		bool ValidateAndSynch();
		void Print();

		// TIFF tags
		uint16_t m_compression;
		uint32_t m_rowsPerStrip;
		uint16_t m_bitsPerSample;
		uint16_t m_samplesPerPixel;
		uint16_t m_sampleFormat;
		uint32_t m_tileHeight;
		uint32_t m_tileWidth;

		// Other TIFF info
		int32_t m_bytesPerStrip;
		int32_t m_bytesPerTile; 

		// GDAL specific tags
		int32_t m_noDataValue;
		bool m_noDataValuePresent;

		// GeoTIFF tags
		double m_ModelPixelScale[3];
		double m_ModelTiepoint[2][3];

		// GeoTIFF keys
		uint16_t m_GTModelTypeGeoKey;
		uint16_t m_GTRasterTypeGeoKey;
		uint16_t m_GeogAngularUnitsGeoKey;
		uint16_t m_ProjectedCSTypeGeoKey;
		uint16_t m_ProjLinearUnitsGeoKey;
		uint16_t m_GeographicTypeGeoKey;
		std::vector<double> m_GeogTOWGS84GeoKey;
		std::string m_GTCitationGeoKey;
		std::string m_GeogCitationGeoKey;
		double m_GeogSemiMajorAxisGeoKey;
		double m_GeogInvFlatteningGeoKey;

		TIFF* m_tiffPtr;
		void* m_readBuf;
		GeoTIFFFileCache m_cache;
	};

	bool pCreateIndexFile(bool overwriteIfExists);
	bool pReadIndexFile();
	bool pReadTagsAndKeys(const char* pathname, GeoTIFFFileInfo& tiffInfo);
	bool pGetGeoTagDoubleArrayValue(TIFF* tif, uint32_t tag, double* dst, uint32_t dstSize);
	bool pGetGeoKeyStringValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::string& dst);
	bool pGetGeoKeyDoubleArrayValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, std::vector<double>& dst);
	bool pGetGeoKeyDoubleValue(TIFF* tif, uint16_t tiffTagLocation, uint16_t numValues, uint16_t valueOffset, double& dst);
	void pUpdateFilesInfo(const char* directory);
	void pUpdateFilesCacheSettings();
	void pUpdateRTree();
	std::vector<std::string> pGetPathnameList(const char* directory, const char* fileExtension);
	void pToLowercase(std::string& s);
	std::string pGetRelativePath(const char* baseDir, const char* pathname);
	typedef bool (GeoTIFFReader::*GetValueMemberFunc)(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* data, double* closestPtLat, double* closestPtLon);
	bool pGetValue(double lat, double lon, void* value, double* closestPtLat, double* closestPtLon, GetValueMemberFunc getValueFunc);
	bool pGetClosestValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value, double* closestPtLat, double* closestPtLon);
	bool pGetClosestIntValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value, double* closestPtLat, double* closestPtLon);
	bool pGetClosestFltValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value, double* closestPtLat, double* closestPtLon);
	bool pGetInterplFltValue(GeoTIFFFileInfo* tiffInfo, double lat, double lon, void* value, double* closestPtLat, double* closestPtLon);	
	bool pGetPixelValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, void* value);
	bool pGetPixelFltValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, float* value);
	bool pGetPixelIntValue(GeoTIFFFileInfo* tiffInfo, uint32_t x, uint32_t y, int* value);
	bool pIsNoDataValue(GeoTIFFFileInfo* tiffInfo, void* value);
	static bool pCompareTiffInfoOnResolution(GeoTIFFFileInfo* tiffInfo1, GeoTIFFFileInfo* tiffInfo2);
	std::vector<GeoTIFFReader::GeoTIFFFileInfo*> pGetGeoTiffFileInfoList(double lat, double lon);
	void pSerializeTiffInfoFile(std::ostream& os, GeoTIFFFileInfo& tiffInfo);
	void pDeserializeTiffInfoFile(std::istream& is, GeoTIFFFileInfo& tiffInfo);
	void pSerializeString(std::ostream& os, std::string& str);
	void pDeserializeString(std::istream& is, std::string& str);
	void pSerializeDoubleVector(std::ostream& os, std::vector<double>& v);
	void pDeserializeDoubleVector(std::istream& is, std::vector<double>& v);

	std::string pDir;
	std::string pFile;
	std::vector<GeoTIFFFileInfo> pGeoTiffs;
	RTree<GeoTIFFFileInfo*, double, 2> pGeoTiffsRTree;
	GeoTIFFFileInfo* pLastTiffUsed;

	static const int32_t GEOTIFF_INDEX_VERSION = 3;

	static const int TIFF_UINT8 = (8 << 16) + 1;
	static const int TIFF_INT8 = (8 << 16) + 2;
	static const int TIFF_UINT16 = (16 << 16) + 1;
	static const int TIFF_INT16 = (16 << 16) + 2;
	static const int TIFF_UINT32 = (32 << 16) + 1;
	static const int TIFF_INT32 = (32 << 16) + 2;
	static const int TIFF_FLOAT32 = (32 << 16) + 3;

	// For debugging purposes
	//int pCacheHitCount;
	//int pCacheMissCount;
};