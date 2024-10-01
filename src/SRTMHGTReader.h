#pragma once
#include "GeoRasterFileInfo.h"
#include <string>
#include <vector>
#include <map>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "RTree.h"
#pragma GCC diagnostic pop


class SRTMHGTReader
{
public:
	SRTMHGTReader();
	SRTMHGTReader(const SRTMHGTReader& original);
	virtual ~SRTMHGTReader();
	const SRTMHGTReader& operator=(const SRTMHGTReader& original);

	void SetDirectory(const char* directory);
	const char* GetDirectory() const;

	bool GetValue(double lat, double lon, bool interpolate, float* value);
	void CloseAllFiles(bool clearCaches);

private:
	class SRTMHGTFileInfo : public GeoRasterFileInfo
	{
	public:
		SRTMHGTFileInfo(std::string& pathname, uint32_t rasterHeight, uint32_t rasterWidth,
		                double minLat, double maxLat, double minLon, double maxLon,
		                double pixelHeightDeg, double pixelWidthDeg, int16_t noDataValue);
		SRTMHGTFileInfo(const SRTMHGTFileInfo& original);
		virtual ~SRTMHGTFileInfo();
		const SRTMHGTFileInfo& operator=(const SRTMHGTFileInfo& original);

		void Close(bool clearCache);
		void Cache(uint32_t x, uint32_t y, int16_t value);
		bool GetFromCache(uint32_t x, uint32_t y, int16_t* value);

		int16_t m_noDataValue;
		FILE* m_filePtr;

		struct CacheElement
		{
			int16_t data[32][32];
		};
		std::map<uint32_t, CacheElement> m_cache;
	};

	struct Path
	{
		std::string pathname;
		std::string filename;
	};

	void pUpdateFilesInfo(const char* directory);
	void pUpdateRTree();
	std::vector<Path> pGetPathList(const char* directory);
	bool pIsFilenameSupported(std::string& filename, double* lat, double* lon);

	bool pGetValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, bool interpolate, float* value);
	bool pGetInterplValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, float* value);
	bool pGetClosestValue(SRTMHGTFileInfo* fileInfo, double lat, double lon, int16_t* value, double* closestPtLat=nullptr, double* closestPtLon=nullptr);
	static bool pCompareFileInfoOnResolution(SRTMHGTFileInfo* fileInfo1, SRTMHGTFileInfo* fileInfo2);
	std::vector<SRTMHGTReader::SRTMHGTFileInfo*> pGetFileInfoList(double lat, double lon);
	bool pReadFromFile(SRTMHGTFileInfo* fileInfo, uint32_t x, uint32_t y, int16_t* value);

	std::string pDirectory;
	std::vector<SRTMHGTFileInfo> pSRTMHGTFiles;
	RTree<SRTMHGTFileInfo*, double, 2> pRTree;
	SRTMHGTFileInfo* pLastUsedFileInfo;
};