#pragma once
#include <map>
#include <queue>
#include <cstdint>


class GeoTIFFFileCache
{
public:
	GeoTIFFFileCache();
	GeoTIFFFileCache(const GeoTIFFFileCache& original);
	virtual ~GeoTIFFFileCache();
	const GeoTIFFFileCache& operator=(const GeoTIFFFileCache& original);

	void SetTotalSizeLimit(uint32_t numBytes);
	void SetCacheEntrySize(uint32_t numPixels, uint8_t pixelSizeInBytes, uint32_t maxStripOrTileSizeInBytes=0);

	bool GetValue(uint32_t stripOrTileIndex, int64_t byteOffset, void* value);
	void CacheStripData(uint32_t stripIndex, void* stripData, int64_t stripDataSizeInBytes, uint32_t requestedByteOffset);
	void CacheTileData(uint32_t tileIndex, void* tileData, int64_t tileDataSizeInBytes, uint32_t requestedByteOffset);
	void Clear();

private:
	struct CacheEntry
	{
		uint64_t m_lastHitNo;
		uint8_t* m_data;
	};

	void pManageSizeLimit();

	std::map<uint64_t, CacheEntry> pMap;
	uint64_t pCurHitNo;
	uint8_t pBytesPerPixel;
	uint8_t pNumSharedBitsPerEntry;
	uint32_t pCacheEntryDataSizeInBytes;
	uint32_t pCacheLimitInBytes;
};