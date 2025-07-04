/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

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

	void SetTotalSizeLimit(uint64_t numBytes);
	void SetCacheEntrySize(uint8_t pixelSizeInBytes, uint32_t stripOrTileSizeInBytes);

	bool GetValue(uint32_t stripOrTileIndex, uint32_t byteOffset, void* value);
	void CacheStripData(uint32_t stripIndex, void* stripData, uint32_t stripDataSizeInBytes);
	void CacheTileData(uint32_t tileIndex, void* tileData, uint32_t tileDataSizeInBytes);
	void Clear();

private:
	struct CacheEntry
	{
		uint64_t m_lastHitNo;
		uint8_t* m_data;
	};

	void pManageSizeLimit();

	std::map<uint32_t, CacheEntry> pMap;
	uint64_t pCurHitNo;
	uint8_t pBytesPerPixel;
	uint32_t pCacheEntryDataSizeInBytes;
	uint64_t pCacheLimitInBytes;
};