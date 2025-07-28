/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoTIFFFileCache.h"
#include <cmath>
#include <cstring>


GeoTIFFFileCache::GeoTIFFFileCache()
{
	pBytesPerPixel = 4;
	pCacheEntryDataSizeInBytes = 256;
	pCacheLimitInBytes = 4E6;
	pCurHitNo = 0;
}

GeoTIFFFileCache::GeoTIFFFileCache(const GeoTIFFFileCache& original)
{
	*this = original;
}

GeoTIFFFileCache::~GeoTIFFFileCache()
{
	std::map<uint32_t, CacheEntry>::iterator iter;
	for (iter = pMap.begin(); iter != pMap.end(); ++iter)
		delete [] iter->second.m_data;
}

const GeoTIFFFileCache& GeoTIFFFileCache::operator=(const GeoTIFFFileCache& original)
{
	if( &original == this )
		return *this;

	Clear();
	pBytesPerPixel = original.pBytesPerPixel;
	pCacheEntryDataSizeInBytes = original.pCacheEntryDataSizeInBytes;
	pCacheLimitInBytes = original.pCacheLimitInBytes;

	// Note: do NOT copy pMap or pCurHitNo (i.e. do not copy cached data)

	return *this;
}

void GeoTIFFFileCache::SetTotalSizeLimit(uint64_t numBytes)
{
	if( pCacheLimitInBytes != numBytes )
	{
		pCacheLimitInBytes = numBytes;
		pManageSizeLimit();
	}
}

// pixelSizeInBytes: size of a pixel value in bytes (usually 1, 2 or 4)
// stripOrTileSizeInBytes: size of a strip or tile (depending on the format of the tiff)
//                         for the file in bytes.
void GeoTIFFFileCache::SetCacheEntrySize(uint8_t pixelSizeInBytes, uint32_t stripOrTileSizeInBytes)
{
	pCacheEntryDataSizeInBytes = stripOrTileSizeInBytes;
	pBytesPerPixel = pixelSizeInBytes;
	Clear();
}

bool GeoTIFFFileCache::GetValue(uint32_t stripOrTileIndex, uint32_t byteOffset, void* value)
{
std::map<uint32_t, CacheEntry>::iterator iter = pMap.find(stripOrTileIndex);

	if( iter != pMap.end() )
	{
		if( (byteOffset+pBytesPerPixel) <= pCacheEntryDataSizeInBytes )
		{
			memcpy(value, iter->second.m_data + byteOffset, pBytesPerPixel);
			iter->second.m_lastHitNo = ++pCurHitNo;
			return true;
		}
	}

	return false;
}

void GeoTIFFFileCache::CacheStripData(uint32_t stripIndex, void* stripData, uint32_t stripDataSizeInBytes)
{
uint32_t key = stripIndex;

	if( pMap.count(key) == 0 )
	{
	CacheEntry* entryPtr;

		entryPtr = &(pMap[key]);
		entryPtr->m_lastHitNo = ++pCurHitNo;
		entryPtr->m_data = new uint8_t[pCacheEntryDataSizeInBytes];
		memcpy(entryPtr->m_data, stripData, std::min(pCacheEntryDataSizeInBytes, stripDataSizeInBytes));
		pManageSizeLimit();
	}
}

void GeoTIFFFileCache::CacheTileData(uint32_t tileIndex, void* tileData, uint32_t tileDataSizeInBytes)
{
	CacheStripData(tileIndex, tileData, tileDataSizeInBytes);
}

void GeoTIFFFileCache::Clear()
{
	std::map<uint32_t, CacheEntry>::iterator iter;
	for (iter = pMap.begin(); iter != pMap.end(); ++iter)
		delete [] iter->second.m_data;
	pMap.clear();
	pCurHitNo = 0;
}

void GeoTIFFFileCache::pManageSizeLimit()
{
	while( (pMap.size()*pCacheEntryDataSizeInBytes) > pCacheLimitInBytes )
	{
	uint64_t minHitNo = UINT64_MAX;
	std::queue< std::map<uint32_t, CacheEntry>::iterator > iterQueue;
	std::map<uint32_t, CacheEntry>::iterator iter;

		for(iter = pMap.begin() ; iter != pMap.end() ; ++iter)
		{
			if(iter->second.m_lastHitNo < minHitNo )
			{
				minHitNo = iter->second.m_lastHitNo;
				iterQueue.push(iter);
				if( iterQueue.size() > 10 ) // will delete up to 10 cache entries to make up space
					iterQueue.pop();
			}
		}
		while( iterQueue.size() != 0 )
		{
			iter = iterQueue.front();
			delete [] iter->second.m_data;
			pMap.erase(iter);
			iterQueue.pop();
		}
	}
}