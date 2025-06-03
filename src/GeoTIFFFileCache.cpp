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
	pNumSharedBitsPerEntry = 8;
	pCacheEntryDataSizeInBytes = pow(2, pNumSharedBitsPerEntry);
	pCacheLimitInBytes = 4E6;
	pCurHitNo = 0;
}

GeoTIFFFileCache::GeoTIFFFileCache(const GeoTIFFFileCache& original)
{
	*this = original;
}

GeoTIFFFileCache::~GeoTIFFFileCache()
{
	std::map<uint64_t, CacheEntry>::iterator iter;
	for (iter = pMap.begin(); iter != pMap.end(); ++iter)
		delete [] iter->second.m_data;
}

const GeoTIFFFileCache& GeoTIFFFileCache::operator=(const GeoTIFFFileCache& original)
{
	if (&original == this)
		return *this;

	Clear();
	pBytesPerPixel = original.pBytesPerPixel;
	pNumSharedBitsPerEntry = original.pNumSharedBitsPerEntry;
	pCacheEntryDataSizeInBytes = original.pCacheEntryDataSizeInBytes;
	pCacheLimitInBytes = original.pCacheLimitInBytes;

	// Note: do NOT copy pMap or pCurHitNo (i.e. do not copy cached data)

	return *this;
}

void GeoTIFFFileCache::SetTotalSizeLimit(uint32_t numBytes)
{
	if( pCacheLimitInBytes != numBytes )
	{
		pCacheLimitInBytes = numBytes;
		pManageSizeLimit();
	}
}

// numPixels: number of pixel values to be stored in each new cache entry, should be a power of two
// pixelSizeInBytes: size of a pixel value in bytes (usually 1, 2 or 4)
// maxStripOrTileSizeInBytes: maximum size of a strip or tile (depending on the format of the tiff)
//                            for the file in bytes. This argument can be omitted.
void GeoTIFFFileCache::SetCacheEntrySize(uint32_t numPixels, uint8_t pixelSizeInBytes, uint32_t maxStripOrTileSizeInBytes/*=0*/)
{
	pCacheEntryDataSizeInBytes = numPixels*pixelSizeInBytes;
	pNumSharedBitsPerEntry = (uint8_t) log2l(pCacheEntryDataSizeInBytes);
	if( maxStripOrTileSizeInBytes != 0 )
		pCacheEntryDataSizeInBytes = std::min(pCacheEntryDataSizeInBytes, maxStripOrTileSizeInBytes);
	pBytesPerPixel = pixelSizeInBytes;
	Clear();
}

bool GeoTIFFFileCache::GetValue(uint32_t stripOrTileIndex, int64_t byteOffset, void* value)
{
uint64_t key = ((uint64_t)stripOrTileIndex << 32) + ((uint64_t)byteOffset >> pNumSharedBitsPerEntry);

	std::map<uint64_t, CacheEntry>::iterator iter = pMap.find(key);
	if (iter != pMap.end())
	{
		memcpy(value, iter->second.m_data + (byteOffset%pCacheEntryDataSizeInBytes), pBytesPerPixel);
		iter->second.m_lastHitNo = ++pCurHitNo;
		//pManageSizeLimit();
		return true;
	}

	return false;
}

void GeoTIFFFileCache::CacheStripData(uint32_t stripIndex, void* stripData, int64_t stripDataSizeInBytes, uint32_t requestedByteOffset)
{
uint64_t key = ((uint64_t)stripIndex << 32) + (requestedByteOffset >> pNumSharedBitsPerEntry);

	if( pMap.count(key) == 0)
	{
	uint32_t cacheEntryStartByte = (requestedByteOffset >> pNumSharedBitsPerEntry)*pCacheEntryDataSizeInBytes;
	uint32_t numRemainingBytes = stripDataSizeInBytes - cacheEntryStartByte;
	CacheEntry* entryPtr;

		entryPtr = &(pMap[key]);
		entryPtr->m_lastHitNo = ++pCurHitNo;
		entryPtr->m_data = new uint8_t[pCacheEntryDataSizeInBytes];
		memcpy(entryPtr->m_data, ((uint8_t*)stripData) + cacheEntryStartByte, std::min(pCacheEntryDataSizeInBytes, numRemainingBytes));
		pManageSizeLimit();
	}
}

void GeoTIFFFileCache::CacheTileData(uint32_t tileIndex, void* tileData, int64_t tileDataSizeInBytes, uint32_t requestedByteOffset)
{
	CacheStripData(tileIndex, tileData, tileDataSizeInBytes, requestedByteOffset);
}

void GeoTIFFFileCache::Clear()
{
	std::map<uint64_t, CacheEntry>::iterator iter;
	for (iter = pMap.begin(); iter != pMap.end(); ++iter)
		delete [] iter->second.m_data;
	pMap.clear();
	pCurHitNo = 0;
}

void GeoTIFFFileCache::pManageSizeLimit()
{
	while( (pMap.size()*(pCacheEntryDataSizeInBytes+sizeof(CacheEntry))) > pCacheLimitInBytes )
	{
	uint64_t minHitNo = UINT64_MAX;
	std::queue< std::map<uint64_t, CacheEntry>::iterator > iterQueue;
	std::map<uint64_t, CacheEntry>::iterator iter;

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