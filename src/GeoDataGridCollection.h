/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "GeoDataGrid.h"
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "RTree.h"
#pragma GCC diagnostic pop
#include <vector>


template <typename T>
class GeoDataGridCollection
{
public:
	GeoDataGridCollection();
	GeoDataGridCollection(const GeoDataGridCollection& original);
	virtual ~GeoDataGridCollection(void);

	const GeoDataGridCollection& operator=(const GeoDataGridCollection& original);

	bool AddData(double minLat, double minLon, double maxLat, double maxLon, unsigned int sizeX, unsigned int sizeY,
	             const T* data, bool defineNoDataValue=false, T noDataValue=0);
	void ClearData();
	bool GetInterplData(double lat, double lon, T* data);
	bool GetClosestData(double lat, double lon, T* data);

private:
	void pRebuildRTree();
	void pAddToTree(size_t gridIndex);
	std::vector<GeoDataGrid<T>*> pGetContainingGridPtrs(double lat, double lon);
	static bool pCompareGridsOnResolution(GeoDataGrid<T>* gridPtr1, GeoDataGrid<T>* gridPtr2);
	bool pGetExtent(const GeoDataGrid<T>* gridPtr, double* minLat, double* minLon, double* maxLat, double* maxLon) const;
	bool pIsWithinExtent(const GeoDataGrid<T>* gridPtr, double lat, double lon) const;

	std::vector<GeoDataGrid<T>> pGrids;
	RTree<size_t, double, 2> pRTree;
	GeoDataGrid<T>* pLastUsedGrid;
};

#include "GeoDataGridCollection.tpp"