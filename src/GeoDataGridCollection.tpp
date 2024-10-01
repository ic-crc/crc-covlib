// Implementation of the GeoDataGridCollection template class,
// this file is included at the end of GeoDataGridCollection.h
#pragma once


template <typename T>
GeoDataGridCollection<T>::GeoDataGridCollection()
{
	pLastUsedGrid = nullptr;
}

template <typename T>
GeoDataGridCollection<T>::GeoDataGridCollection(const GeoDataGridCollection<T>& original)
{
	*this = original;
}

template <typename T>
GeoDataGridCollection<T>::~GeoDataGridCollection(void)
{

}

template <typename T>
const GeoDataGridCollection<T>& GeoDataGridCollection<T>::operator=(const GeoDataGridCollection<T>& original)
{
	if (&original == this)
		return *this;

	pGrids = original.pGrids;
	pRebuildRTree();
	pLastUsedGrid = nullptr;

	return *this;
}

template <typename T>
bool GeoDataGridCollection<T>::AddData(double minLat, double minLon, double maxLat, double maxLon, unsigned int sizeX, unsigned int sizeY,
	                                   const T* data, bool defineNoDataValue/*=false*/, T noDataValue/*=0*/)
{
GeoDataGrid<T> newGrid;
GeoDataGrid<T>* newGridPtr;

	pGrids.push_back(newGrid);
	pLastUsedGrid = nullptr; // the push_back may render the pointer invalid
	newGridPtr = &(pGrids[pGrids.size()-1]);
	if( newGridPtr->SetData(sizeX, sizeY, data) == false )
	{
		pGrids.pop_back();
		return false;
	}
	
	newGridPtr->SetBordersCoordinates(minLat, minLon, maxLat, maxLon);
	if( defineNoDataValue == true )
		newGridPtr->DefineNoDataValue(noDataValue);
	else
		newGridPtr->UndefineNoDataValue();

	// Note: pRTree stores indexes instead of pointers since adding a new grid to pGrids may
	//       render pointers to grids (i.e. &(pGrids[index])) invalid. This way we avoid
	//       rebuilding the whole RTree each time new data is added.
	pAddToTree(pGrids.size()-1);
	
	return true;
}

template <typename T>
void GeoDataGridCollection<T>::ClearData()
{
	pGrids.clear();
	pRTree.RemoveAll();
	pLastUsedGrid = nullptr;
}

template <typename T>
bool GeoDataGridCollection<T>::GetInterplData(double lat, double lon, T* data)
{
	// don't need pIsWithinExtent(), GetInterplData() will return false when outside extent
	if( pLastUsedGrid != nullptr /*&& pIsWithinExtent(pLastUsedGrid, lat, lon) == true*/ )
		if( pLastUsedGrid->GetInterplData(lat, lon, data) == true )
			return true;

	std::vector<GeoDataGrid<T>*> gridPtrs = pGetContainingGridPtrs(lat, lon);
	GeoDataGrid<T>* gridPtr;
	for(size_t i=0 ; i<gridPtrs.size() ; i++)
	{
		gridPtr = gridPtrs[i];
		if( gridPtr != pLastUsedGrid )
		{
			if( gridPtr->GetInterplData(lat, lon, data) == true )
			{
				pLastUsedGrid = gridPtr;
				return true;
			}
		}
	}

	pLastUsedGrid = nullptr;
	return false;
}

template <typename T>
bool GeoDataGridCollection<T>::GetClosestData(double lat, double lon, T* data)
{
	// don't need pIsWithinExtent(), GetClosestData() will return false when outside extent
	if( pLastUsedGrid != nullptr /*&& pIsWithinExtent(pLastUsedGrid, lat, lon) == true*/ )
		if( pLastUsedGrid->GetClosestData(lat, lon, data) == true )
			return true;

	std::vector<GeoDataGrid<T>*> gridPtrs = pGetContainingGridPtrs(lat, lon);
	GeoDataGrid<T>* gridPtr;
	for(size_t i=0 ; i<gridPtrs.size() ; i++)
	{
		gridPtr = gridPtrs[i];
		if( gridPtr != pLastUsedGrid )
		{
			if( gridPtr->GetClosestData(lat, lon, data) == true )
			{
				pLastUsedGrid = gridPtr;
				return true;
			}
		}
	}

	pLastUsedGrid = nullptr;
	return false;
}

template <typename T>
void GeoDataGridCollection<T>::pRebuildRTree()
{
	pRTree.RemoveAll();
	for (size_t i=0 ; i < pGrids.size() ; i++)
		pAddToTree(i);
	pLastUsedGrid = nullptr;
}

template <typename T>
void GeoDataGridCollection<T>::pAddToTree(size_t gridIndex)
{
GeoDataGrid<T>* gridPtr = &(pGrids[gridIndex]);
double minLat, minLon, maxLat, maxLon;
double minCoords[2];
double maxCoords[2];

	if( pGetExtent(gridPtr, &minLat, &minLon, &maxLat, &maxLon) == true )
	{
		minCoords[0] = minLon;
		minCoords[1] = minLat;
		maxCoords[0] = maxLon;
		maxCoords[1] = maxLat;
		pRTree.Insert(minCoords, maxCoords, gridIndex);
	}
}

template <typename T>
bool GeoDataGridCollection<T>::pCompareGridsOnResolution(GeoDataGrid<T>* gridPtr1, GeoDataGrid<T>* gridPtr2)
{
	return (gridPtr1->ResolutionInDegrees() < gridPtr2->ResolutionInDegrees());
}

template <typename T>
std::vector<GeoDataGrid<T>*> GeoDataGridCollection<T>::pGetContainingGridPtrs(double lat, double lon)
{
std::vector<GeoDataGrid<T>*> result;
double pt[2] = {lon, lat};

	auto SearchCallback = [this, lat, lon, &result] (size_t gridIndex) -> bool
	{
		GeoDataGrid<T>* gridPtr = &(this->pGrids[gridIndex]);
		if( this->pIsWithinExtent(gridPtr, lat, lon) == true )
			result.push_back(gridPtr);
		return true; // true to continue searching (in order to get all grids containing the point)
	};
	pRTree.Search(pt, pt, SearchCallback);
	sort(result.begin(), result.end(), pCompareGridsOnResolution);
	return result;
}

// The extent for which we can get a value from the data grid (i.e. up to half a pixel around the borders)
template <typename T>
bool GeoDataGridCollection<T>::pGetExtent(const GeoDataGrid<T>* gridPtr, double* minLat, double* minLon, double* maxLat, double* maxLon) const
{
double borderMinLat, borderMinLon, borderMaxLat, borderMaxLon;
double halfPixelSizeAlongLat, halfPixelSizeAlongLon;
unsigned int sizeX = gridPtr->SizeX(), sizeY = gridPtr->SizeY();

	if( sizeX > 1 && sizeY > 1 )
	{
		gridPtr->GetBordersCoordinates(&borderMinLat, &borderMinLon, &borderMaxLat, &borderMaxLon);
		halfPixelSizeAlongLat = ((borderMaxLat-borderMinLat)/(sizeY-1))/2.0;
		halfPixelSizeAlongLon = ((borderMaxLon-borderMinLon)/(sizeX-1))/2.0;
		*minLon = borderMinLon - halfPixelSizeAlongLon;
		*minLat = borderMinLat - halfPixelSizeAlongLat;
		*maxLon = borderMaxLon + halfPixelSizeAlongLon;
		*maxLat = borderMaxLat + halfPixelSizeAlongLat;
		return true;
	}

	return false;
}

template <typename T>
bool GeoDataGridCollection<T>::pIsWithinExtent(const GeoDataGrid<T>* gridPtr, double lat, double lon) const
{
double minLat, minLon, maxLat, maxLon;

	if( pGetExtent(gridPtr, &minLat, &minLon, &maxLat, &maxLon) == true )
	{
		if( lat > maxLat )
			return false;
		if( lat < minLat )
			return false;
		if( lon > maxLon )
			return false;
		if( lon < minLon )
			return false;
		return true;
	}
	return false;
}