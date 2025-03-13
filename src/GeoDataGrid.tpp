// Implementation of the GeoDataGrid template class,
// this file is included at the end of GeoDataGrid.h
#pragma once
#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif
#include <cstring>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <limits>



template <typename T>
void GeoDataGrid<T>::pInit()
{
	pSizeX = 0;
	pSizeY = 0;
	pDataPtr = nullptr;
	pMinLat = 0;
	pMinLon = 0;
	pMaxLat = 0;
	pMaxLon = 0;
	pDataUnit = "N/A";
	pDataDescription = "data";
	pIsNoDataValueDefined = false;
	pNoDataValue = 0;
}

template <typename T>
GeoDataGrid<T>::GeoDataGrid()
{
	pInit();
}

template <typename T>
GeoDataGrid<T>::GeoDataGrid(unsigned int sizeX, unsigned int sizeY)
{
	pInit();
	Clear(sizeX, sizeY);
}

template <typename T>
GeoDataGrid<T>::GeoDataGrid(const GeoDataGrid<T>& original)
{
	pDataPtr = nullptr;
	*this = original;
}

template <typename T>
GeoDataGrid<T>::~GeoDataGrid(void)
{
	if( pDataPtr != nullptr )
		delete [] pDataPtr;
}

template <typename T>
const GeoDataGrid<T>& GeoDataGrid<T>::operator=(const GeoDataGrid<T>& original)
{
	if( &original != this )
	{
		if( pDataPtr != nullptr )
			delete [] pDataPtr;
		pDataPtr = nullptr;

		pSizeX = original.pSizeX;
		pSizeY = original.pSizeY;
		pMinLat = original.pMinLat;
		pMinLon = original.pMinLon;
		pMaxLat = original.pMaxLat;
		pMaxLon = original.pMaxLon;
		pDataUnit = original.pDataUnit;
		pDataDescription = original.pDataDescription;
		pIsNoDataValueDefined = original.pIsNoDataValueDefined;
		pNoDataValue = original.pNoDataValue;

		try
		{
			if( pSizeX*pSizeY != 0 )
				pDataPtr = new T[pSizeX*pSizeY]; // may succeed even if allocation size is 0
		}
		catch(const std::bad_alloc &)
		{
			pSizeX = pSizeY = 0;
			pDataPtr = nullptr;
		}

		if( pDataPtr != nullptr && original.pDataPtr != nullptr )
			memcpy(pDataPtr, original.pDataPtr, pSizeX*pSizeY*sizeof(T));
	}

	return *this;
}

template <typename T>
bool GeoDataGrid<T>::Clear(unsigned int newSizeX, unsigned int newSizeY)
{
	if( newSizeX < 2 )
		newSizeX = 2;
	if( newSizeY < 2 )
		newSizeY = 2;

	if( pDataPtr != nullptr )
	{
		delete [] pDataPtr;
		pDataPtr = nullptr;
	}

	pSizeX = newSizeX;
	pSizeY = newSizeY;

	try
	{
		pDataPtr = new T[pSizeX*pSizeY];
	}
	catch(const std::bad_alloc &)
	{
		pSizeX = pSizeY = 0;
		pDataPtr = nullptr;
	}

	if( pDataPtr != nullptr)
	{
		memset(pDataPtr, 0, pSizeX*pSizeY*sizeof(T));
		return true;
	}
	else
		return false;
}

template <typename T>
void GeoDataGrid<T>::DefineNoDataValue(T noDataValue)
{
	pNoDataValue = noDataValue;
	pIsNoDataValueDefined = true;
}

template <typename T>
bool GeoDataGrid<T>::IsNoDataValueDefined() const
{
	return pIsNoDataValueDefined;
}

template <typename T>
T GeoDataGrid<T>::GetNoDataValue() const
{
	return pNoDataValue;
}

template <typename T>
void GeoDataGrid<T>::UndefineNoDataValue()
{
	pNoDataValue = 0;
	pIsNoDataValueDefined = false;
}

template <typename T>
unsigned int GeoDataGrid<T>::SizeX() const
{
	return pSizeX;
}

template <typename T>
unsigned int GeoDataGrid<T>::SizeY() const
{
	return pSizeY;
}

template <typename T>
void GeoDataGrid<T>::SetBordersCoordinates(double minLat, double minLon, double maxLat, double maxLon)
{
	pMinLat = std::min(minLat, maxLat);
	pMinLon = std::min(minLon, maxLon);
	pMaxLat = std::max(minLat, maxLat);
	pMaxLon = std::max(minLon, maxLon);
}

template <typename T>
void GeoDataGrid<T>::GetBordersCoordinates(double* minLat, double* minLon, double* maxLat, double* maxLon) const
{
	*minLat = pMinLat;
	*minLon = pMinLon;
	*maxLat = pMaxLat;
	*maxLon = pMaxLon;
}

template <typename T>
double GeoDataGrid<T>::ResolutionInDegrees() const
{
double latResDeg = std::numeric_limits<T>::max();
double lonResDeg = std::numeric_limits<T>::max();

	if( pSizeY > 1 )
		latResDeg = (pMaxLat-pMinLat)/(pSizeY-1);
	if( pSizeX > 1 )
		lonResDeg = (pMaxLon-pMinLon)/(pSizeX-1);

	return std::max(latResDeg, lonResDeg);
}

template <typename T>
Position GeoDataGrid<T>::GetPos(unsigned int x, unsigned int y) const
{
double deltaX;
double deltaY;
Position pos;

	if( pSizeX == 0 || pSizeX == 1 )
		deltaX = 0;
	else
		deltaX = (pMaxLon - pMinLon) / (pSizeX - 1);
	if( pSizeY == 0 || pSizeY == 1 )
		deltaY = 0;
	else
		deltaY = (pMaxLat - pMinLat) / (pSizeY - 1);

	pos.m_lat = pMinLat + (deltaY*y);
	pos.m_lon = pMinLon + (deltaX*x);

	return pos;
}

template <typename T>
void GeoDataGrid<T>::SetData(unsigned int x, unsigned int y, T data)
{
	if( pDataPtr == nullptr )
		return;

	if( x >= pSizeX )
		return;

	if( y >= pSizeY )
		return;

	pDataPtr[(y*pSizeX)+x] = data;	
}

template <typename T>
bool GeoDataGrid<T>::SetData(unsigned int sizeX, unsigned int sizeY, const T* data)
{
bool success = true;

	if( pSizeX != sizeX || pSizeY != sizeY )
		success = Clear(sizeX, sizeY);

	if( success == false )
		return false;

	memcpy(pDataPtr, data, pSizeX*pSizeY*sizeof(T));

	return true;
}

template <typename T>
T GeoDataGrid<T>::GetData(unsigned int x, unsigned int y) const
{
	if( pDataPtr == nullptr )
		return 0;

	if( x >= pSizeX )
		return 0;

	if( y >= pSizeY )
		return 0;

	return pDataPtr[(y*pSizeX)+x];
}

template <typename T>
T* GeoDataGrid<T>::GetDataPtr() const
{
	return pDataPtr;
}

template <typename T>
bool GeoDataGrid<T>::GetInterplData(double lat, double lon, T* data) const
{
double deltaLat, deltaLon;
double x, y;
unsigned int x0, x1, y0, y1;
double t, u;
T valPt00, valPt01, valPt10, valPt11;

	*data = 0;

	if( pDataPtr == nullptr || pSizeX == 0 || pSizeY == 0 )
		return false;

	deltaLat = pMaxLat - pMinLat;
	if( deltaLat == 0 )
		return false;
	deltaLon = pMaxLon - pMinLon;
	if( deltaLon == 0 )
		return false;
	x = (lon - pMinLon)*(pSizeX-1)/deltaLon;
	y = (lat - pMinLat)*(pSizeY-1)/deltaLat;

	if( x >= 0 && y >= 0 )
	{
		x0 = (unsigned int) x;
		x1 = x0 + 1;
		y0 = (unsigned int) y;
		y1 = y0 + 1;

		if( x1 < pSizeX && y1 < pSizeY )
		{
			t = x - x0;
			u = y - y0;

			valPt00 = pDataPtr[(y0*pSizeX)+x0];
			valPt01 = pDataPtr[(y1*pSizeX)+x0];
			valPt10 = pDataPtr[(y0*pSizeX)+x1];
			valPt11 = pDataPtr[(y1*pSizeX)+x1];

			if( pIsNoDataValueDefined == true )
			{
				if( valPt00==pNoDataValue && valPt01==pNoDataValue && valPt10==pNoDataValue && valPt11==pNoDataValue )
					return false;
				if( valPt00==pNoDataValue || valPt01==pNoDataValue || valPt10==pNoDataValue || valPt11==pNoDataValue )
					return GetClosestData(lat, lon, data);
			}

			// bilinear interpolation
			*data = (((1-t)*(1-u)*valPt00) + (t*(1-u)*valPt10) + ((1-t)*u*valPt01) + (t*u*valPt11));
			return true;
		}
	}

	return GetClosestData(lat, lon, data);
}

template <typename T>
bool GeoDataGrid<T>::GetClosestData(double lat, double lon, T* data) const
{
double deltaLat, deltaLon;
long x, y;

	*data = 0;

	if( pDataPtr == nullptr || pSizeX == 0 || pSizeY == 0 )
		return false;

	deltaLat = pMaxLat - pMinLat;
	if( deltaLat == 0 )
		return false;
	deltaLon = pMaxLon - pMinLon;
	if( deltaLon == 0 )
		return false;
	x = lround((lon-pMinLon)*(pSizeX-1)/(deltaLon));
	y = lround((lat-pMinLat)*(pSizeY-1)/(deltaLat));
	
	if( x >= 0 && y >= 0 )
	{
	unsigned int ux = (unsigned int) x;
	unsigned int uy = (unsigned int) y;

		if( ux < pSizeX && uy < pSizeY )
		{
			*data = pDataPtr[(uy*pSizeX)+ux];
			if( pIsNoDataValueDefined == true && *data == pNoDataValue )
				return false;
			return true;
		}
	}

	return false;
}

template <typename T>
void GeoDataGrid<T>::SetDataUnit(const char* unit)
{
	pDataUnit = unit;
}

template <typename T>
const char* GeoDataGrid<T>::GetDataUnit() const
{
	return pDataUnit.c_str();
}

template <typename T>
void GeoDataGrid<T>::SetDataDescription(const char* description)
{
	pDataDescription = description;
}

template <typename T>
const char* GeoDataGrid<T>::GetDataDescription() const
{
	return pDataDescription.c_str();
}

template <typename T>
bool GeoDataGrid<T>::ExportToTextFile(const char* pathname, const char* dataColName) const
{
std::ofstream txtFile;
Position pos;
bool success = false;

	txtFile.open(pathname, std::ios::out | std::ios::trunc);
	if(txtFile)
	{
		txtFile << "latitude,longitude," << dataColName << '\n';
		for(unsigned int y=0 ; y<pSizeY ; y++)
		{
			for(unsigned int x=0 ; x<pSizeX ; x++)
			{
				pos = GetPos(x, y);
				txtFile << std::fixed << std::setprecision(6);
				txtFile << pos.m_lat << "," << pos.m_lon << ",";
				txtFile << std::fixed << std::setprecision(3);
				txtFile << GetData(x, y) << '\n';
			}
		}
		success = true;
	}
	txtFile.close();

	return success;
}

template <typename T>
bool GeoDataGrid<T>::ExportToBILFile(const char* pathname) const
{
bool success = true;

	if(pSizeX==0 || pSizeY==0 || pDataPtr==nullptr)
		return false;

	fs::path bilPathname = pathname;
	bilPathname.replace_extension("bil");
	std::ofstream bilFile;
	bilFile.open(bilPathname, std::ios::out | std::ios::trunc | std::ios::binary);
	if(bilFile)
	{
		unsigned int rowSizeInBytes = sizeof(T)*pSizeX;
		for(unsigned int y=0 ; y<pSizeY ; y++)
			bilFile.write(((char*)pDataPtr) + ((pSizeY-1-y)*rowSizeInBytes), (std::streamsize) rowSizeInBytes);
	}
	else
		success = false;
	bilFile.close();

	fs::path hdrPathname = pathname;
	hdrPathname.replace_extension("hdr");
	std::ofstream hdrFile;
	hdrFile.open(hdrPathname, std::ios::out | std::ios::trunc);
	if(hdrFile)
	{
		hdrFile << "byteorder I" << '\n';
		hdrFile << "nrows " << pSizeY << '\n';
		hdrFile << "ncols " << pSizeX << '\n';
		hdrFile << "nbands 1" << '\n';
		hdrFile << "nbits " << sizeof(T)*8 << '\n';
        if( std::is_floating_point<T>::value )
		    hdrFile << "pixeltype float" << '\n';
        else if( std::is_signed<T>::value )
            hdrFile << "pixeltype signedint" << '\n';
        else
            hdrFile << "pixeltype unignedint" << '\n';
		hdrFile << std::fixed << std::setprecision(16);
		hdrFile << "ulxmap " << pMinLon << '\n';
		hdrFile << "ulymap " << pMaxLat << '\n';
		hdrFile << "xdim " << (pMaxLon-pMinLon)/(pSizeX-1) << '\n';
		hdrFile << "ydim " << (pMaxLat-pMinLat)/(pSizeY-1) << '\n';
		if( pIsNoDataValueDefined == true)
		{
			hdrFile << std::defaultfloat;
			hdrFile << "nodata_value " << pNoDataValue << '\n';
		}
	}
	else
		success = false;
	hdrFile.close();

	fs::path prjPathname = pathname;
	prjPathname.replace_extension("prj");
	std::ofstream prjFile;
	prjFile.open(prjPathname, std::ios::out | std::ios::trunc);
	if(prjFile)
		prjFile << "GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.017453292519943295]]";
	else
		success = false;
	prjFile.close();

	return success;
}
