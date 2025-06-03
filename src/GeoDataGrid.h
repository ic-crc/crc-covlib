/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include <string>
#include <cstdint>
#include "Position.h"


template <typename T>
class GeoDataGrid
{
public:
	GeoDataGrid();
	GeoDataGrid(unsigned int sizeX, unsigned int sizeY);
	GeoDataGrid(const GeoDataGrid& original);
	virtual ~GeoDataGrid(void);

	const GeoDataGrid& operator=(const GeoDataGrid& original);

	unsigned int SizeX() const;
	unsigned int SizeY() const;
	bool Clear(unsigned int newSizeX, unsigned int newSizeY);
	void DefineNoDataValue(T noDataValue);
	bool IsNoDataValueDefined() const;
	T GetNoDataValue() const;
	void UndefineNoDataValue();
	void SetBordersCoordinates(double minLat, double minLon, double maxLat, double maxLon);
	void GetBordersCoordinates(double* minLat, double* minLon, double* maxLat, double* maxLon) const;
	double ResolutionInDegrees() const;
	Position GetPos(unsigned int x, unsigned int y) const;
	void SetData(unsigned int x, unsigned int y, T data);
	bool SetData(unsigned int sizeX, unsigned int sizeY, const T* data);
	T GetData(unsigned int x, unsigned int y) const;
	T* GetDataPtr() const;
	bool GetInterplData(double lat, double lon, T* data) const;
	bool GetClosestData(double lat, double lon, T* data) const;
	void SetDataUnit(const char* unit);
	const char* GetDataUnit() const;
	void SetDataDescription(const char* unit);
	const char* GetDataDescription() const;
	bool ExportToTextFile(const char* pathname, const char* dataColName) const;
	bool ExportToBILFile(const char* pathname) const;

private:
	void pInit();

	unsigned int pSizeX;
	unsigned int pSizeY;
	T* pDataPtr;
	double pMinLat;
	double pMinLon;
	double pMaxLat;
	double pMaxLon;
	std::string pDataUnit;
	std::string pDataDescription;
	bool pIsNoDataValueDefined;
	T pNoDataValue;
};

#include "GeoDataGrid.tpp"