/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include "ContourFillsEngine.h"
#include <vector>
#include <string>

class KmlFileWriter
{
public:
	KmlFileWriter(void);
	~KmlFileWriter(void);

	bool Open(const char* pathname);
	void Close();

	void SetLineOpacity(double opacity_percent);
	void SetFillOpacity(double opacity_percent);

	bool WritePolyPolygon(const ContourFillsEngine::PolyPolygon& polyPolygon, const char* dataUnit);

private:
	struct KmlPolygon
	{
		const ContourFillsEngine::LinearRing* m_outerBoundaryRing;
		std::vector<const ContourFillsEngine::LinearRing*> m_innerBoudaryRings;
	};

	std::string pGetFilename(std::string pathname);
	void pWriteLine(const char* textLine);
	void pWriteStartTagLine(const char* textLine);
	void pWriteEndTagLine(const char* textLine);
	void pIncrementIndentation();
	void pDecrementIndentation();
	unsigned int pRgbToBgr(unsigned int rgbColor);
	void pToKmlPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<KmlPolygon>& kmlPolygons);
	void pToKmlPolygon(const std::vector<const ContourFillsEngine::LinearRing*>& rings, KmlPolygon& kmlPolygon);
	double pRingWidth(const ContourFillsEngine::LinearRing* ring);

	FILE* pFile;
	int pIndentation;
	uint8_t pLineAlpha;
	uint8_t pFillAlpha;
};

