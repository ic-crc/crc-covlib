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
	struct BoxedPolygon
	{
		std::vector<Position> m_nodes;
		double m_minLat;
		double m_maxLat;
		double m_minLon;
		double m_maxLon;
	};

	struct KmlPolygon
	{
		BoxedPolygon m_outerBoundaryRing;
		std::vector<BoxedPolygon> m_innerBoudaryRings;
	};

	void pToKmlPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<KmlPolygon>& kmlPolygons);
	void pGetBoxedPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<BoxedPolygon>& boxedPolygons);
	void pClassifyPolygons(std::vector<BoxedPolygon>& sourcePolygons, std::vector<BoxedPolygon>& outerPolygons, std::vector<BoxedPolygon>& innerPolygons);
	bool pIsAInsideB(BoxedPolygon& A, BoxedPolygon& B);
	bool pIsAInsideB(double A_lat, double A_lon, BoxedPolygon& B);
	std::string pGetFilename(std::string pathname);
	void pWriteLine(const char* textLine);
	void pWriteStartTagLine(const char* textLine);
	void pWriteEndTagLine(const char* textLine);
	void pIncrementIndentation();
	void pDecrementIndentation();
	unsigned int pRgbToBgr(unsigned int rgbColor);

	FILE* pFile;
	int pIndentation;
	uint8_t pLineAlpha;
	uint8_t pFillAlpha;
};

