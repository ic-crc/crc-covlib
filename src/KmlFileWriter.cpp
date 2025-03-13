#ifdef _MSC_VER
	#define _CRT_SECURE_NO_DEPRECATE
#endif
#include "KmlFileWriter.h"
#include <algorithm>
#include <map>

#define INDENT_STR "  "

KmlFileWriter::KmlFileWriter(void)
{
	pFile = NULL;
	pIndentation = 0;
	pLineAlpha = 128;
	pFillAlpha = 128;
}

KmlFileWriter::~KmlFileWriter(void)
{
	Close();
}

bool KmlFileWriter::Open(const char* pathname)
{
	Close();
	pFile = fopen(pathname, "w");
	if(pFile != NULL)
	{
	std::string filename;

		pWriteLine("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
		pWriteLine("<kml xmlns=\"http://earth.google.com/kml/2.1\">");
		pWriteStartTagLine("<Document>");
		filename = "<name>" + pGetFilename(pathname) + "</name>";
		pWriteLine(filename.c_str());
		pWriteLine("<open>1</open>");

		return true;
	}
	else
		return false;
}

void KmlFileWriter::Close()
{
	if( pFile != NULL )
	{
		pWriteEndTagLine("</Document>");
		pWriteLine("</kml>");

		fclose(pFile);
		pFile = NULL;
	}
}

void KmlFileWriter::SetLineOpacity(double opacity_percent)
{
	pLineAlpha = 255*opacity_percent/100.0;
}

void KmlFileWriter::SetFillOpacity(double opacity_percent)
{
	pFillAlpha = 255*opacity_percent/100.0;
}

void KmlFileWriter::pWriteLine(const char* textLine)
{
	for(int i=0 ; i<pIndentation ; i++)
		fprintf(pFile, INDENT_STR);
	fprintf(pFile, "%s\n", textLine);
}

void KmlFileWriter::pWriteStartTagLine(const char* textLine)
{
	pWriteLine(textLine);
	pIncrementIndentation();
}

void KmlFileWriter::pWriteEndTagLine(const char* textLine)
{
	pDecrementIndentation();
	pWriteLine(textLine);
}

void KmlFileWriter::pIncrementIndentation()
{
	pIndentation++;
}

void KmlFileWriter::pDecrementIndentation()
{
	if( pIndentation != 0 )
		pIndentation--;
}

bool KmlFileWriter::WritePolyPolygon(const ContourFillsEngine::PolyPolygon& polyPolygon, const char* dataUnit)
{
const int BUFFER_SIZE = 128;
char str[BUFFER_SIZE];
int borderWidth = 1;
bool showBorder = true;
bool showFill = true;
unsigned int bgrColor = pRgbToBgr(polyPolygon.m_color);
std::vector<KmlPolygon> kmlPolygons;
const KmlPolygon* polygon;

	if( pFile == NULL )
		return false;

	pToKmlPolygons(polyPolygon, kmlPolygons);

	pWriteStartTagLine("<Placemark>");
	snprintf(str, BUFFER_SIZE, "<name>%.2lf to %.2lf %s</name>", polyPolygon.m_fromValue, polyPolygon.m_toValue, dataUnit);
	pWriteLine(str);
	pWriteStartTagLine("<Style>");
	pWriteStartTagLine("<LineStyle>");
	snprintf(str, BUFFER_SIZE, "<color>%02x%06x</color>", pLineAlpha, bgrColor);
	pWriteLine(str);
	snprintf(str, BUFFER_SIZE, "<width>%d</width>", borderWidth);
	pWriteLine(str);
	pWriteEndTagLine("</LineStyle>");
	pWriteStartTagLine("<PolyStyle>");
	snprintf(str, BUFFER_SIZE, "<color>%02x%06x</color>", pFillAlpha, bgrColor);
	pWriteLine(str);
	if( showBorder == false )
		pWriteLine("<outline>0</outline>");
	if( showFill == false )
		pWriteLine("<fill>0</fill>");
	pWriteEndTagLine("</PolyStyle>");
	pWriteEndTagLine("</Style>");
	pWriteStartTagLine("<MultiGeometry>");

	for(size_t i=0 ; i<kmlPolygons.size() ; i++)
	{
		polygon = &(kmlPolygons[i]);

		pWriteStartTagLine("<Polygon>");
		pWriteLine("<extrude>0</extrude>");
		pWriteLine("<altitudeMode>clampToGround</altitudeMode>");

		pWriteStartTagLine("<outerBoundaryIs>");
		pWriteStartTagLine("<LinearRing>");
		pWriteStartTagLine("<coordinates>");

		for(size_t j=0 ; j<polygon->m_outerBoundaryRing->m_nodes.size() ; j++)
		{
			snprintf(str, BUFFER_SIZE, "%.8lf,%.8lf", polygon->m_outerBoundaryRing->m_nodes[j].m_lon,
			                                          polygon->m_outerBoundaryRing->m_nodes[j].m_lat);
			pWriteLine(str);
		}
		pWriteEndTagLine("</coordinates>");
		pWriteEndTagLine("</LinearRing>");
		pWriteEndTagLine("</outerBoundaryIs>");

		for(size_t j=0 ; j<polygon->m_innerBoudaryRings.size() ; j++)
		{
			pWriteStartTagLine("<innerBoundaryIs>");
			pWriteStartTagLine("<LinearRing>");
			pWriteStartTagLine("<coordinates>");
			for(size_t k=0 ; k<polygon->m_innerBoudaryRings[j]->m_nodes.size() ; k++)
			{
				snprintf(str, BUFFER_SIZE, "%.8lf,%.8lf", polygon->m_innerBoudaryRings[j]->m_nodes[k].m_lon,
				                                          polygon->m_innerBoudaryRings[j]->m_nodes[k].m_lat);
				pWriteLine(str);
			}
			pWriteEndTagLine("</coordinates>");
			pWriteEndTagLine("</LinearRing>");
			pWriteEndTagLine("</innerBoundaryIs>");
		}

		pWriteEndTagLine("</Polygon>");
	}

	pWriteEndTagLine("</MultiGeometry>");
	pWriteEndTagLine("</Placemark>");

	return true;
}

// Example: pGetFilename("C:\\temp\\testfile.txt") returns "testfile.txt"
std::string KmlFileWriter::pGetFilename(std::string pathname)
{
std::string::size_type bsPos;
std::string ext = "";
std::string filename;

	if( pathname.size() > 0 )
	{
		std::replace(pathname.begin(), pathname.end(), '/', '\\');
		bsPos = pathname.rfind('\\', pathname.size()-1);
		if( bsPos == std::string::npos )
			filename = pathname;
		else
			filename = pathname.substr(bsPos+1, pathname.size()-bsPos-1);
	}

	return filename;
}

unsigned int KmlFileWriter::pRgbToBgr(unsigned int rgbColor)
{
	unsigned int red =   ( rgbColor & 0x000000FF );
	unsigned int green = ( rgbColor & 0x0000FF00 );
	unsigned int blue =  ( rgbColor & 0x00FF0000 ) >> 16;
	return blue + green + (red << 16);
}

// Converts a ContourFillEngine::PolyPolygon into a vector of KmlPolygon.
// polyPolygon is not modified.
void KmlFileWriter::pToKmlPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<KmlPolygon>& kmlPolygons)
{
std::map<int, std::vector<const ContourFillsEngine::LinearRing*>> polygons;
std::map<int, std::vector<const ContourFillsEngine::LinearRing*>>::iterator it;
const ContourFillsEngine::LinearRing* ringPtr;

	kmlPolygons.clear();

	for(size_t i=0 ; i<polyPolygon.m_rings.size() ; i++)
	{
		ringPtr = &(polyPolygon.m_rings[i]);
		polygons[ringPtr->m_polygonId].push_back(ringPtr);
	}
	
	for (it = polygons.begin() ; it != polygons.end() ; it++)
	{
		KmlPolygon kmlPolygon;
		if( it->second.size() == 1 )
			kmlPolygon.m_outerBoundaryRing = it->second[0];
		else
			pToKmlPolygon(it->second, kmlPolygon);
		kmlPolygons.push_back(kmlPolygon);
	}
}

void KmlFileWriter::pToKmlPolygon(const std::vector<const ContourFillsEngine::LinearRing*>& rings, KmlPolygon& kmlPolygon)
{
double maxWidth = std::numeric_limits<double>::lowest();
double width;

	kmlPolygon.m_outerBoundaryRing = nullptr;
	kmlPolygon.m_innerBoudaryRings.clear();

	for(size_t i=0 ; i<rings.size() ; i++)
	{
		width = pRingWidth(rings[i]);
		if( width > maxWidth )
		{
			maxWidth = width;
			kmlPolygon.m_outerBoundaryRing = rings[i];
		}
	}

	for(size_t i=0 ; i<rings.size() ; i++)
	{
		if( rings[i] != kmlPolygon.m_outerBoundaryRing )
			kmlPolygon.m_innerBoudaryRings.push_back(rings[i]);
	}
}

double KmlFileWriter::pRingWidth(const ContourFillsEngine::LinearRing* ring)
{
double minLon = std::numeric_limits<double>::max();
double maxLon = std::numeric_limits<double>::lowest();
double lon;

	for(size_t i=0 ; i<ring->m_nodes.size() ; i++)
	{
		lon = ring->m_nodes[i].m_lon;
		minLon = std::min(minLon, lon);
		maxLon = std::max(maxLon, lon);
	}
	return maxLon - minLon;
}
