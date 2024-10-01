#ifdef _MSC_VER
	#define _CRT_SECURE_NO_DEPRECATE
#endif
#include "KmlFileWriter.h"
#include <algorithm>

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
	if( pFile == NULL )
		return;

	for(int i=0 ; i<pIndentation ; i++)
		fprintf(pFile, INDENT_STR);
	fprintf(pFile, "%s", textLine);
	fprintf(pFile, "\n");
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

// Converts a ContourFillEngine::PolyPolygon object into a vector of BoxedPolygon objects.
// polyPolygon is not modified
void KmlFileWriter::pGetBoxedPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<BoxedPolygon>& boxedPolygons)
{
	boxedPolygons.clear();
	for(size_t i=0 ; i<polyPolygon.m_rings.size() ; i++)
	{
		BoxedPolygon boxedPolygon;
		boxedPolygon.m_maxLat = boxedPolygon.m_maxLon = -10000;
		boxedPolygon.m_minLat = boxedPolygon.m_minLon = 10000;
		boxedPolygon.m_nodes = polyPolygon.m_rings[i].m_nodes;
		for(size_t j=0 ; j<boxedPolygon.m_nodes.size() ; j++)
		{
			Position pos;
			pos.m_lat = boxedPolygon.m_nodes[j].m_lat;
			pos.m_lon = boxedPolygon.m_nodes[j].m_lon;
			if( pos.m_lat > boxedPolygon.m_maxLat ) boxedPolygon.m_maxLat = pos.m_lat;
			if( pos.m_lon > boxedPolygon.m_maxLon ) boxedPolygon.m_maxLon = pos.m_lon;
			if( pos.m_lat < boxedPolygon.m_minLat ) boxedPolygon.m_minLat = pos.m_lat;
			if( pos.m_lon < boxedPolygon.m_minLon ) boxedPolygon.m_minLon = pos.m_lon;
		}
		boxedPolygons.push_back(boxedPolygon);
	}
}

// Polygons from sourcePolygons that are not contained by any other Polygon in sourcePolygons are
// copied into outerPolygons. Other Polygons (i.e. those who are contained) are copied into innerPolygons.
// sourcePolygons is not modified.
void KmlFileWriter::pClassifyPolygons(std::vector<BoxedPolygon>& sourcePolygons, std::vector<BoxedPolygon>& outerPolygons, std::vector<BoxedPolygon>& innerPolygons)
{
	outerPolygons.clear();
	innerPolygons.clear();
	for(size_t i=0 ; i<sourcePolygons.size() ; i++)
	{
		BoxedPolygon curBoxedPolygon = sourcePolygons[i];
		bool isInner = false;
		for(size_t j=0 ; j<sourcePolygons.size() ; j++)
		{
			if( i != j )
			{
				if( pIsAInsideB(curBoxedPolygon, sourcePolygons[j]) )
				{
					isInner = true;
					break;
				}
			}
		}
		if( isInner == false )
			outerPolygons.push_back(curBoxedPolygon);
		else
			innerPolygons.push_back(curBoxedPolygon);
	}
}

// Return true if polygon A is inside polygon B, false otherwise.
// Assumes polygon A is either entirely inside B or entirely outside B.
bool KmlFileWriter::pIsAInsideB(BoxedPolygon& A, BoxedPolygon& B)
{
	if( !(A.m_minLat >= B.m_minLat && A.m_maxLat <= B.m_maxLat && A.m_minLon >= B.m_minLon && A.m_maxLon <= B.m_maxLon) )
		return false;

	// Normally we should only encounter polygons that are either fully inside or fully outside
	// of one another, so we only need to check for 1 point.
	return pIsAInsideB(A.m_nodes[0].m_lat, A.m_nodes[0].m_lon, B);

	// This should not happen with PolyPolygons from ContourFillsEngine, but if we wanted to have 
	// additional robustness against edge cases where a polygon's node would fall exactly on the
	// edge of the other (the return value of pIsAInsideB() below is then unpredictable), we could
	// do the check using a few more points (up to five here) instead of just one.
	/*
	int inCount = 0, outCount = 0;
	size_t maxNumPointsCheck = 5;
	for(size_t i=0 ; i<A.m_nodes.size() && i<maxNumPointsCheck ; i++)
	{
		if( pIsAInsideB(A.m_nodes[i].m_lat, A.m_nodes[i].m_lon, B) == true )
			inCount++;
		else
			outCount++;
	}

	return inCount > outCount;
	*/
}

// Returns true when point at coords A_lat, A_lon is located inside the specifed polygon,
// false otherwise. Note that method may return either true or false when the point falls
// right on the edge of the polygon.
// References: http://alienryderflex.com/polygon/
bool KmlFileWriter::pIsAInsideB(double A_lat, double A_lon, BoxedPolygon& B)
{
// -1 because the algorithm for this method assumes no node repeats while PolyPolygons from
// ContourFillsEngine creates them with repeated nodes (i.e. first node's position = last node's position)
size_t polyCorners = B.m_nodes.size() -1;
size_t j = polyCorners-1;
bool oddNodes = false;
double x = A_lon;
double y = A_lat;

	for(size_t i=0 ; i<polyCorners ; i++)
	{
	double polyYi = B.m_nodes[i].m_lat;
	double polyYj = B.m_nodes[j].m_lat;
	double polyXi = B.m_nodes[i].m_lon;
	double polyXj = B.m_nodes[j].m_lon;

		if( (polyYi<y && polyYj>=y) || (polyYj<y && polyYi>=y) )
		{
			if(polyXi+(y-polyYi)/(polyYj-polyYi)*(polyXj-polyXi)<x)
				oddNodes=!oddNodes;
		}
		j=i;
	}

	return oddNodes;
}

// Converts a ContourFillEngine::PolyPolygon into a vector of KmlPolygon.
// polyPolygon is not modified.
void KmlFileWriter::pToKmlPolygons(const ContourFillsEngine::PolyPolygon& polyPolygon, std::vector<KmlPolygon>& kmlPolygons)
{
std::vector<BoxedPolygon> sourcePolygons, outerPolygons, innerPolygons;
int roundCount = 0;

	kmlPolygons.clear();
	pGetBoxedPolygons(polyPolygon, sourcePolygons);
	while( sourcePolygons.size() > 0 )
	{
		pClassifyPolygons(sourcePolygons, outerPolygons, innerPolygons);
		for(size_t i=0 ; i<outerPolygons.size() ; i++)
		{
			if( roundCount %2 == 0 )
			{
				KmlPolygon kmlPolygon;
				kmlPolygon.m_outerBoundaryRing = outerPolygons[i];
				kmlPolygons.push_back(kmlPolygon);
			}
			else
			{
				size_t size = kmlPolygons.size();
				size_t kmlPolyIndex;
				for(size_t j=0 ; j<size ; j++)
				{
					kmlPolyIndex = size-1-j; // use reverse order so to add inner polygon to one of those
					                         // added to kmlPolygons in the previous round
					if( pIsAInsideB(outerPolygons[i], kmlPolygons[kmlPolyIndex].m_outerBoundaryRing) == true )
					{
						kmlPolygons[kmlPolyIndex].m_innerBoudaryRings.push_back(outerPolygons[i]);
						break;
					}
				}
			}
		}
		sourcePolygons = innerPolygons;
		roundCount++;

		// safeguard against infinite loop
		if( roundCount > 12 )
			break;
	}
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
KmlPolygon* polygon;

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

		for(size_t j=0 ; j<polygon->m_outerBoundaryRing.m_nodes.size() ; j++)
		{
			snprintf(str, BUFFER_SIZE, "%.8lf,%.8lf", polygon->m_outerBoundaryRing.m_nodes[j].m_lon,
			                                          polygon->m_outerBoundaryRing.m_nodes[j].m_lat);
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
			for(size_t k=0 ; k<polygon->m_innerBoudaryRings[j].m_nodes.size() ; k++)
			{
				snprintf(str, BUFFER_SIZE, "%.8lf,%.8lf", polygon->m_innerBoudaryRings[j].m_nodes[k].m_lon,
				                                          polygon->m_innerBoudaryRings[j].m_nodes[k].m_lat);
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
