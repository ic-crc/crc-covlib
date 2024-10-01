#include "ContourFillsEngine.h"
#include "KmlFileWriter.h"
#include <map>
#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif


const int ContourFillsEngine::pUNDEFINED_ID = -1;

ContourFillsEngine::ContourFillsEngine(void)
{

}

ContourFillsEngine::~ContourFillsEngine(void)
{
}

std::vector<ContourFillsEngine::PolyPolygon> ContourFillsEngine::GeneratePolyPolygons(const GeoDataGrid<float>& grid, const std::vector<FillZone>& fillZoneList)
{
std::vector<PolyPolygon> result;

	pInitNodePositions(grid);

	for(size_t i=0 ; i<fillZoneList.size() ; i++)
	{
	PolyPolygon polyPolygon;

		pGetPolyPolygon(fillZoneList[i], grid, polyPolygon);
		result.push_back(polyPolygon);

		pSegmentSet.clear();
		pGroups.clear();
		pTesseraGroupIds.clear();
	}

	pPosVector.clear();

	return result;
}

void ContourFillsEngine::pInitNodePositions(const GeoDataGrid<float>& grid)
{
pNumNodesX = std::max((int) grid.SizeX(), 512); // ensure minimum resolution (power of 2 is not required)
pNumNodesY = std::max((int) grid.SizeY(), 512);
double minLat, maxLat, minLon, maxLon;
double deltaLat, deltaLon;
Position nodePos; 
int nodeId = 0;

	grid.GetBordersCoordinates(&minLat, &minLon, &maxLat, &maxLon);
	deltaLat = (maxLat-minLat)/(pNumNodesY-1);
	deltaLon = (maxLon-minLon)/(pNumNodesX-1);

	pPosVector.resize((size_t)(pNumNodesX*pNumNodesY));
	for(int x=0 ; x<pNumNodesX ; x++)
	{
		nodePos.m_lon = minLon + (x*deltaLon);
		for(int y=0 ; y<pNumNodesY ; y++)
		{
			nodePos.m_lat = minLat + (y*deltaLat);
			pPosVector[(size_t)nodeId] = nodePos;
			nodeId++;
		}
	}
}

// llc = lower left corner, ulc = upper left corner, urc = upper right corner, lrc = lower left corner 
void ContourFillsEngine::pGetTesseraNodeIds(int tesseraId, int& llcNodeId, int& ulcNodeId, int& urcNodeId, int& lrcNodeId)
{
int x = tesseraId / (pNumNodesY-1);
int y = tesseraId % (pNumNodesY-1);

	llcNodeId = (x*pNumNodesY)+y;
	ulcNodeId = llcNodeId+1; // (x*pNumNodesY)+(y+1);
	lrcNodeId = ((x+1)*pNumNodesY)+y;
	urcNodeId = lrcNodeId+1; // ((x+1)*pNumNodesY)+(y+1);
}

Position* ContourFillsEngine::pGetNodePos(int nodeId)
{
	return &(pPosVector[(size_t)nodeId]);
}

void ContourFillsEngine::pGetPolyPolygon(const FillZone& fillZone, const GeoDataGrid<float>& grid, PolyPolygon& polyPolygon)
{
double fillMin, fillMax;

	polyPolygon.m_rings.clear();
	polyPolygon.m_fromValue = fillZone.m_fromValue;
	polyPolygon.m_toValue = fillZone.m_toValue;
	polyPolygon.m_color = fillZone.m_color;

	fillMin = std::min(fillZone.m_fromValue, fillZone.m_toValue);
	fillMax = std::max(fillZone.m_fromValue, fillZone.m_toValue);
	pProcess(fillMin, fillMax, grid, polyPolygon);
}

void ContourFillsEngine::pProcess(double fillZoneMin, double fillZoneMax, const GeoDataGrid<float>& grid, PolyPolygon& polyPolygon)
{
int numTesseraX = pNumNodesX-1;
int numTesseraY = pNumNodesY-1;
int tesseraId = 0;

	pTesseraGroupIds.resize((size_t)(numTesseraX*numTesseraY), pUNDEFINED_ID);

	for(int x=0 ; x<numTesseraX ; x++)
	{
		for(int y=0 ; y<numTesseraY ; y++)
		{
			pProcessTessera(tesseraId, fillZoneMin, fillZoneMax, grid);
			tesseraId++;
		}
	}

	pProcessSegmentsAndGroups(polyPolygon);
}

void ContourFillsEngine::pProcessTessera(int tesseraId, double fillZoneMin, double fillZoneMax, const GeoDataGrid<float>& grid)
{
int llcNodeId, ulcNodeId, urcNodeId, lrcNodeId;
double centerLat, centerLon, centerVal;
float val;

	pGetTesseraNodeIds(tesseraId, llcNodeId, ulcNodeId, urcNodeId, lrcNodeId);
	centerLat = ((pGetNodePos(llcNodeId)->m_lat + pGetNodePos(ulcNodeId)->m_lat)) / 2.0;
	centerLon = ((pGetNodePos(llcNodeId)->m_lon + pGetNodePos(lrcNodeId)->m_lon)) / 2.0;
	grid.GetInterplData(centerLat, centerLon, &val);
	centerVal = (double) val;
	if( centerVal >= fillZoneMin &&  centerVal <= fillZoneMax )
	{
		pAddSegment(llcNodeId, ulcNodeId, tesseraId);
		pAddSegment(ulcNodeId, urcNodeId, tesseraId);
		pAddSegment(urcNodeId, lrcNodeId, tesseraId);
		pAddSegment(lrcNodeId, llcNodeId, tesseraId);
	}
}

void ContourFillsEngine::pAddSegment(int nodeId0, int nodeId1, int tesseraId)
{
Segment seg;
std::set<Segment, SegmentComparator>::iterator iter;

	seg.m_nodeId0 = nodeId0;
	seg.m_nodeId1 = nodeId1;
	seg.m_tesseraId = tesseraId;

	iter = pSegmentSet.find(seg);
	if (iter != pSegmentSet.end())
	{
		// Same line segment already present, this means it is inside the polygon (i.e. not part of its outer or inner border(s)).
		// The segment won't have to be "drawn", it is therefore removed.
		// This also means that the two tesserae from which the segments are originating from are part of the same polygon.
		pGroupTesserae(iter->m_tesseraId, tesseraId);
		pSegmentSet.erase(iter);
	}
	else
		pSegmentSet.insert(seg);
}

// Use this method to specify that tesseraId0 and tesseraId1 belong to the same polygon.
// This will update pGroups and pTesseraGroupIds accordingly
//
//   std::vector<std::list<int>> pGroups:
//     - Each vector item represents a different group (or polygon). The groupId corresponds
//       to the vector's index at which the group is stored.
//     - Each list object contains the list of tesseraIds compositing the polygon.
//   pGroups is utilized for quickly udpating pTesseraGroupIds, which stores to groupId for
//   each tesseraId.
void ContourFillsEngine::pGroupTesserae(int tesseraId0, int tesseraId1)
{
int tess0GroupId = pGetGroupId(tesseraId0);
int tess1GroupId = pGetGroupId(tesseraId1);

	if( tess0GroupId == pUNDEFINED_ID && tess1GroupId == pUNDEFINED_ID  )
	{ // tesseraId0 and tesseraId1 were not found in any group, add a new group (i.e. polygon) for them
		std::list<int> newGroup;
		int groupId = newGroup.size(); // groupId is the vector index for pGroups
		newGroup.push_back(tesseraId0);
		newGroup.push_back(tesseraId1);
		pGroups.push_back(newGroup);
		pSetGroupId(tesseraId0, groupId);
		pSetGroupId(tesseraId1, groupId);
	}
	else if( tess0GroupId != pUNDEFINED_ID && tess1GroupId == pUNDEFINED_ID )
	{
		pGroups[(size_t)tess0GroupId].push_back(tesseraId1);
		pSetGroupId(tesseraId1, tess0GroupId);
	}
	else if( tess0GroupId == pUNDEFINED_ID && tess1GroupId != pUNDEFINED_ID )
	{
		pGroups[(size_t)tess1GroupId].push_back(tesseraId0);
		pSetGroupId(tesseraId0, tess1GroupId);
	}
	else
	{
		if( tess0GroupId != tess1GroupId )
		{ // tesseraId0 and tesseraId1 were found in two different groups, merge the two groups
		  // together since tesseraId0 and tesseraId1 belong to the same group (polygon).
			std::list<int>* group0 = &(pGroups[(size_t)tess0GroupId]);
			std::list<int>* group1 = &(pGroups[(size_t)tess1GroupId]);
			std::list<int>::iterator iter;

			// merge the smaller group into the bigger one			
			if( group0->size() > group1->size() )
			{
				for(iter = group1->begin() ; iter != group1->end() ; ++iter)
					pSetGroupId(*iter, tess0GroupId);
				group0->splice(group0->end(), *group1);
			}
			else
			{
				for(iter = group0->begin() ; iter != group0->end() ; ++iter)
					pSetGroupId(*iter, tess1GroupId);
				group1->splice(group1->end(), *group0);
			}
			// note: do not delete the smaller group (i.e. the vector item) so the groupIds (indexes) stay valid
		}
		// else, they were found in same group, nothing to do
	}
}

void ContourFillsEngine::pSetGroupId(int tesseraId, int groupId)
{
	pTesseraGroupIds[(size_t)tesseraId] = groupId;
}

int ContourFillsEngine::pGetGroupId(int tesseraId)
{
	return pTesseraGroupIds[(size_t)tesseraId];
}

// Use data from pSegmentSet and pTesseraGroupIds to generate the polyPolygon's rings
void ContourFillsEngine::pProcessSegmentsAndGroups(PolyPolygon& polyPolygon)
{
	struct NodePolyPair
	{
		int m_nodeId;
		int m_polygonId;
	};
	struct NodePolyPairComparator
	{
		bool operator()(const NodePolyPair& pair1, const NodePolyPair& pair2) const
		{
			if( pair1.m_nodeId == pair2.m_nodeId )
				return pair1.m_polygonId < pair2.m_polygonId;
			else
				return pair1.m_nodeId < pair2.m_nodeId;
		}
	};
	// segmentMultimap will allow to quickly search for segments when having only one nodeId at hand.
	std::multimap<NodePolyPair, int , NodePolyPairComparator> segmentMultimap;
	typedef std::multimap<NodePolyPair, int, NodePolyPairComparator>::iterator MMapIterator;

	// populate segmentMultimap from segmentSet
	std::set<Segment, SegmentComparator>::iterator iter;
	for(iter = pSegmentSet.begin() ; iter != pSegmentSet.end() ; ++iter)
	{
	Segment seg = *iter;
	NodePolyPair npp;

		npp.m_polygonId = pGetGroupId(seg.m_tesseraId);

		// add two entries to be able to search for a segment by any of its nodeId
		npp.m_nodeId = seg.m_nodeId0;
		segmentMultimap.insert(std::pair<NodePolyPair, int>(npp, seg.m_nodeId1));
		npp.m_nodeId = seg.m_nodeId1;
		segmentMultimap.insert(std::pair<NodePolyPair, int>(npp, seg.m_nodeId0));
	}

	auto GetNextNodeId = [&segmentMultimap] (int fromNodeId, int polygonId) -> int
	{
		NodePolyPair npp;
		npp.m_nodeId = fromNodeId;
		npp.m_polygonId = polygonId;
		std::pair<MMapIterator, MMapIterator> result = segmentMultimap.equal_range(npp);
		for (MMapIterator it = result.first ; it != result.second ; ++it)
			return it->second;
		return pUNDEFINED_ID;
	};

	// call to delete both entries in segmentMultimap for the specified segment data
	auto DeleteMultimapEntries = [&segmentMultimap] (int nodeId0, int nodeId1, int polygonId) -> void
	{
		NodePolyPair npp;
		npp.m_nodeId = nodeId0;
		npp.m_polygonId = polygonId;
		std::pair<MMapIterator, MMapIterator> result = segmentMultimap.equal_range(npp);
		for (MMapIterator it = result.first ; it != result.second ; ++it)
		{
			if( it->second == nodeId1 )
			{
				segmentMultimap.erase(it);
				break;
			}
		}
		npp.m_nodeId = nodeId1;
		result = segmentMultimap.equal_range(npp);
		for (MMapIterator it = result.first ; it != result.second ; ++it)
		{
			if( it->second == nodeId0 )
			{
				segmentMultimap.erase(it);
				break;
			}
		}
	};

	auto AddRingNode = [this] (LinearRing& ring, std::vector<int>& nodeIds, int nodeId) -> void
	{
		ring.m_nodes.push_back(*(this->pGetNodePos(nodeId)));
		nodeIds.push_back(nodeId);
	};

	// call to delete entry in pSegmentSet for the specified segment data
	auto DeleteSegmentSetEntry = [this] (int nodeId0, int nodeId1) -> void
	{
		Segment seg;
		seg.m_nodeId0 = nodeId0;
		seg.m_nodeId1 = nodeId1;
		this->pSegmentSet.erase(seg);
	};

	while( pSegmentSet.size() > 0 )
	{
	std::set<Segment, SegmentComparator>::iterator segIter;
	LinearRing newRing;
	std::vector<int> newRingNodeIds;
	int polygonId;
	size_t numNodes;

		// take one segment to start generating a polygon's ring
		segIter = pSegmentSet.begin();

		AddRingNode(newRing, newRingNodeIds, segIter->m_nodeId0);
		AddRingNode(newRing, newRingNodeIds, segIter->m_nodeId1);
		polygonId = pGetGroupId(segIter->m_tesseraId);

		DeleteMultimapEntries(segIter->m_nodeId0, segIter->m_nodeId1, polygonId);
		pSegmentSet.erase(segIter);

		while( newRingNodeIds[0] != newRingNodeIds.back() )
		{
			AddRingNode(newRing, newRingNodeIds, GetNextNodeId(newRingNodeIds.back(), polygonId));

			numNodes = newRingNodeIds.size();
			DeleteSegmentSetEntry(newRingNodeIds[numNodes-2], newRingNodeIds[numNodes-1]);
			DeleteMultimapEntries(newRingNodeIds[numNodes-2], newRingNodeIds[numNodes-1], polygonId);
		}

		// Delete last node to avoid duplication of first and last nodes (...commented out: some software
		// require this first/last node duplication)
		//newRing.m_nodes.erase(newRing.m_nodes.end()-1);

		polyPolygon.m_rings.push_back(newRing);
	}
}

bool ContourFillsEngine::ExportToMifFile(const char* pathname, const char* dataUnit, const std::vector<PolyPolygon>& polyPolygons)
{
FILE* mifFile;
FILE* midFile;
std::string mifPathname = fs::path(pathname).replace_extension(".mif").string();
std::string midPathname = fs::path(pathname).replace_extension(".mid").string();
bool midFileCreated = false;
bool success = false;

	mifFile = fopen(mifPathname.c_str(), "wt");
	if(mifFile != NULL)
	{
		midFile = fopen(midPathname.c_str(), "wt");
		if(midFile != NULL)
			midFileCreated = true;

		fprintf(mifFile, "Version 300\n");
		fprintf(mifFile, "Charset \"WindowsLatin1\"\n");
		fprintf(mifFile, "Delimiter \";\"\n");
		fprintf(mifFile, "CoordSys Earth Projection 1, 0\n");
		fprintf(mifFile, "Columns 1\n");
		fprintf(mifFile, "  VALUE char (32)\n");
		fprintf(mifFile, "Data\n\n");

		for(unsigned int i=0 ; i<polyPolygons.size() ; i++)
		{
			fprintf(mifFile, "Region %zu\n", polyPolygons[i].m_rings.size());
			for(unsigned int j=0 ; j<polyPolygons[i].m_rings.size() ; j++)
			{
				fprintf(mifFile, "%zu\n", polyPolygons[i].m_rings[j].m_nodes.size());
				for(unsigned int k=0 ; k<polyPolygons[i].m_rings[j].m_nodes.size() ; k++)
					fprintf(mifFile, "%.8lf %.8lf\n", polyPolygons[i].m_rings[j].m_nodes[k].m_lon,
					                                  polyPolygons[i].m_rings[j].m_nodes[k].m_lat);
			}
			fprintf(mifFile, "PEN(1, 1, %u)\n", polyPolygons[i].m_color);
			fprintf(mifFile, "BRUSH(2, %u)\n", polyPolygons[i].m_color);
			fprintf(mifFile, "\n");

			if( midFileCreated == true )
				fprintf(midFile, "%.2lf to %.2lf %s\n", polyPolygons[i].m_fromValue, polyPolygons[i].m_toValue, dataUnit);
		}

		fclose(mifFile);

		if( midFileCreated == true )
		{
			fclose(midFile);
			success = true;
		}
	}

	return success;
}

bool ContourFillsEngine::ExportToKmlFile(const char* pathname, double fillOpacity_percent, double lineOpacity_percent, 
                                         const char* dataUnit, const std::vector<PolyPolygon>& polyPolygons)
{
KmlFileWriter kmlFileWriter;
std::string kmlPathname = fs::path(pathname).replace_extension(".kml").string();
bool success = false;

	kmlFileWriter.SetFillOpacity(fillOpacity_percent);
	kmlFileWriter.SetLineOpacity(lineOpacity_percent);
	if( kmlFileWriter.Open(kmlPathname.c_str()) )
	{
		for(size_t i=0 ; i<polyPolygons.size() ; i++)
			kmlFileWriter.WritePolyPolygon(polyPolygons[i], dataUnit);

		kmlFileWriter.Close();
		success = true;
	}

	return success;
}
