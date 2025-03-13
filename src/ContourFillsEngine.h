#pragma once
#include "GeoDataGrid.h"
#include <vector>
#include <set>
#include <list>


class ContourFillsEngine
{
public:
	ContourFillsEngine(void);
	virtual ~ContourFillsEngine(void);

	struct FillZone
	{
		double m_fromValue;
		double m_toValue;
		unsigned int m_color;
	};

	struct LinearRing
	{
		std::vector<Position> m_nodes;
		int m_polygonId; // rings with the same polygonId within the same PolyPolygon
		                 // are part of the same polygon (i.e. polygon = 1 outer ring + inner ring(s))
	};

	struct PolyPolygon
	{
		std::vector<LinearRing> m_rings;
		double m_fromValue;
		double m_toValue;
		unsigned int m_color;
	};

	std::vector<PolyPolygon> GeneratePolyPolygons(const GeoDataGrid<float>& grid, const std::vector<FillZone>& fillZoneList);
	static bool ExportToMifFile(const char* pathname, const char* dataUnit, const std::vector<PolyPolygon>& polyPolygons);
	static bool ExportToKmlFile(const char* pathname, double fillOpacity_percent, double lineOpacity_percent, const char* dataUnit, const std::vector<PolyPolygon>& polyPolygons);

protected:

	struct Segment
	{
		int m_nodeId0;
		int m_nodeId1;
		int m_tesseraId; // tessera from which the segment originated from
	};
	struct SegmentComparator
	{
		bool operator()(const Segment& seg1, const Segment& seg2) const
		{
			int minNodeIdSeg1 = std::min(seg1.m_nodeId0, seg1.m_nodeId1);
			int maxNodeIdSeg1 = std::max(seg1.m_nodeId0, seg1.m_nodeId1);
			int minNodeIdSeg2 = std::min(seg2.m_nodeId0, seg2.m_nodeId1);
			int maxNodeIdSeg2 = std::max(seg2.m_nodeId0, seg2.m_nodeId1);
			if( minNodeIdSeg1 == minNodeIdSeg2 )
				return maxNodeIdSeg1 < maxNodeIdSeg2;
			else
				return minNodeIdSeg1 < minNodeIdSeg2;
		}
	};

	void pGetPolyPolygon(const FillZone& fillZone, const GeoDataGrid<float>& grid, PolyPolygon& polyPolygon);
	void pInitNodePositions(const GeoDataGrid<float>& grid);
	void pGetTesseraNodeIds(int tesseraId, int& llcNodeId, int& ulcNodeId, int& urcNodeId, int& lrcNodeId);
	Position* pGetNodePos(int nodeId);
	void pProcess(double fillZoneMin, double fillZoneMax, const GeoDataGrid<float>& grid, PolyPolygon& polyPolygon);
	void pProcessTessera(int tesseraId, double fillZoneMin, double fillZoneMax, const GeoDataGrid<float>& grid);
	void pAddSegment(int nodeId0, int nodeId1, int tesseraId);
	void pGroupTesserae(int tesseraId0, int tesseraId1);
	void pSetGroupId(int tesseraId, int groupId);
	int pGetGroupId(int tesseraId);
	void pProcessSegmentsAndGroups(PolyPolygon& polyPolygon);

	static const int pUNDEFINED_ID;
	int pNumNodesX;
	int pNumNodesY;
	std::vector<Position> pPosVector;
	std::set<Segment, SegmentComparator> pSegmentSet;
	std::vector<std::list<int>> pGroups;
	std::vector<int> pTesseraGroupIds;
};
