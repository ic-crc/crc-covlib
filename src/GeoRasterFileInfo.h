#pragma once
#include <string>
#include <cstdint>
#include <vector>


class GeoRasterFileInfo
{
public:
	GeoRasterFileInfo();
	virtual ~GeoRasterFileInfo();

	enum CoordinateSystem
	{
		GEOGRAPHIC = 1,
		UTM = 2,
		EPSG_3979 = 3
	};

	void Clear();
	
	double ResolutionInMeters();

	bool Wgs84ToNativeCoords(double lat_wgs84, double lon_wgs84, double* nativeXCoord, double* nativeYCoord);
	void NativeToWgs84Coords(double nativeXCoord, double nativeYCoord, double* lat_wgs84, double* lon_wgs84);

	bool IsIn(double lat_wgs84, double lon_wgs84);
	bool IsBoundingBoxIntersectingWith(double minLat_wgs84, double minLon_wgs84, double maxLat_wgs84, double maxLon_wgs84);
	void GetWgs84BoundingBox(double* minLat_wgs84, double* minLon_wgs84, double* maxLat_wgs84, double* maxLon_wgs84);

	void GetPixelNativeCoord(uint32_t xIndex, uint32_t yIndex, double* xNativeCoord, double* yNativeCoord);
	void GetPixelWgs84Coord(uint32_t xIndex, uint32_t yIndex, double* lat_wgs84, double* lon_wgs84);
	void GetPixelIndex(double lat_wgs84, double lon_wgs84, uint32_t* xIndex, uint32_t* yIndex);
	void GetPixelRealIndex(double lat_wgs84, double lon_wgs84, double* xRealIndex, double* yRealIndex);
	void GetSurroundingPixelIndexes(double lat_wgs84, double lon_wgs84, uint32_t* x1, uint32_t* x2, uint32_t* y1, uint32_t* y2, double* xRealIndex=nullptr, double* yRealIndex=nullptr);

	static void BilinearInterpl(double x1, double x2, double y1, double y2, double Q11, double Q12, double Q21, double Q22, double x, double y, double* result);

	uint8_t m_coordSystem;

	uint32_t m_rasterHeight; // in number of pixels
	uint32_t m_rasterWidth; // in number of pixels

	// Limits in degrees or meters depending on the coordinate system
	// (i.e. degrees for GEOGRAPHIC, meters otherwise).
	// Indicates area covered by the raster file
	double m_topLimit; 
	double m_bottomLimit;
	double m_leftLimit;
	double m_rightLimit;

	// In degrees or meters depending on the coordinate system
	// (i.e. degrees for GEOGRAPHIC, meters otherwise).
	double m_pixelHeight;
	double m_pixelWidth;

	// Only used if coordinate system is UTM
	int32_t m_zone;
	bool m_northp; // true means North, false means South

	bool m_applyDatumTransform;
	std::vector<double> m_toWgs84HelmertParams;

	std::string m_pathname;

private:
	bool pGeogToUTM(double lat, double lon, double* easting, double* northing);
	void pUTMtoGeog(double easting, double northing, double* lat, double* lon);

	void pToWgs84(double lat, double lon, double* lat_wgs84, double* lon_wgs84);
	void pFromWgs84(double lat_wgs84, double lon_wgs84, double* lat, double* lon);

	void pHelmert7ParamsForward(double x, double y, double z, double dx, double dy, double dz,
	                            double rx, double ry, double rz, double s,
	                            double* x_out, double* y_out, double* z_out);
	void pHelmert7ParamsReverse(double x, double y, double z, double dx, double dy, double dz,
	                            double rx, double ry, double rz, double s,
	                            double* x_out, double* y_out, double* z_out);
};