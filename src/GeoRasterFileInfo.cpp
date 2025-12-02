/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include "GeoRasterFileInfo.h"
#include <GeographicLib/GeoCoords.hpp>
#include <GeographicLib/Geocentric.hpp>
#include "Projections.h"


GeoRasterFileInfo::GeoRasterFileInfo()
{
	Clear();
}

GeoRasterFileInfo::~GeoRasterFileInfo()
{

}

void GeoRasterFileInfo::Clear()
{
	m_coordSystem = GEOGRAPHIC;
	m_rasterHeight = 0;
	m_rasterWidth = 0;
	m_topLimit = 0;
	m_bottomLimit = 0;
	m_leftLimit = 0;
	m_rightLimit = 0;
	m_pixelHeight = 0;
	m_pixelWidth = 0;
	m_zone = 0;
	m_northp = true;
	m_applyDatumTransform = false;
	m_toWgs84HelmertParams.clear();
	m_pathname = "";
}

double GeoRasterFileInfo::ResolutionInMeters()
{
	if(m_coordSystem == GEOGRAPHIC)
	{
	double centerLat = (m_topLimit + m_bottomLimit) / 2.0;
	double horizResMeters = cos(centerLat*0.01745)*m_pixelWidth*111319.5;
	double vertResMeters = m_pixelHeight*111319.5;

		return std::max(horizResMeters, vertResMeters);
	}
	else
		return std::max(m_pixelHeight, m_pixelWidth);
}

bool GeoRasterFileInfo::Wgs84ToNativeCoords(double lat_wgs84, double lon_wgs84, double* nativeXCoord, double* nativeYCoord)
{
double lat, lon;

	pFromWgs84(lat_wgs84, lon_wgs84, &lat, &lon);

	if(m_coordSystem == GEOGRAPHIC)
	{
		*nativeXCoord = lon;
		*nativeYCoord = lat;
	}
	else if(m_coordSystem == UTM)
	{
		if( pGeogToUTM(lat, lon, nativeXCoord, nativeYCoord) == false )
			return false;
	}
	else if(m_coordSystem ==EPSG_3979)
	{
	const static LambertConicConformal2SP proj(49.0, 77.0, 49.0, -95.0, 0.0, 0.0, 6378137.0, 298.257222101);

		return proj.GeographicToProjected(lat, lon, nativeXCoord, nativeYCoord);
	}
	else
		return false;

	return true;
}

void GeoRasterFileInfo::NativeToWgs84Coords(double nativeXCoord, double nativeYCoord, double* lat_wgs84, double* lon_wgs84)
{
double lat, lon;

	if(m_coordSystem == GEOGRAPHIC)
	{
		lat = nativeYCoord;
		lon = nativeXCoord;
	}
	else if(m_coordSystem == UTM)
		pUTMtoGeog(nativeXCoord, nativeYCoord, &lat, &lon);
	else if(m_coordSystem ==EPSG_3979)
	{
	const static LambertConicConformal2SP proj(49.0, 77.0, 49.0, -95.0, 0.0, 0.0, 6378137.0, 298.257222101);

		proj.ProjectedToGeographic(nativeXCoord, nativeYCoord, &lat, &lon);
	}
	else
		return;

	pToWgs84(lat, lon, lat_wgs84, lon_wgs84);
}

bool GeoRasterFileInfo::IsIn(double lat_wgs84, double lon_wgs84)
{
double x, y;

	if( Wgs84ToNativeCoords(lat_wgs84, lon_wgs84, &x, &y) == false )
		return false;

	if( y >= m_topLimit )
		return false;
	if( y < m_bottomLimit )
		return false;
	if( x >= m_rightLimit )
		return false;
	if( x < m_leftLimit )
		return false;
	return true;
}

bool GeoRasterFileInfo::IsBoundingBoxIntersectingWith(double minLat_wgs84, double minLon_wgs84, double maxLat_wgs84, double maxLon_wgs84)
{
// check intersection with file's native box converted to lat/lon WGS84

double lat1_wgs84, lon1_wgs84, lat2_wgs84, lon2_wgs84, lat3_wgs84, lon3_wgs84, lat4_wgs84, lon4_wgs84;
double bottomLimit, topLimit, leftLimit, rightLimit;

	NativeToWgs84Coords(m_leftLimit, m_bottomLimit, &lat1_wgs84, &lon1_wgs84);
	NativeToWgs84Coords(m_leftLimit, m_topLimit, &lat2_wgs84, &lon2_wgs84);
	NativeToWgs84Coords(m_rightLimit, m_bottomLimit, &lat3_wgs84, &lon3_wgs84);
	NativeToWgs84Coords(m_rightLimit, m_topLimit, &lat4_wgs84, &lon4_wgs84);

	bottomLimit = std::min({lat1_wgs84, lat2_wgs84, lat3_wgs84, lat4_wgs84});
	topLimit = std::max({lat1_wgs84, lat2_wgs84, lat3_wgs84, lat4_wgs84});
	leftLimit = std::min({lon1_wgs84, lon2_wgs84, lon3_wgs84, lon4_wgs84});
	rightLimit = std::max({lon1_wgs84, lon2_wgs84, lon3_wgs84, lon4_wgs84});

	if( bottomLimit > maxLat_wgs84 )
		return false;
	if( topLimit < minLat_wgs84 )
		return false;
	if( leftLimit > maxLon_wgs84 )
		return false;
	if( rightLimit < minLon_wgs84 )
		return false;


// check intersection with function params' box converted to native

double xNative1, yNative1, xNative2, yNative2, xNative3, yNative3, xNative4, yNative4;
double xNativeMin, xNativeMax, yNativeMin, yNativeMax;

	Wgs84ToNativeCoords(minLat_wgs84, minLon_wgs84, &xNative1, &yNative1);
	Wgs84ToNativeCoords(minLat_wgs84, maxLon_wgs84, &xNative2, &yNative2);
	Wgs84ToNativeCoords(maxLat_wgs84, minLon_wgs84, &xNative3, &yNative3);
	Wgs84ToNativeCoords(maxLat_wgs84, maxLon_wgs84, &xNative4, &yNative4);

	xNativeMin = std::min({xNative1, xNative2, xNative3, xNative4});
	xNativeMax = std::max({xNative1, xNative2, xNative3, xNative4});
	yNativeMin = std::min({yNative1, yNative2, yNative3, yNative4});
	yNativeMax = std::max({yNative1, yNative2, yNative3, yNative4});

	if( m_bottomLimit > yNativeMax )
		return false;
	if( m_topLimit < yNativeMin )
		return false;
	if( m_leftLimit > xNativeMax )
		return false;
	if( m_rightLimit < xNativeMin )
		return false;

	return true;
}

void GeoRasterFileInfo::GetWgs84BoundingBox(double* minLat_wgs84, double* minLon_wgs84, double* maxLat_wgs84, double* maxLon_wgs84)
{
int num_x_pts = 9;
int num_y_pts = 9;
double delta_x = fabs(m_leftLimit-m_rightLimit)/(num_x_pts-1);
double delta_y = fabs(m_bottomLimit-m_topLimit)/(num_y_pts-1);
double min_x = std::min(m_leftLimit, m_rightLimit);
double min_y = std::min(m_bottomLimit, m_topLimit);
double lat_wgs84, lon_wgs84;
std::vector<double> lats;
std::vector<double> lons;

	for(int i=0 ; i<num_x_pts ; i++)
	{
		for(int j=0 ; j<num_y_pts ; j++)
		{
			if( i==0 || i==(num_x_pts-1) || j==0 || j==(num_y_pts-1))
			{
				double x = min_x + delta_x*i;
				double y = min_y + delta_y*j;
				NativeToWgs84Coords(x, y, &lat_wgs84, &lon_wgs84);
				lats.push_back(lat_wgs84);
				lons.push_back(lon_wgs84);
			}
		}
	}
	auto [min_lat_it, max_lat_it] = std::minmax_element(lats.begin(), lats.end());
	auto [min_lon_it, max_lon_it] = std::minmax_element(lons.begin(), lons.end());
	*minLat_wgs84 = *min_lat_it;
	*maxLat_wgs84 = *max_lat_it;
	*minLon_wgs84 = *min_lon_it;
	*maxLon_wgs84 = *max_lon_it;
}

// assumes (x=0, y=0) is the raster's top left pixel
void GeoRasterFileInfo::GetPixelNativeCoord(uint32_t xIndex, uint32_t yIndex, double* xNativeCoord, double* yNativeCoord)
{
	*xNativeCoord = m_leftLimit + (m_pixelWidth/2) + (xIndex*m_pixelWidth);
	*yNativeCoord = m_topLimit - (m_pixelHeight/2) - (yIndex*m_pixelHeight);
}

// assumes (x=0, y=0) is the raster's top left pixel
void GeoRasterFileInfo::GetPixelWgs84Coord(uint32_t xIndex, uint32_t yIndex, double* lat_wgs84, double* lon_wgs84)
{
double nativeXCoord, nativeYCoord;

	GetPixelNativeCoord(xIndex, yIndex, &nativeXCoord, &nativeYCoord);
	NativeToWgs84Coords(nativeXCoord, nativeYCoord, lat_wgs84, lon_wgs84);
}

// assumes (x=0, y=0) is the raster's top left pixel
void GeoRasterFileInfo::GetPixelIndex(double lat_wgs84, double lon_wgs84, uint32_t* xIndex, uint32_t* yIndex)
{
double xRealIndex, yRealIndex;

	GetPixelRealIndex(lat_wgs84, lon_wgs84, &xRealIndex, &yRealIndex);
	if( xRealIndex < 0 )
		*xIndex = 0;
	else
		*xIndex = std::min((uint32_t) lround(xRealIndex), m_rasterWidth-1);
	if( yRealIndex < 0 )
		*yIndex = 0;
	else
		*yIndex = std::min((uint32_t) lround(yRealIndex), m_rasterHeight-1);
}

// assumes (x=0, y=0) is the raster's top left pixel
void GeoRasterFileInfo::GetPixelRealIndex(double lat_wgs84, double lon_wgs84, double* xRealIndex, double* yRealIndex)
{
double nativeXCoord, nativeYCoord;

	if( Wgs84ToNativeCoords(lat_wgs84, lon_wgs84, &nativeXCoord, &nativeYCoord) == true )
	{
		*xRealIndex = (nativeXCoord - m_leftLimit - (m_pixelWidth/2)) / m_pixelWidth;
		*yRealIndex = (-nativeYCoord + m_topLimit - (m_pixelHeight/2)) / m_pixelHeight;
	}
	else
		*xRealIndex = *yRealIndex = -1;
}

// note: returned x1 and x2 may be equal when (lat,lon) is near the edge of the raster. Same with y1 and y2.
void GeoRasterFileInfo::GetSurroundingPixelIndexes(double lat_wgs84, double lon_wgs84, 
                                                   uint32_t* x1, uint32_t* x2, uint32_t* y1, uint32_t* y2,
                                                   double* xRealIndex, double* yRealIndex)
{
double xDblIndex, yDblIndex;

	GetPixelRealIndex(lat_wgs84, lon_wgs84, &xDblIndex, &yDblIndex);
	
	if(xDblIndex < 0)
		*x1 = *x2  = 0;
	else
	{
		*x1 = std::min((uint32_t) xDblIndex, m_rasterWidth-1);
		*x2 = std::min(*x1+1, m_rasterWidth-1);
	}

	if(yDblIndex < 0)
		*y1 = *y2  = 0;
	else
	{
		*y1 = std::min((uint32_t) yDblIndex, m_rasterHeight-1);
		*y2 = std::min(*y1+1, m_rasterHeight-1);
	}

	if( xRealIndex != nullptr )
		*xRealIndex = xDblIndex;
	if( yRealIndex != nullptr )
		*yRealIndex = yDblIndex;		
}

void GeoRasterFileInfo::BilinearInterpl(double x1, double x2, double y1, double y2,
                                        double Q11, double Q12, double Q21, double Q22,
                                        double x, double y, double* result)
{
double x2_x1 = x2-x1;
double y2_y1 = y2-y1;

	if( y2_y1 != 0 && x2_x1 != 0 )
	{
	double R1, R2;
	double a = (x2-x)/x2_x1;
	double b = (x-x1)/x2_x1;

		R1 = Q11*a + Q21*b;
		R2 = Q12*a + Q22*b;
		*result = (R1*(y2-y) + R2*(y-y1))/y2_y1;
	}
	// if interpolation cannot be performed, set result to the closest point's value
	else if( y2_y1 == 0 && x2_x1 == 0 )
		*result = Q11;
	else if( y2_y1 == 0 )
	{
		if( fabs(x-x1) < fabs(x-x2) )
			*result = Q11;
		else
			*result = Q21;
	}
	else
	{
		if( fabs(y-y1) < fabs(y-y2) )
			*result = Q11;
		else
			*result = Q12;
	}
}

// Returns false (failure) when (lat, lon) cannot be converted using the raster's
// assigned UTM zone (i.e. m_zone)
bool GeoRasterFileInfo::pGeogToUTM(double lat, double lon, double* easting, double* northing)
{
	try
	{
	GeographicLib::GeoCoords pt(lat, lon, m_zone);

		*easting = pt.Easting();
		*northing = pt.Northing();
	}
	catch(const std::exception& e)
	{
		return false;
	}
	
	return true;
}

void GeoRasterFileInfo::pUTMtoGeog(double easting, double northing, double* lat, double* lon)
{
GeographicLib::GeoCoords pt(m_zone, m_northp, easting, northing);

	*lat = pt.Latitude();
	*lon = pt.Longitude();
}

void GeoRasterFileInfo::pToWgs84(double lat, double lon, double* lat_wgs84, double* lon_wgs84)
{
	if( m_applyDatumTransform && m_toWgs84HelmertParams.size() == 7 )
	{
	const GeographicLib::Geocentric& earth = GeographicLib::Geocentric::WGS84();
	double* params = m_toWgs84HelmertParams.data();
	double x, y, z, x_out, y_out, z_out, h;

		earth.Forward(lat, lon, 0, x, y, z); // convert to geocentric coordinates
		pHelmert7ParamsForward(x, y, z, params[0], params[1], params[2],
		                       params[3], params[4], params[5], params[6],
		                       &x_out, &y_out, &z_out);
		earth.Reverse(x_out, y_out, z_out, *lat_wgs84, *lon_wgs84, h);
	}
	else
	{
		*lat_wgs84 = lat;
		*lon_wgs84 = lon;
	}
}

void GeoRasterFileInfo::pFromWgs84(double lat_wgs84, double lon_wgs84, double* lat, double* lon)
{
	if( m_applyDatumTransform && m_toWgs84HelmertParams.size() == 7 )
	{
	const GeographicLib::Geocentric& earth = GeographicLib::Geocentric::WGS84();
	double* params = m_toWgs84HelmertParams.data();
	double x, y, z, x_out, y_out, z_out, h;

		earth.Forward(lat_wgs84, lon_wgs84, 0, x, y, z); // convert to geocentric coordinates
		pHelmert7ParamsReverse(x, y, z, params[0], params[1], params[2],
		                       params[3], params[4], params[5], params[6],
		                       &x_out, &y_out, &z_out);
		earth.Reverse(x_out, y_out, z_out, *lat, *lon, h);
	}
	else
	{
		*lat = lat_wgs84;
		*lon = lon_wgs84;
	}
}

void GeoRasterFileInfo::pHelmert7ParamsForward(double x, double y, double z, double dx, double dy, double dz,
                                               double rx, double ry, double rz, double s,
                                               double* x_out, double* y_out, double* z_out)
{
	double scale = 1 + (s*1e-6);
	*x_out = scale*(x + rz*y - ry*z) + dx;
	*y_out = scale*(-rz*x + y + rx*z) + dy;
	*z_out = scale*(ry*x - rx*y + z) + dz;
}

void GeoRasterFileInfo::pHelmert7ParamsReverse(double x, double y, double z, double dx, double dy, double dz,
                                               double rx, double ry, double rz, double s,
                                               double* x_out, double* y_out, double* z_out)
{
	double scale = 1 + (s*1e-6);
	x = (x-dx)/scale;
	y = (y-dy)/scale;
	z = (z-dz)/scale;
	*x_out = (x - rz*y + ry*z);
	*y_out = (rz*x + y - rx*z);
	*z_out = (-ry*x + rx*y + z);
}
