#include "AntennaPattern.h"
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>


AntennaPattern::AntennaPattern(void)
{
	pElectricalTiltDeg = 0;
	pMechanicalTiltDeg = 0;
	pMechanicalTiltAzimuthDeg = 0;
	pLinearGainInterpolation = true;
}

AntennaPattern::~AntennaPattern(void)
{
}

void AntennaPattern::ClearHPattern()
{
	pHorizPattern.clear();
}

unsigned int AntennaPattern::NumHPatternEntries() const
{
	return pHorizPattern.size();
}

AntennaPattern::HPatternEntry AntennaPattern::GetHPatternEntry(unsigned int entryIndex) const
{
	if( entryIndex < pHorizPattern.size() )
		return pHorizPattern[entryIndex];

	return {0,0};
}

// azimuth_deg: 0 to 359.99
// gain_dB: gain relative to maximum antenna gain (should be negative)
void AntennaPattern::AddHPatternEntry(double azimuth_deg, double gain_dB)
{
HPatternEntry entry;

	if( azimuth_deg < 0 || azimuth_deg >= 360 )
		return;

	entry.azimuth_deg = azimuth_deg;
	entry.gain_dB = gain_dB;

	pHorizPattern.push_back(entry);

	std::sort(pHorizPattern.begin(), pHorizPattern.end());
}

double AntennaPattern::GetHPatternGain(double azimuth_deg) const
{
double azm1, azm2;
double gain1, gain2;
std::size_t hPatternSize = pHorizPattern.size();

	if( hPatternSize == 0 )
		return 0;

	if( hPatternSize == 1 )
		return pHorizPattern[0].gain_dB;

	while( azimuth_deg >= 360 )
		azimuth_deg -= 360;

	while( azimuth_deg < 0 )
		azimuth_deg += 360;

	for(std::size_t i=0 ; i<hPatternSize-1 ; i++)
	{
		azm1 = pHorizPattern[i].azimuth_deg;
		azm2 = pHorizPattern[i+1].azimuth_deg;
		if( azimuth_deg >= azm1 && azimuth_deg <= azm2 )
		{
			gain1 = pHorizPattern[i].gain_dB;
			gain2 = pHorizPattern[i+1].gain_dB;
			return pGainInterpolation(azimuth_deg, gain1, azm1, gain2, azm2, pLinearGainInterpolation);
		}
	}

	azm1 = pHorizPattern[hPatternSize-1].azimuth_deg;
	azm2 = pHorizPattern[0].azimuth_deg;
	if( azimuth_deg >= azm1 )
		azm2 += 360;
	else
		azm1 -= 360;
	if( azimuth_deg >= azm1 && azimuth_deg <= azm2 )
	{
		gain1 = pHorizPattern[hPatternSize-1].gain_dB;
		gain2 = pHorizPattern[0].gain_dB;
		return pGainInterpolation(azimuth_deg, gain1, azm1, gain2, azm2, pLinearGainInterpolation);
	}

	return 0;
}

double AntennaPattern::NormalizeHPattern()
{
std::vector<HPatternEntry>::iterator iter;

	iter = std::max_element(pHorizPattern.begin(), pHorizPattern.end(), [](HPatternEntry a, HPatternEntry b)
	{
		return a.gain_dB < b.gain_dB;
	});

	if( iter != pHorizPattern.end() )
	{
	double maxEntryGain = (*iter).gain_dB;

		for(std::size_t i=0 ; i<pHorizPattern.size() ; i++)
			pHorizPattern[i].gain_dB -= maxEntryGain;
		return -maxEntryGain;
	}
	else
		return 0;
}

void AntennaPattern::ClearVPattern()
{
	pVertPatternSlices.clear();
	pUpdateVertPatternSortedAzms();
}

void AntennaPattern::ClearVPatternSlice(int sliceAzimuth_deg)
{
std::map<int, std::vector<VPatternEntry>>::iterator iter = pVertPatternSlices.find(sliceAzimuth_deg);

	if( iter != pVertPatternSlices.end() )
	{
		pVertPatternSlices.erase(iter);
		pUpdateVertPatternSortedAzms();
	}
}

void AntennaPattern::pUpdateVertPatternSortedAzms()
{
	pVertPatternSortedAzms.clear();

	for(std::map<int, std::vector<VPatternEntry>>::const_iterator iter = pVertPatternSlices.begin() ; iter != pVertPatternSlices.end() ; ++iter)
		pVertPatternSortedAzms.push_back(iter->first);
	
	std::sort(pVertPatternSortedAzms.begin(), pVertPatternSortedAzms.end());
}

std::vector<int> AntennaPattern::GetVPatternSlices() const
{
	return pVertPatternSortedAzms;
}

unsigned int AntennaPattern::NumVPatternSliceEntries(int sliceAzimuth_deg) const
{
std::map<int, std::vector<VPatternEntry>>::const_iterator iter = pVertPatternSlices.find(sliceAzimuth_deg);

	if( iter != pVertPatternSlices.end() )
		return iter->second.size();

	return 0;
}

// sliceAzimuth_deg: 0 to 359
// elevAngle_deg: -90 (towards sky) to +90 (towards ground) 
// gain_dB: gain relative to maximum antenna gain (should be negative)
void AntennaPattern::AddVPatternSliceEntry(int sliceAzimuth_deg, double elevAngle_deg, double gain_dB)
{
VPatternEntry entry;

	if( sliceAzimuth_deg < 0 || sliceAzimuth_deg >= 360 )
		return;

	if( elevAngle_deg < -90 || elevAngle_deg > 90 )
		return;

	//entry.azimuth_deg = sliceAzimuth_deg;
	entry.elevAngle_deg = elevAngle_deg;
	entry.gain_dB = gain_dB;
	pVertPatternSlices[sliceAzimuth_deg].push_back(entry);
	std::sort(pVertPatternSlices[sliceAzimuth_deg].begin(), pVertPatternSlices[sliceAzimuth_deg].end());
	pUpdateVertPatternSortedAzms();
}

AntennaPattern::VPatternEntry AntennaPattern::GetVPatternSliceEntry(int sliceAzimuth_deg, unsigned int entryIndex) const
{
std::map<int, std::vector<VPatternEntry>>::const_iterator iter = pVertPatternSlices.find(sliceAzimuth_deg);

	if (iter != pVertPatternSlices.end())
	{
		if( entryIndex < pVertPatternSlices.at(sliceAzimuth_deg).size() )
			return pVertPatternSlices.at(sliceAzimuth_deg)[entryIndex];
	}

	return {0,0};
}

// elevAngle_deg: -90 (towards sky) to +90 (towards ground)
double AntennaPattern::GetVPatternSliceGain(int sliceAzimuth_deg, double elevAngle_deg) const
{
	std::map<int, std::vector<VPatternEntry>>::const_iterator iter = pVertPatternSlices.find(sliceAzimuth_deg);
	if( iter == pVertPatternSlices.end() )
		return 0;

	const std::vector<VPatternEntry>& VPattern = iter->second;

	if( VPattern.size() == 0 )
		return 0;

	if( VPattern.size() == 1 )
		return VPattern[0].gain_dB;

	if( elevAngle_deg <= VPattern[0].elevAngle_deg )
		return VPattern[0].gain_dB; // will return here if sliceAzimuth_deg < -90

	if( elevAngle_deg >= VPattern.back().elevAngle_deg )
		return VPattern.back().gain_dB; // will return here if sliceAzimuth_deg > 90

	for(size_t i=0 ; i<VPattern.size()-1 ; i++)
	{
		if( elevAngle_deg >= VPattern[i].elevAngle_deg && elevAngle_deg <= VPattern[i+1].elevAngle_deg )
		{
		double gain1 = VPattern[i].gain_dB;
		double angle1 = VPattern[i].elevAngle_deg;
		double gain2 = VPattern[i+1].gain_dB;
		double angle2 = VPattern[i+1].elevAngle_deg;

			return pGainInterpolation(elevAngle_deg, gain1, angle1, gain2, angle2, pLinearGainInterpolation);
		}
	}

	return 0;
}

// elevAngle_deg: -90 (towards sky) to +90 (towards ground)
double AntennaPattern::GetVPatternGain(double azimuth_deg, double elevAngle_deg) const
{
const std::vector<int> & azimuths = pVertPatternSortedAzms;
int azm1, azm2;
double gain1, gain2;
std::size_t numAzimuths = azimuths.size();

	if( numAzimuths == 0 )
		return 0;

	if( numAzimuths == 1 )
		return GetVPatternSliceGain(azimuths[0], elevAngle_deg);

	while( azimuth_deg >= 360 )
		azimuth_deg -= 360;

	while( azimuth_deg < 0 )
		azimuth_deg += 360;

	for(std::size_t i=0 ; i<numAzimuths-1 ; i++)
	{
		azm1 = azimuths[i];
		azm2 = azimuths[i+1];
		if( azimuth_deg >= azm1 && azimuth_deg <= azm2 )
		{
			gain1 = GetVPatternSliceGain(azm1, elevAngle_deg);
			gain2 = GetVPatternSliceGain(azm2, elevAngle_deg);
			return pGainInterpolation(azimuth_deg, gain1, azm1, gain2, azm2, pLinearGainInterpolation);
		}
	}

	// azimuth_deg should be between the last and first azimuth of the pattern
	azm1 = azimuths[numAzimuths-1];
	azm2 = azimuths[0];
	if( azimuth_deg >= azm1 )
		azm2 += 360;
	else
		azm1 -= 360;
	if( azimuth_deg >= azm1 && azimuth_deg <= azm2 )
	{
		gain1 = GetVPatternSliceGain(azimuths[numAzimuths-1], elevAngle_deg);
		gain2 = GetVPatternSliceGain(azimuths[0], elevAngle_deg);
		return pGainInterpolation(azimuth_deg, gain1, azm1, gain2, azm2, pLinearGainInterpolation);
	}

	return 0;
}

double AntennaPattern::NormalizeVPattern()
{
const std::vector<int> & azimuths = pVertPatternSortedAzms;
int azm;
unsigned int numEntries;
double maxEntryGain = std::numeric_limits<double>::lowest();

	for(std::size_t i=0 ; i<azimuths.size() ; i++)
	{
		azm = azimuths[i];
		numEntries = NumVPatternSliceEntries(azm);
		for(unsigned int j=0 ; j<numEntries ; j++)
			maxEntryGain = std::max(maxEntryGain, GetVPatternSliceEntry(azm, j).gain_dB);
	}

	if( maxEntryGain != std::numeric_limits<double>::lowest() )
	{
		for(std::size_t i=0 ; i<azimuths.size() ; i++)
		{
			azm = azimuths[i];
			numEntries = NumVPatternSliceEntries(azm);
			for(unsigned int j=0 ; j<numEntries ; j++)
				pVertPatternSlices[azm][j].gain_dB -= maxEntryGain;
		}
		return -maxEntryGain;
	}
	else
		return 0;
}

// -90 (towards sky) to +90 (towards ground)
void AntennaPattern::SetElectricalTilt(double elecricalTilt_deg)
{
	if( elecricalTilt_deg >= -90.0 && elecricalTilt_deg <= 90.0 )
		pElectricalTiltDeg = elecricalTilt_deg;
}
	
double AntennaPattern::GetElectricalTilt() const
{
	return pElectricalTiltDeg;
}

// -90 (towards sky) to +90 (towards ground)
void AntennaPattern::SetMechanicalTilt(double azimuth_deg, double mechanicalTilt_deg)
{
	if( mechanicalTilt_deg >= -90.0 && mechanicalTilt_deg <= 90.0 )
		pMechanicalTiltDeg = mechanicalTilt_deg;
	else
		return;

	while( azimuth_deg >= 360 )
		azimuth_deg -= 360;

	while( azimuth_deg < 0 )
		azimuth_deg += 360;

	pMechanicalTiltAzimuthDeg = azimuth_deg;
}
	
double AntennaPattern::GetMechanicalTilt() const
{
	return pMechanicalTiltDeg;
}
	
double AntennaPattern::GetMechanicalTiltAzimuth() const
{
	return pMechanicalTiltAzimuthDeg;
}

void AntennaPattern::Print() const
{
const std::vector<int> & azimuths = pVertPatternSortedAzms;
unsigned int numEntries;
VPatternEntry ventry;

	printf("Horizontal Pattern [azm (deg), gain (dB)]:\n");
	for(std::size_t i=0 ; i<pHorizPattern.size() ; i++)
		printf("%.2f %.2f\n", pHorizPattern[i].azimuth_deg, pHorizPattern[i].gain_dB);
	
	printf("Vertical Pattern [azm (deg), elev (deg), gain (dB)], tilts not included:\n");
	for(std::size_t i=0 ; i<azimuths.size() ; i++)
	{
		numEntries = NumVPatternSliceEntries(azimuths[i]);
		for(unsigned int j=0 ; j<numEntries ; j++)
		{
			ventry = GetVPatternSliceEntry(azimuths[i], j);
			printf("%d %.2f %.2f\n", azimuths[i], ventry.elevAngle_deg, ventry.gain_dB);
		}
	}

	printf("\n");
}

// elevAngle_deg: -90 (towards sky) to +90 (towards ground)
double AntennaPattern::Gain(double azimuth_deg, double elevAngle_deg, INTERPOLATION_ALGORITHM algorithm, bool applyTilt) const
{
double gain_dB = 0;
constexpr double PI_ON_180 = 0.017453292519943295769;

	if( applyTilt )
	{
		elevAngle_deg -= pElectricalTiltDeg;
		elevAngle_deg -= pMechanicalTiltDeg*cos(PI_ON_180*(pMechanicalTiltAzimuthDeg+azimuth_deg));
	}

	switch (algorithm)
	{
	case H_PATTERN_ONLY:
		gain_dB = GetHPatternGain(azimuth_deg);
		break;
	case V_PATTERN_ONLY:
		gain_dB = GetVPatternGain(azimuth_deg, elevAngle_deg);
		break;
	case SUMMING:
		gain_dB = pSumGain(azimuth_deg, elevAngle_deg);
		break;
	case WEIGHTED_SUMMING:
		gain_dB = pWeightedSummingGain(azimuth_deg, elevAngle_deg);
		break;
	case HYBRID:
		gain_dB = pHybridGain(azimuth_deg, elevAngle_deg);
		break;
	case MODIFIED_WEIGHTED_SUMMING:
		gain_dB = pModifiedWeightedSummingGain(azimuth_deg, elevAngle_deg);
		break;
	case HPI:
		gain_dB = pHorizontalProjectionInterpolationGain(azimuth_deg, elevAngle_deg);
		break;
	default:
		gain_dB = pSumGain(azimuth_deg, elevAngle_deg);
		break;
	}

	return gain_dB;
}

double AntennaPattern::pSumGain(double azimuth_deg, double elevAngle_deg) const
{
double GH = GetHPatternGain(azimuth_deg);
//double GV = GetVPatternGain(azimuth_deg, elevAngle_deg);
double GV = GetVPatternSliceGain(0, elevAngle_deg); // should only use front plane vertical pattern slice data

	return GH + GV;
}

// https://www.mathworks.com/help/antenna/ref/patternfromslices.html
// https://www.eurasip.org/Proceedings/Ext/WSA2009/manuscripts/9031.pdf
// https://www.researchgate.net/publication/228799992_Modeling_of_3D_field_patterns_of_downtilted_antennas_and_their_impact_on_cellular_systems
double AntennaPattern::pWeightedSummingGain(double azimuth_deg, double elevAngle_deg) const
{
double k = 2.0;
double GH = GetHPatternGain(azimuth_deg);
//double GV = GetVPatternGain(azimuth_deg, elevAngle_deg);
double GV = GetVPatternSliceGain(0, elevAngle_deg); // should only use front plane vertical pattern slice data
double hor = pow(10.0, GH/10.0);
double vert = pow(10.0, GV/10.0);
double w1 = vert*(1.0-hor);
double w2 = hor*(1.0-vert);

	if( w1 == 0 && w2 == 0 )
		return 0;
	else
		return (GH*w1+GV*w2) / pow(pow(w1,k)+pow(w2,k),1.0/k);
}

// Hybrid between Summing and Weighted Summing
// https://ieeexplore.ieee.org/document/1461546
// https://www.researchgate.net/publication/3018156_A_novel_technique_for_the_approximation_of_3-D_antenna_radiation_patterns
double AntennaPattern::pHybridGain(double azimuth_deg, double elevAngle_deg) const
{
double k = 2.0;
double GH = GetHPatternGain(azimuth_deg);
//double GV = GetVPatternGain(azimuth_deg, elevAngle_deg);
double GV = GetVPatternSliceGain(0, elevAngle_deg); // should only use front plane vertical pattern slice data
double hor = pow(10.0, GH/10.0);
double vert = pow(10.0, GV/10.0);
double w1 = vert*(1.0-hor);
double w2 = hor*(1.0-vert);
if( w1 == 0 && w2 == 0 ) return 0;
double Gcs = (GH*w1+GV*w2) / pow(pow(w1,k)+pow(w2,k),1.0/k);
double Gsum = GH + GV;
double n = 3.5;
double w3 = pow(hor*vert, 1.0/n);

	return Gsum*w3 + Gcs*(1.0-w3);
}

// WARNING: Not sure this implementation is correct
// https://ieeexplore.ieee.org/document/7268850
// https://www.researchgate.net/publication/281746121_A_Three-dimensional_Directive_Antenna_Pattern_Interpolation_Method
// also https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=801481
double AntennaPattern::pModifiedWeightedSummingGain(double azimuth_deg, double elevAngle_deg) const
{
constexpr double PI_ON_180 = 0.017453292519943295769;
constexpr double PI = 3.14159265358979323846;
double Gtheta1, Gtheta2, Gphi1, Gphi2;
double theta, phi;
double W1, W2, W3;
double GHm, GVm;

	Gtheta1 = GetVPatternGain(azimuth_deg, -90);
	Gtheta2 = GetHPatternGain(azimuth_deg);
	Gphi1 = GetVPatternGain(0, elevAngle_deg);
	Gphi2 = GetVPatternGain(180, elevAngle_deg);

	theta = elevAngle_deg+90;
	while( theta > 90 ) theta -= 180;
	while( theta < -90 ) theta += 180;
	theta *= PI_ON_180;

	phi = azimuth_deg;
	while( phi > 180 ) phi -= 360;
	while( phi < -180 ) phi += 360;
	phi *= PI_ON_180;

	W1 = 1 - (2*fabs(theta)/PI);
	W2 = 1 - (fabs(phi)/PI);
	W3 = (2/PI)*fabs((PI/4)-fabs(theta)+(PI/2)-fabs((PI/2)-fabs(phi)));

	GHm = Gtheta1*W1 + Gtheta2*(1-W1);
	GVm = Gphi1*W2 + Gphi2*(1-W2);

	return GHm*W3 + GVm*(1-W3);
}

// https://2021.help.altair.com/2021.1.2/winprop/topics/winprop/user_guide/aman/introduction/horizontal_projection_interpolation_winprop.htm
double AntennaPattern::pHorizontalProjectionInterpolationGain(double azimuth_deg, double elevAngle_deg) const
{
double GH = GetHPatternGain(azimuth_deg);
double GH0 = GetHPatternGain(0);
double GH180 = GetHPatternGain(180.0);
double GV = GetVPatternGain(azimuth_deg, elevAngle_deg);
double GV180_ = GetVPatternGain(azimuth_deg, -elevAngle_deg);
double absHA;

	while( azimuth_deg > 180 ) azimuth_deg -= 360;
	while( azimuth_deg < -180 ) azimuth_deg += 360;
	absHA = fabs(azimuth_deg);

	return GH - ( (1.0-(absHA/180.0))*(GH0-GV) + absHA/180.0*(GH180-GV180_) );
}

double AntennaPattern::pdBtoLinear(double gain_dB) const
{
	return pow(10.0, gain_dB/20.0);
}
	
double AntennaPattern::pLinearTodB(double gain_linear) const
{
	return 20.0*log10(gain_linear);
}

double AntennaPattern::pGainInterpolation(double angle, double gain1, double angle1, double gain2, double angle2, bool linearGainInterpolation) const
{
	if( angle2-angle1 != 0 )
	{
		if( linearGainInterpolation )
			return pLinearTodB(pdBtoLinear(gain1) + ((angle-angle1)*((pdBtoLinear(gain2)-pdBtoLinear(gain1))/(angle2-angle1))));
		else
			return gain1 + ((angle-angle1)*((gain2-gain1)/(angle2-angle1)));
	}
	else
		return gain1;
}

bool AntennaPattern::ExportToRadioMobileV3File(const char* pathname, bool applyTilt)
{
std::ofstream file;
bool success = false;

	file.open(pathname, std::ios::out | std::ios::trunc);
	if(file)
	{
		file << std::fixed << std::showpoint << std::setprecision(2);
		for(int azm=0 ; azm<360 ; azm++)
			file << Gain(azm, 0, H_PATTERN_ONLY, applyTilt) << std::endl;
		for(int elv=-90 ; elv<90 ; elv++)
			file << Gain(0, elv, V_PATTERN_ONLY, applyTilt) << std::endl;
		for(int elv=90 ; elv>-90 ; elv--)
			file << Gain(180, elv, V_PATTERN_ONLY, applyTilt) << std::endl;
		success = true;
	}
	file.close();
	return success;
}

void AntennaPattern::LoadRadioMobileV3File(const char* pathname)
{
std::ifstream file;
double gain_db;

	ClearHPattern();
	ClearVPattern();

	file.open(pathname, std::ios::in);
	if(file)
	{
		for(int azm=0 ; azm<360 ; azm++)
		{
			file >> gain_db;
			AddHPatternEntry(azm, gain_db);
		}
		file >> gain_db;
		AddVPatternSliceEntry(0, -90, gain_db);
		AddVPatternSliceEntry(180, -90, gain_db);
		for(int elv=-89 ; elv<90 ; elv++)
		{
			file >> gain_db;
			AddVPatternSliceEntry(0, elv, gain_db);
		}
		file >> gain_db;
		AddVPatternSliceEntry(0, 90, gain_db);
		AddVPatternSliceEntry(180, 90, gain_db);
		for(int elv=89 ; elv>-90 ; elv--)
		{
			file >> gain_db;
			AddVPatternSliceEntry(180, elv, gain_db);
		}
	}
	file.close();
}

// Exported file may be opened with Antenna Pattern Editor 2 for visuallization:
// https://www.wireless-planning.com/antenna-pattern-editor
bool AntennaPattern::ExportTo3DCsvFile(const char* pathname, bool applyTilt, int azmStep_deg, int elvStep_deg, INTERPOLATION_ALGORITHM algorithm)
{
std::ofstream file;
bool success = false;
std::vector<int> azms;
std::vector<int> elvs;
int azm, elv;
double gain_dB;

	for(int i=0 ; i<=360 ; i+=azmStep_deg)
		azms.push_back(i);

	for(int i=0 ; i<=180 ; i+=elvStep_deg)
		elvs.push_back(i);

	file.open(pathname, std::ios::out | std::ios::trunc);
	if(file)
	{
		file << "3D;";
		for(std::size_t i=0 ; i<azms.size()-1 ; i++)
			file << azms[i] << ";";
		file << azms.back() << std::endl;

		file << std::fixed << std::showpoint << std::setprecision(2);

		for(std::size_t i=0 ; i<elvs.size() ; i++)
		{
			elv = elvs[i];
			file << elv << ";";
			for(std::size_t j=0 ; j<azms.size()-1 ; j++)
			{
				azm = azms[j];
				gain_dB = Gain(azm, elv-90, algorithm, applyTilt);
				file << gain_dB << ";";
			}
			gain_dB = Gain(azms.back(), elv-90, algorithm, applyTilt);
			file << gain_dB << std::endl;
		}
		success = true;
	}
	file.close();
	return success;
}

void AntennaPattern::Load3DCsvFile(const char* pathname)
{
std::ifstream file;
double gain_dB;
std::vector<int> azms;
std::string line;
int azm, elv;
std::string str;
std::istringstream iss;

	ClearHPattern();
	ClearVPattern();

	file.open(pathname, std::ios::in);
	if(file)
	{
		std::getline(file, line);
		iss.str(line);
		std::getline(iss, str, ';'); // "3D"
		while( std::getline(iss, str, ';') )
			azms.push_back(atoi(str.c_str()));

		while( std::getline(file, line) )
		{
			iss.clear();
			iss.str(line);
			std::getline(iss, str, ';');
			elv = atoi(str.c_str());
			for(unsigned int i=0 ; std::getline(iss, str, ';') ; i++ )
			{
				azm = azms[i];
				gain_dB = atof(str.c_str());
				AddVPatternSliceEntry(azm, elv-90, gain_dB);
			}
		}
	}
	file.close();
}