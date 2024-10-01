#pragma once
#include <vector>
#include <map>


class AntennaPattern
{
public:
	AntennaPattern(void);
	virtual ~AntennaPattern(void);

	struct HPatternEntry
	{
		double azimuth_deg; // 0 to 359.99 (0=North, 90=East, etc.)
		double gain_dB;
		bool operator < (const HPatternEntry &other) const { return azimuth_deg < other.azimuth_deg; }
	};

	void ClearHPattern();
	unsigned int NumHPatternEntries() const;
	void AddHPatternEntry(double azimuth_deg, double gain_dB);
	HPatternEntry GetHPatternEntry(unsigned int entryIndex) const;
	double GetHPatternGain(double azimuth_deg) const;
	double NormalizeHPattern();

	struct VPatternEntry
	{
		//double azimuth_deg; // 0 to 359.99 (0=North, 90=East, etc.)
		double elevAngle_deg; // -90 (towards sky) to +90 (towards ground)
		double gain_dB;
		bool operator < (const VPatternEntry &other) const { return elevAngle_deg < other.elevAngle_deg; }
	};

	void ClearVPattern();
	void ClearVPatternSlice(int sliceAzimuth_deg);
	std::vector<int> GetVPatternSlices() const;
	unsigned int NumVPatternSliceEntries(int sliceAzimuth_deg) const;
	void AddVPatternSliceEntry(int sliceAzimuth_deg, double elevAngle_deg, double gain_dB);
	VPatternEntry GetVPatternSliceEntry(int sliceAzimuth_deg, unsigned int entryIndex) const;
	double GetVPatternSliceGain(int sliceAzimuth_deg, double elevAngle_deg) const;
	double GetVPatternGain(double azimuth_deg, double elevAngle_deg) const;
	double NormalizeVPattern();

	void SetElectricalTilt(double elecricalTilt_deg);
	double GetElectricalTilt() const;
	void SetMechanicalTilt(double azimuth_deg, double mechanicalTilt_deg);
	double GetMechanicalTilt() const;
	double GetMechanicalTiltAzimuth() const;

	enum INTERPOLATION_ALGORITHM
	{
		H_PATTERN_ONLY = 1,
		V_PATTERN_ONLY = 2,
		SUMMING = 3,
		WEIGHTED_SUMMING = 4,
		HYBRID = 5,
		MODIFIED_WEIGHTED_SUMMING = 6,
		HPI = 7
	};

	double Gain(double azimuth_deg, double elevAngle_deg, INTERPOLATION_ALGORITHM algorithm=WEIGHTED_SUMMING, bool applyTilt=true) const;

	void Print() const;

	bool ExportToRadioMobileV3File(const char* pathname, bool applyTilt);
	void LoadRadioMobileV3File(const char* pathname);
	bool ExportTo3DCsvFile(const char* pathname, bool applyTilt, int azmStep_deg, int elvStep_deg, INTERPOLATION_ALGORITHM algorithm);
	void Load3DCsvFile(const char* pathname);

private:
	void pUpdateVertPatternSortedAzms();
	double pdBtoLinear(double gain_dB) const;
	double pLinearTodB(double gain_linear) const;
	double pGainInterpolation(double angle, double gain1, double angle1, double gain2, double angle2, bool linearGainInterpolation) const;

	double pSumGain(double azimuth_deg, double elevAngle_deg) const;
	double pWeightedSummingGain(double azimuth_deg, double elevAngle_deg) const;
	double pHybridGain(double azimuth_deg, double elevAngle_deg) const;
	double pModifiedWeightedSummingGain(double azimuth_deg, double elevAngle_deg) const;
	double pHorizontalProjectionInterpolationGain(double azimuth_deg, double elevAngle_deg) const;

	std::vector<HPatternEntry> pHorizPattern;
	std::map<int, std::vector<VPatternEntry>> pVertPatternSlices;
	std::vector<int> pVertPatternSortedAzms;
	double pElectricalTiltDeg;
	double pMechanicalTiltDeg;
	double pMechanicalTiltAzimuthDeg;
	double pLinearGainInterpolation;
};
