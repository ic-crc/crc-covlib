/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../../src/ITURP_2109.h"


struct P2109Input
{
	double freq;
	double prob;
	int bldgType;
	double elevAngle;
	double BEL;
};

std::vector<std::string> tokenize(const char* text, const char* separators)
{
int i=0;
std::string str;
std::vector<std::string> strList;
bool separatorFound;

	while( text[i] != '\0' )
	{
		separatorFound = false;
		for(int j=0 ; separators[j] != '\0' ; j++)
		{
			if( text[i] == separators[j] )
			{
				separatorFound = true;
				break;
			}
		}

		if( separatorFound == false )
			str.push_back(text[i]);
		else
		{
			strList.push_back(str);
			str.clear();
		}
		i++;
	}
	strList.push_back(str);

	return strList;
}

std::vector<P2109Input> readCsv()
{
std::fstream csvFile;
std::string line;
std::vector<std::string> tokens;
std::vector<P2109Input> content;

	// file generated from official matlab implementation at
	// https://github.com/eeveetza/p2109
	csvFile.open("validation.csv", std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		getline(csvFile, line);
		while(csvFile.good())
		{
		P2109Input input;

			tokens = tokenize(line.c_str(), ",");
			if( tokens.size() >= 5 )
			{
				input.freq = atof(tokens[0].c_str());
				input.prob = atof(tokens[1].c_str());
				input.bldgType = atoi(tokens[2].c_str());
				input.elevAngle = atof(tokens[3].c_str());
				input.BEL = atof(tokens[4].c_str());
				content.push_back(input);
			}
			getline(csvFile, line);
		}
	}
	csvFile.close();

	return content;
}

int main(int argc, char* argv[])
{
std::vector<P2109Input> inputs;
double covlibBEL;
double BELDiff;
double maxBELDiff=0;
P2109Input* in;

	inputs = readCsv();
	for(size_t i=0 ; i<inputs.size() ; i++)
	{
		in = &(inputs[i]);
		covlibBEL = ITURP_2109::BuildingEntryLoss(in->freq, in->prob, (ITURP_2109::BuildingType)in->bldgType, in->elevAngle);
		BELDiff = fabs(covlibBEL-in->BEL);
		maxBELDiff = std::max(maxBELDiff, BELDiff);
	}

	std::cout << "Building entry loss (dB) max diff: " << maxBELDiff << " " << (maxBELDiff < 0.0001 ? "PASSED" : "FAILED") << std::endl;

	return 0;
}