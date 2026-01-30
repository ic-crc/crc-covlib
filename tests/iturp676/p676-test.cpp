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
#include "../../src/ITURP_676.h"


struct P676Input
{
	double f;
	double P;
	double rho;
	double T;
	double gamma0;
	double gammaW;
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

std::vector<P676Input> readCsv()
{
std::fstream csvFile;
std::string line;
std::vector<std::string> tokens;
std::vector<P676Input> content;

	csvFile.open("validation.csv", std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		getline(csvFile, line);
		while(csvFile.good())
		{
		P676Input input;

			tokens = tokenize(line.c_str(), ",");
			if( tokens.size() >= 6 )
			{
				input.f = atof(tokens[0].c_str());
				input.P = atof(tokens[1].c_str());
				input.rho = atof(tokens[2].c_str());
				input.T = atof(tokens[3].c_str());
				input.gamma0 = atof(tokens[4].c_str());
				input.gammaW = atof(tokens[5].c_str());
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
std::vector<P676Input> inputs;
double covlibGamma0, covlibGammaW;
double gamma0Diff, gammaWDiff;
double maxGamma0Diff=0, maxGammaWDiff=0;
P676Input* in;

	inputs = readCsv();
	for(size_t i=0 ; i<inputs.size() ; i++)
	{
		in = &(inputs[i]);
		covlibGamma0 = ITURP_676::AttenuationDueToDryAir(in->f, in->P, in->T, in->rho);
		covlibGammaW = ITURP_676::AttenuationDueToWaterVapour(in->f, in->P, in->T, in->rho);

		gamma0Diff = fabs(covlibGamma0-in->gamma0);
		gammaWDiff = fabs(covlibGammaW-in->gammaW);

		maxGamma0Diff = std::max(maxGamma0Diff, gamma0Diff);
		maxGammaWDiff = std::max(maxGammaWDiff, gammaWDiff);
	}

	std::cout << "Gamma (oxygen)      max diff: " << maxGamma0Diff << " " << (maxGamma0Diff < 0.0001 ? "PASSED" : "FAILED") << std::endl;
	std::cout << "Gamma (water vapor) max diff: " << maxGammaWDiff << " " << (maxGammaWDiff < 0.0001 ? "PASSED" : "FAILED") << std::endl;

	return 0;
}