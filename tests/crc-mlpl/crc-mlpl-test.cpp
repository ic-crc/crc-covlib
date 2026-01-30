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
#include "crc-ml.h"


struct csvRow
{
	double f_MHz;
	double d_m;
	double obs_m;
	double excessLoss_dB;
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

std::vector<csvRow> readCsv()
{
std::fstream csvFile;
std::string line;
std::vector<std::string> tokens;
std::vector<csvRow> content;

	csvFile.open("validation.csv", std::ios::in);
	if(csvFile)
	{
		getline(csvFile, line);
		getline(csvFile, line);
		while(csvFile.good())
		{
		csvRow row;

			tokens = tokenize(line.c_str(), ",");
			if( tokens.size() >= 4 )
			{
				row.f_MHz = atof(tokens[0].c_str());
				row.d_m = atof(tokens[1].c_str());
				row.obs_m = atof(tokens[2].c_str());
				row.excessLoss_dB = atof(tokens[3].c_str());
				content.push_back(row);
			}
			getline(csvFile, line);
		}
	}
	csvFile.close();

	return content;
}

int main(int argc, char* argv[])
{
std::vector<csvRow> rows;
csvRow* row;
IMLPLModel* model = NewMLPLModel();
double diff;
double maxDiff=0;
float result_dB;

	rows = readCsv();
	for(size_t i=0 ; i<rows.size() ; i++)
	{
		row = &(rows[i]);
		result_dB = model->ExcessLoss(row->f_MHz, row->d_m, row->obs_m);
		diff = fabs(result_dB - row->excessLoss_dB);
		maxDiff = std::max(maxDiff, diff);
	}
	model->Release();
	std::cout << rows.size() << " test rows" << std::endl;
	std::cout << "max diff: " << maxDiff << " " << (maxDiff < 0.0001 ? "PASSED" : "FAILED") << std::endl;

	return 0;
}
