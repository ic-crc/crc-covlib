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
	float freq_MHz;
	float dist_m;
	float obsDepth_m;
	float intersectDept_m;
	int obsBlockCount;
	float terrainRatio;
	int aboveFzCount;
	float pathLoss_dB;
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
				row.freq_MHz = atof(tokens[0].c_str());
				row.dist_m = atof(tokens[1].c_str());
				row.obsDepth_m = atof(tokens[2].c_str());
				row.intersectDept_m = atof(tokens[3].c_str());
				row.obsBlockCount = atoi(tokens[4].c_str());
				row.terrainRatio = atof(tokens[5].c_str());
				row.aboveFzCount = atoi(tokens[6].c_str());
				row.pathLoss_dB = atof(tokens[7].c_str());
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
IPathObscuraModel* model = NewPathObscuraModel();
double diff;
double maxDiff=0;
float result_dB;

	rows = readCsv();
	for(size_t i=0 ; i<rows.size() ; i++)
	{
		row = &(rows[i]);
		result_dB = model->PathLoss(row->freq_MHz, row->dist_m, row->obsDepth_m, row->intersectDept_m,
		                            row->obsBlockCount, row->terrainRatio, row->aboveFzCount);
		diff = fabs(result_dB - row->pathLoss_dB);
		maxDiff = std::max(maxDiff, diff);
	}
	model->Release();
	std::cout << rows.size() << " test rows" << std::endl;
	std::cout << "max diff: " << maxDiff << " " << (maxDiff < 0.0001 ? "PASSED" : "FAILED") << std::endl;

	return 0;
}
