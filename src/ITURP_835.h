/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once


class ITURP_835
{
public:
	ITURP_835();
	~ITURP_835();

    static double StandardPressure(double geometricHeight_km);

protected:
    static double _ToGeopotentialHeight(double geometricHeight_km);
};