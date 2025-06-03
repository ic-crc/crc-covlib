/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once


class ITURP_1057
{
public:
    ITURP_1057();
    ~ITURP_1057();

    static double Qinv(double p);
    static double Finv(double p);
};