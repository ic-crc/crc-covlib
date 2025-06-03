/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once

class TopographicSource
{
public:
	TopographicSource();
	virtual ~TopographicSource();

	// To ask the topographic source to clear any cache, close opened files, etc.
	virtual void ReleaseResources(bool clearCaches) = 0;
};