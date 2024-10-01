#pragma once

class TopographicSource
{
public:
	TopographicSource();
	virtual ~TopographicSource();

	// To ask the topographic source to clear any cache, close opened files, etc.
	virtual void ReleaseResources(bool clearCaches) = 0;
};