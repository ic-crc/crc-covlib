/*
 * Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
 * Industry through the Communications Research Centre Canada.
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text.
 */

#pragma once
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <mutex>


class Logger
{
public:
	enum class Level
	{
		ERROR_LVL = 1,   // use for critical issues that require immediate attention and prevent normal operation
		WARNING_LVL = 2, // use for situations that don't prevent operation but indicate potential problems
		INFO_LVL = 3,    // use for important events and milestones in normal operation
		DEBUG_LVL = 4    // use for diagnosing issues during development and troubleshooting
	};

	// Singleton Pattern using the Meyers' Singleton technique
	static Logger& GetInstance()
	{
		static Logger instance;
		return instance;
	}

	void SetLevel(Level level)
	{
		pLevel = level;
	}

	void SwitchToFileMode(const std::string& filename, const std::string& sep=",")
	{
		pFileStream = std::make_unique<std::ofstream>(filename, std::ios::app);
		if (pFileStream->is_open())
		{
			pOutputStream = pFileStream.get();
			pSeparator = sep;
		}
		else
		{
			Logger::GetInstance().Log(Logger::Level::WARNING_LVL,
				"Could not open or create the log file, using console for logs.");
		}
	}

	void SwitchToConsoleMode(const std::string& sep=" ")
	{
		pOutputStream = &std::cout;
		pFileStream.reset();
		pSeparator = sep;
	}

	void SetSeparator(const std::string& sep)
	{
		pSeparator = sep;
	}

	void SetFloatPrecision(int precision)
	{
		pFloatPrecision = precision;
	}

	template<typename... Args>
	void Log(Level level, Args&&... args)
	{
		if (level > pLevel)
			return; // Early exit for performance

		std::lock_guard<std::mutex> lock(pMutex);
		*pOutputStream << std::fixed << std::setprecision(pFloatPrecision);
		*pOutputStream << "[" << pLevelToString(level) << "]";
		((*pOutputStream << pSeparator << args), ...);
		*pOutputStream << std::endl;
		*pOutputStream << std::defaultfloat; // Reset to default
	}

private:
	Logger() : pLevel(Level::WARNING_LVL), pOutputStream(&std::cout), pSeparator(" "), pFloatPrecision(7)
	{
	}

	std::string pLevelToString(Level level) const
	{
		switch(level)
		{
			case Level::ERROR_LVL: return "ERROR";
			case Level::WARNING_LVL: return "WARN";
			case Level::INFO_LVL: return "INFO";
			case Level::DEBUG_LVL: return "DEBUG";
			default: return "UNKNOWN";
		}
	}

	Level pLevel;
	std::ostream* pOutputStream;
	std::unique_ptr<std::ofstream> pFileStream;
	std::string pSeparator;
	int pFloatPrecision;
	std::mutex pMutex;
};

#define LOG_ERROR(...) Logger::GetInstance().Log(Logger::Level::ERROR_LVL, __VA_ARGS__)
#define LOG_WARNING(...) Logger::GetInstance().Log(Logger::Level::WARNING_LVL, __VA_ARGS__)
#define LOG_INFO(...) Logger::GetInstance().Log(Logger::Level::INFO_LVL, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::GetInstance().Log(Logger::Level::DEBUG_LVL, __VA_ARGS__)
