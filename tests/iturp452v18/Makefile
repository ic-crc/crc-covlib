# Configuration: release or debug
CONFIG ?= release

APP_NAME = p452v18-test
SRC_DIR = ../../src/
LDFLAGS =
LDLIBS = -lstdc++fs

ifeq ($(CONFIG), debug)
	CXXFLAGS = -std=c++17 -g
else
	CXXFLAGS = -std=c++17 -O3
endif
CXXFLAGS += -Wall -Wextra -Wpedantic -Wshadow -Wunreachable-code -Wsign-conversion

ifeq ($(OS), Windows_NT)
	EXEC_EXT = exe
else
	EXEC_EXT = out
endif


default: build

build:
	g++ -c $(CXXFLAGS) -o ITURP_DigitalMaps.o $(SRC_DIR)ITURP_DigitalMaps.cpp
	g++ -c $(CXXFLAGS) -o ITURP_2001.o $(SRC_DIR)ITURP_2001.cpp
	g++ -c $(CXXFLAGS) -o ITURP_676.o $(SRC_DIR)ITURP_676.cpp
	g++ -c $(CXXFLAGS) -o ITURP_835.o $(SRC_DIR)ITURP_835.cpp
	g++ -c $(CXXFLAGS) -o ITURP_452_1812_common.o $(SRC_DIR)ITURP_452_1812_common.cpp
	g++ -c $(CXXFLAGS) -o ITURP_452_common.o $(SRC_DIR)ITURP_452_common.cpp
	g++ -c $(CXXFLAGS) -o ITURP_452_v18.o $(SRC_DIR)ITURP_452_v18.cpp
	g++ -c $(CXXFLAGS) -o $(APP_NAME).o $(APP_NAME).cpp
	g++ $(LDFLAGS) -o $(APP_NAME).$(EXEC_EXT) ITURP_DigitalMaps.o ITURP_2001.o ITURP_676.o \
	ITURP_835.o ITURP_452_1812_common.o ITURP_452_common.o ITURP_452_v18.o $(APP_NAME).o $(LDLIBS)

run:
	./$(APP_NAME).$(EXEC_EXT)

rebuild: clean build

clean:
	rm -f *.o *.exe *.out
