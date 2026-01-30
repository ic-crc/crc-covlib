# make [CONFIG={debug|release}] : build test program
# make rebuild [CONFIG={debug|release}] : rebuild test program
# make run : run the test program
# make clean : delete generated files

# Configuration: release or debug
CONFIG ?= release

APP_NAME = crc-mlpl-test
SRC_DIR = ./
LDFLAGS =

# directory that contains crc-ml.h
INCL_DIR = ../../src/crc-ml/include/

# directory containing the static lib file for CRC-MLPL
LIB_DIR = ../../src/crc-ml/libs/


ifeq ($(CONFIG), debug)
	CXXFLAGS = -std=c++17 -g
else
	CXXFLAGS = -std=c++17 -O3
endif

ifeq ($(OS), Windows_NT)
	EXEC_EXT = exe
	LDLIBS = $(LIB_DIR)libcrcml.lib
else
	EXEC_EXT = out
	LDLIBS = $(LIB_DIR)libcrcml.a
endif


default: build

build:
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) -o crc-mlpl-test.o $(SRC_DIR)crc-mlpl-test.cpp
	g++ $(LDFLAGS) -o $(APP_NAME).$(EXEC_EXT) crc-mlpl-test.o $(LDLIBS)

run:
	./$(APP_NAME).$(EXEC_EXT)

rebuild: clean build

clean:
	rm -f *.o *.exe *.out