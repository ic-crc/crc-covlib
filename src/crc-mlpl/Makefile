# NOTE: requires frugally-deep's include/fdeep/ directory to present along with its
#       dependencies (FunctionalPlus, Eigen and json). See:
# https://github.com/Dobiasd/frugally-deep/tree/master/include/fdeep
# https://github.com/Dobiasd/FunctionalPlus/tree/master/include/fplus
# https://gitlab.com/libeigen/eigen/-/tree/master/Eigen?ref_type=heads
# https://github.com/nlohmann/json/tree/develop/include/nlohmann


# Configuration: release or debug
CONFIG ?= release

LIB_NAME = libcrcmlpl
SRC_DIR = ./src/
# Add fdeep/, fplus/, Eigen/ and nlohmann/ directories of header only files in ./deps/include/
# before building
DEPS_INCL_DIR = ./deps/include/
INCL_DIR = ./include/

CXXFLAGS = -std=c++17 -fPIC -Wall -DEIGEN_MPL2_ONLY=1
ifeq ($(CONFIG), debug)
	CXXFLAGS += -g
	OUT_DIR = build/debug/
else
	CXXFLAGS += -O3
	OUT_DIR = build/release/
endif

ifeq ($(OS), Windows_NT)
	LIB_EXT = lib
else
	LIB_EXT = a
endif

STATIC_LIB_PATHNAME = $(OUT_DIR)$(LIB_NAME).$(LIB_EXT)

default: build

build: clean $(STATIC_LIB_PATHNAME) copyfiles

$(STATIC_LIB_PATHNAME):
	mkdir -p $(OUT_DIR)
	g++ -I$(INCL_DIR) -I$(DEPS_INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)crc-mlpl.cpp -o $(OUT_DIR)crc-mlpl.o
	g++ -I$(INCL_DIR) -I$(DEPS_INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)MLPLModel.cpp -o $(OUT_DIR)MLPLModel.o
	g++ -I$(INCL_DIR) -I$(DEPS_INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)MLPLModel-json.cpp -o $(OUT_DIR)MLPLModel-json.o
	ar rcs $(STATIC_LIB_PATHNAME) $(OUT_DIR)crc-mlpl.o $(OUT_DIR)MLPLModel.o $(OUT_DIR)MLPLModel-json.o

rebuild: build

copyfiles:
	mkdir -p ./libs/
	cp -f $(STATIC_LIB_PATHNAME) ./libs/

clean:
	rm -f $(OUT_DIR)*.o $(OUT_DIR)*.$(LIB_EXT)
