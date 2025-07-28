# NOTE: requires frugally-deep's include/fdeep/ directory to present along with its
# dependencies (FunctionalPlus, Eigen and json).
#
# See:
#   https://github.com/Dobiasd/frugally-deep/tree/master/include/fdeep
#   https://github.com/Dobiasd/FunctionalPlus/tree/master/include/fplus
#   https://gitlab.com/libeigen/eigen/-/tree/master/Eigen?ref_type=heads
#   https://github.com/nlohmann/json/tree/develop/include/nlohmann
#
# In other words, add the fdeep/, fplus/, Eigen/ and nlohmann/ directories of header
# only files from the links above into ./include/ before building.


# Configuration: release or debug
CONFIG ?= release

LIB_NAME = libcrcml
SRC_DIR = ./src/
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
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)crc-ml.cpp -o $(OUT_DIR)crc-ml.o
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)FrugallyDeepModel.cpp -o $(OUT_DIR)FrugallyDeepModel.o
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)MLPLModel.cpp -o $(OUT_DIR)MLPLModel.o
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)MLPLModel-json.cpp -o $(OUT_DIR)MLPLModel-json.o
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)PathObscuraModel.cpp -o $(OUT_DIR)PathObscuraModel.o
	g++ -I$(INCL_DIR) -c $(CXXFLAGS) $(SRC_DIR)PathObscuraModel-json.cpp -o $(OUT_DIR)PathObscuraModel-json.o
	ar rcs $(STATIC_LIB_PATHNAME) $(OUT_DIR)crc-ml.o $(OUT_DIR)FrugallyDeepModel.o \
	                              $(OUT_DIR)MLPLModel.o $(OUT_DIR)MLPLModel-json.o \
	                              $(OUT_DIR)PathObscuraModel.o $(OUT_DIR)PathObscuraModel-json.o

rebuild: build

copyfiles:
	mkdir -p ./libs/
	cp -f $(STATIC_LIB_PATHNAME) ./libs/

clean:
	rm -f $(OUT_DIR)*.o $(OUT_DIR)*.$(LIB_EXT)
