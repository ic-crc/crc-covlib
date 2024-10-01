# Configuration: release or debug
CONFIG ?= release

CXX = g++
CXXFLAGS = -std=c++17 -fPIC -DP528LIB_EXPORTS=1
#CXXFLAGS += -Wall
LDFLAGS = -shared
ifeq ($(CONFIG), debug)
	CXXFLAGS += -g
	OUT_DIR = build/debug
else
	CXXFLAGS += -O2
	OUT_DIR = build/release
endif

ifeq ($(OS), Windows_NT)
	LDFLAGS += -static
	DYN_LIB_PATHNAME = $(OUT_DIR)/iturp528.dll
else
	DYN_LIB_PATHNAME = $(OUT_DIR)/iturp528.so
endif

P528_SRC_DIR = src/p528
P676_SRC_DIR = src/p676
P835_SRC_DIR = src/p835
SRCS = $(wildcard $(P528_SRC_DIR)/*.cpp) $(wildcard $(P676_SRC_DIR)/*.cpp) $(wildcard $(P835_SRC_DIR)/*.cpp) iturp528.cpp
OBJS = $(addprefix $(OUT_DIR)/, $(SRCS:.cpp=.o))


default: build

build: prep $(DYN_LIB_PATHNAME) copyfiles

$(DYN_LIB_PATHNAME): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(DYN_LIB_PATHNAME) $^

$(OUT_DIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

prep:
	mkdir -p $(OUT_DIR)/$(P528_SRC_DIR)
	mkdir -p $(OUT_DIR)/$(P676_SRC_DIR)
	mkdir -p $(OUT_DIR)/$(P835_SRC_DIR)

rebuild: clean build

clean:
	@rm -f $(DYN_LIB_PATHNAME) $(OBJS)

copyfiles:
	cp -f $(DYN_LIB_PATHNAME) ../../python-wrapper/crc_covlib/helper/bin_64bit/