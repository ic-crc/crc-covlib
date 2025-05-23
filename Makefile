# Configuration: release or debug
CONFIG ?= release

CXX = g++

CXXFLAGS = -std=c++17 -Wall -fPIC -DCRCCOVLIB_EXPORTS=1 -DCRCCOVLIB_WRAP=1
# Report additional compiler errors and warnings
CXXFLAGS += -Wextra -Wpedantic -Wshadow -Wunreachable-code -Wsign-conversion
ifeq ($(CONFIG), debug)
	CXXFLAGS += -g
else
	CXXFLAGS += -O2
endif	

# see https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html
LDFLAGS = -shared

# In Windows, we include third-party libraries into crc-covlib (i.e. they are statically linked).
# However in Linux, those libraries would need to be recompiled with the -fPIC flag for the
# linker to allow this (and to even consider doing this), they are dynamically linked instead.
ifeq ($(OS), Windows_NT)
	LDFLAGS += -static
	LDLIBS = -lGeographicLib -ltiff -ljpeg -lz -ljbig -llzma -ldeflate -lwebp -lzstd -lLerc -lsharpyuv
#	Static lib for CRC-MLPL
	LDLIBS += src/crc-mlpl/libs/libcrcmlpl.lib
#   May be required for <filesystem> (see https://en.cppreference.com/w/cpp/filesystem)
	LDLIBS += -lstdc++fs 
else
#   dynamically linked
	LDLIBS = -lGeographicLib -ltiff
#   Use libGeographic.so if libGeographicLib.so is not available
	LDCONF = $(shell ldconfig -p | grep libGeographicLib)
	ifeq ($(LDCONF), )
		LDLIBS = -lGeographic -ltiff
	endif
	LDLIBS += -lstdc++fs
#	Static lib for CRC-MLPL
	LDLIBS += -Bstatic src/crc-mlpl/libs/libcrcmlpl.a
#	LDLIBS += -Wl,-rpath,'$$ORIGIN'	
endif

SRC_DIR = src
ITM_SRC_DIR = src/ntia_itm/src
EHATA_SRC_DIR = src/ntia_ehata/src
ifeq ($(CONFIG), debug)
	OUT_DIR = build/debug
else
	OUT_DIR = build/release
endif

SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(ITM_SRC_DIR)/*.cpp) $(wildcard $(EHATA_SRC_DIR)/*.cpp)

ifeq ($(OS), Windows_NT)
	DYN_LIB_PATHNAME = $(OUT_DIR)/crc-covlib.dll
	IMPORT_LIB_PATHNAME = $(OUT_DIR)/crc-covlib.lib
	LDFLAGS += -Wl,--out-implib,$(IMPORT_LIB_PATHNAME)
	STATIC_LIB_PATHNAME = 
else
	SRCS := $(filter-out $(SRC_DIR)/dllmain.cpp, $(SRCS))
#	Note: the import library file is not required in Linux as the .so file doubles
#         as both a dynamic library and an import library
	DYN_LIB_PATHNAME = $(OUT_DIR)/libcrc-covlib.so
	STATIC_LIB_PATHNAME = $(OUT_DIR)/libcrc-covlib.a
endif

OBJS = $(addprefix $(OUT_DIR)/, $(SRCS:.cpp=.o))
ifeq ($(OS), Windows_NT)
	OBJS += $(OUT_DIR)/$(SRC_DIR)/verinfo.o
endif


.PHONY: build prep rebuild clean copyfiles

build: prep $(DYN_LIB_PATHNAME) $(STATIC_LIB_PATHNAME) copyfiles

$(STATIC_LIB_PATHNAME): $(OBJS)
	ar rcs $(STATIC_LIB_PATHNAME) $^

$(DYN_LIB_PATHNAME): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(DYN_LIB_PATHNAME) $^ $(LDLIBS)

$(OUT_DIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

prep:
	mkdir -p $(OUT_DIR)/$(SRC_DIR)
	mkdir -p $(OUT_DIR)/$(ITM_SRC_DIR)
	mkdir -p $(OUT_DIR)/$(EHATA_SRC_DIR)
ifeq ($(OS), Windows_NT)
	windres $(SRC_DIR)/verinfo.rc $(OUT_DIR)/$(SRC_DIR)/verinfo.o
endif

rebuild: clean build

clean:
	@rm -f $(DYN_LIB_PATHNAME) $(IMPORT_LIB_PATHNAME) $(STATIC_LIB_PATHNAME) $(OBJS)

ifeq ($(OS), Windows_NT)
copyfiles:
	cp -f $(DYN_LIB_PATHNAME) ./python-wrapper/crc_covlib/bin_64bit/
	cp -f $(DYN_LIB_PATHNAME) ./dist/
	cp -f $(IMPORT_LIB_PATHNAME) ./dist/
	cp -f $(SRC_DIR)/CRC-COVLIB.h ./dist/
else
copyfiles:
	cp -f $(DYN_LIB_PATHNAME) ./python-wrapper/crc_covlib/bin_64bit/
endif
