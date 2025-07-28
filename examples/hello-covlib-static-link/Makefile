# make [CONFIG={debug|release}] : build crc-covlib and example program
# make rebuild [CONFIG={debug|release}] : rebuild crc-covlib and example program
# make run : run the example program
# make clean : delete generated files

# Configuration: release or debug
CONFIG ?= release

APP_NAME = hello-covlib-static-link

# To statically link to system libraries:
#LDFLAGS = -static
# Otherwise:
LDFLAGS =

ifeq ($(CONFIG), debug)
	COVLIB_DIR = ../../build/debug/
	CXXFLAGS = -g
else
	COVLIB_DIR = ../../build/release/
	CXXFLAGS = -O2
endif

ifeq ($(OS), Windows_NT)
$(error No static version of crc-covlib is currently available on Windows, please use the DLL instead.)
#	EXEC_EXT = exe
#	COVLIB_FILENAME =
#	LDLIBS = 
else
	EXEC_EXT = out
	COVLIB_FILENAME = libcrc-covlib.a

#   Need to link to third-party libraries used by crc-covlib. They can either be
#   linked statically or dynamically.

#	For dynamic linking (unless -static is used):
#	LDLIBS = -lGeographicLib -ltiff

#	For static linking (some development files packages may need to be installed first, like libwebp-dev, libzstd-dev, etc.):
	LDLIBS = -l:libGeographicLib.a -l:libtiff.a -l:libz.a -l:libjpeg.a -l:libjbig.a -l:liblzma.a -l:libwebp.a -l:libzstd.a -l:libdeflate.a -l:libLerc.a -l:libsharpyuv.a

#    Or use this instead on older systems:
#    LDLIBS = -l:libGeographic.a -l:libtiff.a -l:libz.a -l:libjpeg.a -l:libjbig.a -l:liblzma.a -l:libwebp.a -l:libzstd.a -l:libdeflate.a

	LDLIBS += ../../src/crc-ml/libs/libcrcml.a
endif


default: covlib_build example_build

rebuild: covlib_rebuild example_build

example_build:
	g++ -c $(CXXFLAGS) -o $(APP_NAME).o $(APP_NAME).cpp
	g++ -o $(APP_NAME).$(EXEC_EXT) $(APP_NAME).o -L$(COVLIB_DIR) $(LDFLAGS) -l:$(COVLIB_FILENAME) $(LDLIBS)

covlib_build:
	$(MAKE) -C ../../ build CONFIG=$(CONFIG)

covlib_rebuild:
	$(MAKE) -C ../../ rebuild CONFIG=$(CONFIG)

run:
	./$(APP_NAME).$(EXEC_EXT)

clean:
	rm -f *.o *.out