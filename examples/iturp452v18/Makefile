# make [CONFIG={debug|release}] : build crc-covlib and example program
# make rebuild [CONFIG={debug|release}] : rebuild crc-covlib and example program
# make run : run the example program
# make clean : delete generated files

# Configuration: release or debug
CONFIG ?= release

APP_NAME = iturp452v18

# Note: you can statically link those libraries below to avoid having to
#       distribute some dlls to run the exe on non-development machines
#LDFLAGS = -static-libgcc -static-libstdc++
LDFLAGS =

ifeq ($(CONFIG), debug)
	COVLIB_DIR = ../../build/debug/
	CXXFLAGS = -g -Wall
else
	COVLIB_DIR = ../../build/release/
	CXXFLAGS = -O2 -Wall
endif

ifeq ($(OS), Windows_NT)
	EXEC_EXT = exe
	COVLIB_FILENAME = crc-covlib.dll
else
	EXEC_EXT = out
	COVLIB_FILENAME = libcrc-covlib.so
endif


default: covlib_build example_build

rebuild: covlib_rebuild example_build

example_build:
	g++ -c $(CXXFLAGS) -o $(APP_NAME).o $(APP_NAME).cpp
	g++ $(LDFLAGS) -o $(APP_NAME).$(EXEC_EXT) $(APP_NAME).o -L$(COVLIB_DIR) -l:$(COVLIB_FILENAME) -Wl,-R.
	cp -f $(COVLIB_DIR)$(COVLIB_FILENAME) ./$(COVLIB_FILENAME)

covlib_build:
	$(MAKE) -C ../../ build CONFIG=$(CONFIG)

covlib_rebuild:
	$(MAKE) -C ../../ rebuild CONFIG=$(CONFIG)

run:
	./$(APP_NAME).$(EXEC_EXT)

clean:
	rm -f *.o *.exe *.dll *.out *.so *.txt *.kml *.mif *.mid *.bil *.hdr *.prj *.stx *.aux.xml