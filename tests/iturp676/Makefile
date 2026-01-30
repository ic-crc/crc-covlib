# Configuration: release or debug
CONFIG ?= release

APP_NAME = p676-test
SRC_DIR = ../../src/
LDFLAGS =

ifeq ($(CONFIG), debug)
	CXXFLAGS = -std=c++17 -g
else
	CXXFLAGS = -std=c++17 -O3
endif

ifeq ($(OS), Windows_NT)
	EXEC_EXT = exe
else
	EXEC_EXT = out
endif


default: build

build:
	g++ -c $(CXXFLAGS) -o ITURP_676.o $(SRC_DIR)ITURP_676.cpp
	g++ -c $(CXXFLAGS) -o $(APP_NAME).o $(APP_NAME).cpp
	g++ $(LDFLAGS) -o $(APP_NAME).$(EXEC_EXT) ITURP_676.o $(APP_NAME).o

run:
	./$(APP_NAME).$(EXEC_EXT)

rebuild: clean build

clean:
	rm -f *.o *.exe *.out
