# Configuration: release or debug
CONFIG ?= release

APP_NAME = longley-rice-test
LDFLAGS =
LDLIBS = -lstdc++fs

ITM_SRC_DIR = ../../src/ntia_itm/src
SRCS = $(wildcard $(ITM_SRC_DIR)/*.cpp) ./$(APP_NAME).cpp
OBJS = $(notdir $(SRCS:.cpp=.o))

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

build: ./$(APP_NAME).$(EXEC_EXT)

./$(APP_NAME).$(EXEC_EXT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(APP_NAME).$(EXEC_EXT) $^ $(LDLIBS)

./%.o: $(ITM_SRC_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

run:
	./$(APP_NAME).$(EXEC_EXT)

rebuild: clean build

clean:
	rm -f *.o *.exe *.out
