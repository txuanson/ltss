CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -I. -Isrc -std=c++17 -D_CRT_SECURE_NO_WARNINGS
CC = gcc
CFLAGS = -O3 -Wall -Wextra -I. -Isrc -std=c11 -D_CRT_SECURE_NO_WARNINGS
LDFLAGS = -lm

SRC_C = src/common.c \
        src/stages/s1_preprocess.c \
        src/stages/s2_keypoints.c \
        src/stages/s3_descriptors.c \
        src/stages/s4_matching.c \
        src/stages/s5_homography.c \
        src/stages/s6_warp_blend.c

SRC_CPP = src/main.cpp

# Platform-specific target and commands
ifeq ($(OS),Windows_NT)
    TARGET = stitch.exe
    CXXFLAGS += -static
    RM = del /Q /F
else
    TARGET = stitch
    RM = rm -f
endif

all: $(TARGET)

$(TARGET): $(SRC_C) $(SRC_CPP)
	$(CXX) $(CXXFLAGS) $(SRC_C) $(SRC_CPP) -o $(TARGET) $(LDFLAGS)

clean:
	-$(RM) $(TARGET)
