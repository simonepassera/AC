# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: naive_mm tiled_mm

naive_mm.o: naive_mm.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

naive_mm: naive_mm.o
	$(NVXX) naive_mm.o -o naive_mm

tiled_mm.o: tiled_mm.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tiled_mm: tiled_mm.o
	$(NVXX) tiled_mm.o -o tiled_mm

clean:
	rm -f tiled_mm
	rm *.o

.DEFAULT_GOAL := all
.PHONY: all clean
