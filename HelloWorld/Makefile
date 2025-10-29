# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: hello_world

hello_world.o: hello_world.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

hello_world: hello_world.o
	$(NVXX) hello_world.o -o hello_world

clean:
	rm -f hello_world
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
