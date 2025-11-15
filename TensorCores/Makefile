# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80
NVLDLIBRARIES   = -lcublas

all: test_intrinsics tensor_example_1 tensor_example_2 tensor_mm

test_intrinsics.o: test_intrinsics.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

test_intrinsics: test_intrinsics.o
	$(NVXX) test_intrinsics.o -o test_intrinsics

tensor_example_1.o: tensor_example_1.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tensor_example_1: tensor_example_1.o
	$(NVXX) tensor_example_1.o -o tensor_example_1

tensor_example_2.o: tensor_example_2.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tensor_example_2: tensor_example_2.o
	$(NVXX) tensor_example_2.o -o tensor_example_2

tensor_mm.o: tensor_mm.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tensor_mm: tensor_mm.o
	$(NVXX) tensor_mm.o -o tensor_mm

clean:
	rm -f test_intrinsics
	rm -f tensor_example_1
	rm -f tensor_example_2
	rm -f tensor_mm
	rm *.o

.DEFAULT_GOAL := all
.PHONY: all clean
