# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: vector_sum_gpu vector_sum_cpu

vector_sum_gpu.o: vector_sum_gpu.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

vector_sum_gpu: vector_sum_gpu.o
	$(NVXX) vector_sum_gpu.o -o vector_sum_gpu

vector_sum_cpu.o: vector_sum_cpu.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

vector_sum_cpu: vector_sum_cpu.o
	$(NVXX) vector_sum_cpu.o -o vector_sum_cpu

clean:
	rm -f vector_sum_gpu
	rm -f vector_sum_cpu
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
