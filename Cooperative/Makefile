# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: printRank cg_reduce_block cg_reduce_warp cg_tile_reduce

printRank.o: printRank.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

printRank: printRank.o
	$(NVXX) printRank.o -o printRank

cg_reduce_block.o: cg_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) -DBLOCK_BASIS $< -c -o $@

cg_reduce_block: cg_reduce_block.o
	$(NVXX) cg_reduce_block.o -o cg_reduce_block

cg_reduce_warp.o: cg_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) -DWARP_BASIS $< -c -o $@

cg_reduce_warp: cg_reduce_warp.o
	$(NVXX) cg_reduce_warp.o -o cg_reduce_warp

cg_tile_reduce.o: cg_tile_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

cg_tile_reduce: cg_tile_reduce.o
	$(NVXX) cg_tile_reduce.o -o cg_tile_reduce

clean:
	rm -f printRank
	rm -f cg_reduce_block
	rm -f cg_reduce_warp
	rm -f cg_tile_reduce
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
