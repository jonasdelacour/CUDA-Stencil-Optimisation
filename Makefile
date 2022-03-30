CXX := g++
NVCC := nvcc
#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3

# Add profiling to code
#OPT=-O1 -pg

# Faster compilation time
#OPT=-O1

#INCLUDE := -I../include

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++14

NVCCFLAGS := --restrict --ptxas-options=-v  -arch= sm_80 -lineinfo -maxrregcount=32 -use_fast_math -lineinfo

ACC := -acc=gpu -Minfo=acc


.PHONY: clean all

all: sequential naive cache cooperative async invert alignment pipeline vectorize double_buffering acc

sequential: seq.cpp
	$(CXX) $(CXXFLAGS) seq.cpp -o sequential

naive: naive.cu
	$(NVCC) $(NVCCFLAGS) naive.cu -o naive

vectorize: vectorize.cu
	$(NVCC) $(NVCCFLAGS) vectorize.cu -o vectorize

cache: cache.cu
	$(NVCC) $(NVCCFLAGS) cache.cu -o cache

cooperative: cooperative.cu
	$(NVCC) $(NVCCFLAGS) cooperative.cu -o cooperative

async: async.cu
	$(NVCC) $(NVCCFLAGS) async.cu -o async

invert: invert.cu
	$(NVCC) $(NVCCFLAGS) invert.cu -o invert

alignment: alignment.cu
	$(NVCC) $(NVCCFLAGS) alignment.cu -o alignment

pipeline: pipeline.cu
	$(NVCC) $(NVCCFLAGS) pipeline.cu -o pipeline
    
double_buffering: double_buffering.cu
	$(NVCC) $(NVCCFLAGS) double_buffering.cu -o double_buffering

acc: acc.cpp
	$(CXX) $(CXXFLAGS) $(ACC) acc.cpp -o acc

clean:
	rm -f seq sequential naive vectorize cache cooperative async invert alignment pipeline double_buffering acc *.data
