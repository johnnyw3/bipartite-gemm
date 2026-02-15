CXXVERSION?=20
INSTALL?=install

CXXFLAGS=-O3 -std=c++$(CXXVERSION) -g -I.

ifeq ($(USE_OPENBLAS),no)
CXXFLAGS+= -DNO_OPENBLAS
else
CXXFLAGS+= -lopenblas
endif

# set up thread count for multithreaded CPU matrix multiply
OPENBLAS_NUM_THREADS?="$$(nproc)"

# Test options
TEST_N?=4096
TEST_MAX_ELEMENT?=1
# Will vary by GPU, so users should specify at the command line
NUM_SMS?=28


# Default value for Turing. Use TARGET=sm_86 for Ampere
TARGET?=sm_75

all: build/bench

build/bench: bench/main.cu bench/gemm_experiment.h bipartite-gemm/cuda_common.h bipartite-gemm/GEMM.h
	@mkdir -p build
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) nvcc -o build/bench bench/main.cu \
    -arch=$(TARGET) -DNUM_SMS=$(NUM_SMS) -DTEST_N=$(TEST_N) \
    -DTEST_MAX_ELEMENT=$(TEST_MAX_ELEMENT) $(CXXFLAGS)

install:
	$(INSTALL) -CD -m 644 bipartite-gemm/GEMM.h $(PREFIX)/usr/include/bipartite/GEMM.h
	$(INSTALL) -CD -m 644 bipartite-gemm/cuda_common.h $(PREFIX)/usr/include/bipartite/cuda_common.h

clean:
	rm -f build/*
