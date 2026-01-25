# BipartiteGEMM

 High throughput data-parallel GEMM implementations in Cuda using Cuda cores and
 Tensor cores.

# Prerequisites
You should have a CUDA environment installed with a GPU of compute
capability 7.5 or higher. Our code has been tested on both the Turing and Ampere
architectures.

If using the included benchmark/verification program, OpenBLAS is helpful for checking correctness of output, but not necessary.
See the 'Build' section for more details.

# Build
A makefile is provided in  the top-level directory which handles building the application. 
The default target is `sm_75` (eg, Compute Capability 7.5 graphics cards such as the Turing T4).

```bash
$ make
```
is sufficient to build the library with default options.

## Build options
To specify the `arch` string for your target, use the `TARGET` variable when
calling `make`. For example, for a Compute Capability 8.6 graphics card such as the RTX 3060, use:
```bash
$ make TARGET=sm_86
```

To configure the tests ran by `bench`, the following options may also be passed:

 * `NUM_SMS=n` For accurate GFLOPs per SM calculations, this defines the number
    of SMs to consider. default=28
 * `TEST_N=n` Input size to use for test calculations. Note that this may be
   padded as required by the individual library functions. default=4096
 * `TEST_MAX_ELEMENT=n` For library functions that operate on numerical data,
   specifies the maximum element to (randomly) generate for test data.
   default=1
 * `CXXVERSION=20`: if compiling with CUDA < 12, you can set this to `17` to
   compile using only `C++17` features.

The following options may also be specified to configure use of the OpenBLAS
library in `bench`:

* `USE_OPENBLAS=yes`: By default, the OpenBLAS library is called to
  perform a CPU matrix multiplication to serve as a baseline to check for
  correctness. If you do not have OpenBLAS on your system, set this variable
  to `no` to use a naive provided n^3 CPU implementation.
* `OPENBLAS_NUM_THREADS=$(nproc)`: If `USE_OPENBLAS=yes`, this option can be
  specified to reduce the number of threads used by OpenBLAS.

A clean is required before switching configurations.

# Install & run
Simply run `make install PREFIX=/your/sysroot/dir` to install the library for
use in other applications. The interface is provided within the `bipartite` namespace.

A `bench` executable is generated in the `build` directory for testing and verification purposes. 
Simply run this file to both verify output correctness and benchmark
the library - it does not need to be installed to run. Note that the test parameters must be passed at compile time using
the options specified in the `build` section of this document - this is to
ensure best performance possible through hte use of compile-time constants.

For debugging and output inspection, `bench` provides a `-p` argument. When this
argument is provided, both the expected and actual outputs is printed.

**PRECISION NOTE:** We provide an FP16 version of our tensor matrix multiplication,
but since FP16 only has a 10-bit mantissa, it can be quite inaccurate for larger
matrices (or matrices with large values). We use a fixed epsilon (currently set
to `0.00001`) for output checking. This works on the provided example, but if the
parameters in `main.cu` are changed, **the test may report a fail for the FP16 runs
even though the output is as accurate as practical for FP16**.
