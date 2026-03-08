#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <bipartite/GEMM.h>
#include <iterator> // std::ostream_iterator
#include <cstdlib> // EXIT_SUCCESS
#include <vector>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

#include "bench/gemm_experiment.h"

using namespace bipartite;

/** 
 * main
 * @brief Performs GEMM on GPU CUDA Cores and Tensor Cores.
 */
int main(int argc, char **argv)
{   
    bool print_result = false;
    const char *mat_file_a = nullptr;
    const char *mat_file_b = nullptr;

    // Parse arguments: optional -p flag, optional two matrix file paths
    int arg_idx = 1;
    if (arg_idx < argc && !strncmp(argv[arg_idx], "-p", 3)) {
        print_result = true;
        arg_idx++;
    }
    if (arg_idx + 1 < argc) {
        mat_file_a = argv[arg_idx];
        mat_file_b = argv[arg_idx + 1];
    }

    constexpr int multiple = 128;

    // Create a random device
    std::random_device rd;
    // Generate a random seed to ensure all experiments use the same random values
    unsigned int seed = rd();

    /*
    ****************************
    * CUDA Core GEMM Experiment
    ****************************
    */

    GemmExperiment<uint32_t, uint32_t> cudaCoreExp{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result};
    cudaCoreExp.run_experiment( 
        [&cudaCoreExp] (uint32_t* a, uint32_t* b, uint32_t* c) {
            uint32_t block_dim_sz = (uint32_t)(cudaCoreExp.get_n() / WARP_SZ);
            const dim3 gridDim{block_dim_sz, block_dim_sz, block_dim_sz};
            const dim3 blockDim{WARP_SZ, WARP_SZ, 1};
            cudacores::matrix_mult<<< gridDim, blockDim >>>(a, b, c, cudaCoreExp.get_n());
            }, "CUDA Core GEMM Implementation" );

    /*
    **********************************
    * Tensor Core FP32 GEMM Experiment
    **********************************
    */
    
    GemmExperiment<half, float> tensorCoreExpFp32{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result};
    tensorCoreExpFp32.run_experiment(
        [&tensorCoreExpFp32] (half* a, half* b, float* c) {
            const dim3 blockDim { WARP_SZ * 8, 4, 1 };
            dim3 gridDim;
            gridDim.x = (tensorCoreExpFp32.get_n() + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
            gridDim.y = (tensorCoreExpFp32.get_n() + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
            tensorcores::gemm<half, float><<< gridDim, blockDim >>>(a, b, c, tensorCoreExpFp32.get_n());
            }, "Tensor Core GEMM FP32 Implementation", 10, 128 );

    /*
    **********************************
    * Tensor Core FP32 (multi-stream) GEMM Experiment
    **********************************
    */
    std::size_t superblock_sz = 128;
    GemmExperiment<half, float> tensorCoreExpFp32Streams{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result, superblock_sz};
    std::size_t padded_n = tensorCoreExpFp32Streams.get_n();
    tensorCoreExpFp32Streams.run_experiment_streams( 
        [&tensorCoreExpFp32Streams, padded_n, superblock_sz] (half *a, half *b, float *c, cudaStream_t stream) {
            const dim3 blockDim { WARP_SZ * 8, 4, 1 };
            dim3 gridDim;
            gridDim.x = (padded_n + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
            gridDim.y = (superblock_sz + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
            tensorcores::gemm<half, float><<< gridDim, blockDim, 0, stream >>>(a, b, c, padded_n, superblock_sz);
            }, "Tensor Core GEMM FP32 (two streams) Implementation", 1, 128 );

    /*
    **********************************
    * 2:4 SpMM Experiment (file-loaded)
    **********************************
    */
    if (mat_file_a && mat_file_b) {
        GemmExperiment<half, float> spmmExp{mat_file_a, mat_file_b, (std::size_t)multiple, print_result};
        std::size_t spmm_n = spmmExp.get_n();
        spmmExp.run_experiment_sparse(
            [spmm_n] (half* a, half* b, float* c, uint32_t* idx, std::size_t k_offset) {
                const dim3 blockDim { WARP_SZ * 8, 4, 1 };
                dim3 gridDim;
                gridDim.x = (spmm_n + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
                gridDim.y = (spmm_n + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
                tensorcores::spmm_24<half, float><<<gridDim, blockDim>>>(a, b, c, idx, spmm_n, 0, k_offset);
            }, "2:4 SpMM Implementation", 10, 128);
    }

    return 0;
    /*
    **********************************
    * Tensor Core FP16 GEMM Experiment
    **********************************
    */

    GemmExperiment<half, half> tensorCoreExpFp16{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result};
    padded_n = tensorCoreExpFp16.get_n();
    tensorCoreExpFp16.run_experiment( 
        [&tensorCoreExpFp16, padded_n] (half* a, half* b, half* c) {
            const dim3 blockDim { WARP_SZ * 4, 4, 1 };
            dim3 gridDim;
            gridDim.x = (padded_n + (WMMA_M * blockDim.x / WARP_SZ - 1)) / (WMMA_M * blockDim.x / WARP_SZ);
            gridDim.y = (padded_n + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
            tensorcores::gemm<half, half><<< gridDim, blockDim >>>(a, b, c, padded_n);
            }, "Tensor Core GEMM FP16 Implementation", 10 );


    /*
    **********************************
    * Tensor Core FP16 (multi-stream) GEMM Experiment
    **********************************
    */

    superblock_sz = 128;
    GemmExperiment<half, half> tensorCoreExpFp16Streams{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result, superblock_sz};
    padded_n = tensorCoreExpFp16Streams.get_n();
    tensorCoreExpFp16Streams.run_experiment_streams( 
        [&tensorCoreExpFp16Streams, padded_n, superblock_sz] (half *a, half *b, half *c, cudaStream_t stream) {
            const dim3 blockDim { WARP_SZ * 4, 4, 1 };
            dim3 gridDim;
            gridDim.x = (padded_n + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
            gridDim.y = (superblock_sz + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
            tensorcores::gemm<half, half><<< gridDim, blockDim, 0, stream >>>(a, b, c, padded_n, superblock_sz);
            }, "Tensor Core GEMM FP16 (two streams) Implementation", 10 );

    /*
    **********************************
    * Tensor Core INT8 GEMM Experiment
    **********************************
    */

    GemmExperiment<unsigned char, int> tensorCoreExpInt8{TEST_N, TEST_MAX_ELEMENT, multiple, seed, print_result};
    tensorCoreExpInt8.run_experiment(
        [&tensorCoreExpInt8] (unsigned char *a, unsigned char *b, int *c) {
            const dim3 blockDim { WARP_SZ * 4, 4, 1 };
            dim3 gridDim;
            gridDim.x = (tensorCoreExpInt8.get_n() + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
            gridDim.y = (tensorCoreExpInt8.get_n() + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
            tensorcores::gemm<unsigned char, int><<< gridDim, blockDim >>>(a, b, c, tensorCoreExpInt8.get_n());
            }, "Tensor Core GEMM INT8 Implementation", 10 );

    return EXIT_SUCCESS;
}
