#include <cstddef>
#include <cassert>
#include <mma.h>
#include <stdint.h>

#define WARP_SZ 32
#define WMMA_M  16
#define WMMA_K  16
#define WMMA_N  16

using namespace nvcuda;

namespace bipartite{
namespace tensorcores{


/** gemm
  * @brief perform a gemm on two matricies (A*B) of type I using tensor wmma
  *        instructions, saving the results in the type R matrix
  * @param matrix_a - a pointer to the A matrix in device memory
  * @param matrix_b - a pointer to the B matrix in device memory
  * @param res - a pointer to an area of device memory to store the results
  * @pre matrix_a, matrix_b, and res are n x n matricies
  */
template<typename I, typename R>
__global__
void gemm(I *matrix_a, I *matrix_b, R *res, std::size_t n, std::size_t superblock_sz=0)
{

    // Note that threadblocks are a 4x4 2D grid of warps
    std::size_t a_col = 0; 
    const std::size_t a_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_K;

    const std::size_t b_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_K;
    std::size_t b_row = 0;

    const std::size_t c_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_M;
    const std::size_t c_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    // Safe as this will be consistent for an entire kernel launch
    const std::size_t num_rows = (superblock_sz) ? superblock_sz : n; 

    if (a_row >= num_rows || b_col >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, R> acc;
    wmma::fill_fragment(acc, R(0));

    for (std::size_t k = 0; k < n; k += WMMA_K)
    {
        a_col = k;
        b_row = k;
        wmma::load_matrix_sync(afrag, matrix_a + a_row * n + a_col, n);
        wmma::load_matrix_sync(bfrag, matrix_b + b_row * n + b_col, n);
        wmma::mma_sync(acc, afrag, bfrag, acc);
    }

    wmma::store_matrix_sync(res + c_row * n + c_col, acc, n, wmma::mem_row_major);
}

/** gemm_wrapper
  * @brief High-level interinterface for gemm designed to be called from general-purpose code
  */
template<typename I, typename R>
void gemm_wrapper(I *matrix_a, I *matrix_b, R *res, std::size_t n,
                  std::size_t superblock_sz=0, std::size_t first_superblock=0)
{
    I* d_matrix_a;
    I* h_matrix_a;
    I* d_matrix_b;
    R* d_matrix_c;
    R* h_matrix_c;

    size_t c_size = sizeof( R ) * (n*n - n*superblock_sz*first_superblock);
    size_t a_size = sizeof( I ) * (n*n - n*superblock_sz*first_superblock);

    if (a_size == 0)
        return;

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    cudaMalloc( &d_matrix_a, sizeof( I ) * 2 * superblock_sz * n );
    cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
    cudaMalloc( &d_matrix_c, sizeof( R ) * 2 * superblock_sz * n);

    // Create pinned memory buffers for matricies we will be accessing
    // multiple times
    cudaMallocHost((void**) &h_matrix_c, c_size);
    h_matrix_a = (I*)((uint8_t*)h_matrix_c + c_size - a_size); 

    // Copy b to device using the pinned buffer we created for a
    // (needs to be done first since b is row-major)
    memcpy(h_matrix_c, matrix_b,  sizeof( I ) * n * n);
    cudaMemcpy( d_matrix_b, h_matrix_c,  sizeof( I ) * n * n, cudaMemcpyHostToDevice );

    // Now we can actually use a's pinned buffer for a
    memcpy(h_matrix_a, matrix_a + n*superblock_sz*first_superblock, a_size);

    assert( n % superblock_sz ==0 && "superblock_sz must be a factor of n" );

    // i+=2 because two superblocks are computed in separate streams concurrently
    for (std::size_t i = first_superblock; i < n/superblock_sz; i+=2)
    {
      cudaMemcpyAsync( d_matrix_a, h_matrix_a+superblock_sz*(i-first_superblock)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[0] );
      cudaMemcpyAsync( d_matrix_a + superblock_sz*n, h_matrix_a+superblock_sz*(i-first_superblock+1)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[1] );

      const dim3 blockDim { WARP_SZ * 4, 4, 1 };
      dim3 gridDim;
      gridDim.x = (n + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
      gridDim.y = (superblock_sz + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);
      gemm<I, R><<< gridDim, blockDim, 0, streams[0] >>>(d_matrix_a, d_matrix_b, d_matrix_c, n, superblock_sz);
      gemm<I, R><<< gridDim, blockDim, 0, streams[1] >>>(d_matrix_a+superblock_sz*n, d_matrix_b, d_matrix_c+superblock_sz*n, n, superblock_sz);

      cudaMemcpyAsync(  h_matrix_c + superblock_sz*(i-first_superblock)*n, d_matrix_c, sizeof( R ) * superblock_sz*n, cudaMemcpyDeviceToHost, streams[0] );
      cudaMemcpyAsync(  h_matrix_c + superblock_sz*(i-first_superblock+1)*n, d_matrix_c + superblock_sz*n, sizeof( R ) * superblock_sz*n, cudaMemcpyDeviceToHost, streams[1] );
    }

    cudaDeviceSynchronize();
    memcpy(res+n*superblock_sz*first_superblock, h_matrix_c,  c_size );

    cudaFree( &d_matrix_a );
    cudaFree( &d_matrix_b );
    cudaFree( &d_matrix_c );
    cudaFreeHost( (void*) h_matrix_c );

}

} // namespace tensorcores


namespace cudacores{

/**
 * warp_sum
 * @brief Perform a warp sum reduction using given th_val
 */
__device__
std::size_t warp_sum(std::size_t th_val)
{
  std::size_t th_id = threadIdx.x;
  std::size_t new_val = 0;
  uint32_t shuffle_mask = 0xFFFFFFFF;

  for (std::size_t stride = 1; stride < WARP_SZ; stride <<= 1)
  {
      new_val = __shfl_down_sync(0xFFFFFFFF, th_val, stride);
      // Only add the new value if this thread is in the mask!
      if ((0x1 << th_id) & shuffle_mask){
        th_val += new_val;
      }
      shuffle_mask >>= stride;
  }

  return th_val;

}

/**
  * matrix_mult
  * @brief Compute the partial product of a 32x32 tile of matrix_a and matrix_b, storing results in result matrix.
  * @pre matrix_a, matrix_b, and result have dimensions of n x n
*/
__global__
void matrix_mult( uint32_t* matrix_a, uint32_t* matrix_b, uint32_t* result, std::size_t n)
{
    // Remember: Multiple z dimensions at block-level ONLY

    // A
    std::size_t a_col = blockIdx.z * blockDim.x + threadIdx.x;
    std::size_t a_row = blockIdx.y * blockDim.y + threadIdx.y;

    // B
    std::size_t b_col = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t b_row = blockIdx.z * blockDim.y + threadIdx.y;

    // C
    //std::size_t c_col = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t c_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy tile of B (transposed) into smem
    __shared__ uint32_t smem[1024];
    smem[(threadIdx.x * blockDim.x) + threadIdx.y ] = matrix_b[(b_row * n) + b_col];
    __syncthreads();

    // Each thread performs calculations for a fixed a value, retrieve it here
    std::size_t a_val = matrix_a[(a_row * n) + a_col];

    for (std::size_t b_tile_col = 0; b_tile_col < blockDim.x; b_tile_col++)
    {
      // Perform single cell product of a and b for thread
      std::size_t product =  a_val * smem[(b_tile_col * blockDim.x) + threadIdx.x];

      // Make sure that all accesses to smem are complete before we perform warp_sum
      __syncwarp();

      // Use warp primitives to add
      std::size_t dot_product = warp_sum(product);
      if (!threadIdx.x)
        atomicAdd(result + (c_row * n) + (blockIdx.x * blockDim.x) + b_tile_col, dot_product);
    }

    return;
}

} // namespace cudacores
} // namespace bipartite
