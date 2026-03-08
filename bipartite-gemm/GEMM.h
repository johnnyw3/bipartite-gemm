#include <cstddef>
#include <cassert>
#include <mma.h>
#include <stdint.h>

#define WARP_SZ 32
#define WMMA_M  16
#define WMMA_K  16
#define WMMA_N  8

using namespace nvcuda;

namespace bipartite{
namespace tensorcores{

extern "C" __device__ void gemm_helper(uint64_t matrix_a, uint64_t matrix_b,
                                       uint64_t out_addr, uint64_t a_row,
                                       uint64_t b_col, uint32_t n);
extern "C" __device__ void gemm_helper_mma(uint64_t matrix_a, uint64_t matrix_b,
                                       uint64_t out_addr, uint32_t n);
extern "C" __device__ void spmm_helper_mma(uint64_t matrix_a, uint64_t matrix_b,
                                       uint64_t idx_vec,
                                       uint64_t out_addr, uint32_t n);

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
    constexpr std::size_t BLOCK_SIZE_A = 16;
    constexpr std::size_t BLOCK_SIZE_B = 8;
    
    // Declare shared memory arrays with fragment packing layout
    // smem_a stores the A matrix fragment (64 rows x n cols) packed in 16x16 blocks
    // smem_b stores the B matrix fragment (n rows x 64 cols) packed in 8x8 blocks
    //__shared__ I smem_a[WMMA_M*4*n];
    //__shared__ I smem_b[n*WMMA_N*8];
    __shared__ I smem_a[WMMA_M*4*128];
    __shared__ I smem_b[128*WMMA_N*8];

    // Safe as this will be consistent for an entire kernel launch
    const std::size_t num_rows = (superblock_sz) ? superblock_sz : n; 

    // Calculate thread ID and total threads in block
    const std::size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const std::size_t num_threads = blockDim.x * blockDim.y;
    
    // Calculate starting rows/cols for this threadblock
    const std::size_t block_a_row_start = blockIdx.y * blockDim.y * WMMA_M;
    const std::size_t block_b_col_start = blockIdx.x * blockDim.x / WARP_SZ * WMMA_N;
    
    const std::size_t num_a_rows = WMMA_M * 4;  // 64 rows
    const std::size_t num_b_cols = WMMA_N * 8;  // 64 cols
    
    // Pack A matrix data into 16x16 blocks stored sequentially
    // A: (WMMA_M*4) rows x n cols = 64 rows x n cols
    const std::size_t num_blocks_a_row = num_a_rows / BLOCK_SIZE_A;  // 4 blocks vertically
    const std::size_t num_blocks_a_col = 128 / BLOCK_SIZE_A;           // n/16 blocks horizontally
    const std::size_t total_a_blocks = num_blocks_a_row * num_blocks_a_col;
    
    for (std::size_t block_idx = tid; block_idx < total_a_blocks; block_idx += num_threads)
    {
        std::size_t block_row = block_idx / num_blocks_a_col;
        std::size_t block_col = block_idx % num_blocks_a_col;
        
        // Each thread copies one 16x16 block
        for (std::size_t i = 0; i < BLOCK_SIZE_A; i++)
        {
            for (std::size_t j = 0; j < BLOCK_SIZE_A; j++)
            {
                std::size_t gmem_row = block_a_row_start + block_row * BLOCK_SIZE_A + i;
                std::size_t gmem_col = block_col * BLOCK_SIZE_A + j;
                std::size_t smem_idx = block_idx * BLOCK_SIZE_A * BLOCK_SIZE_A + i * BLOCK_SIZE_A + j;
                
                if (gmem_row < num_rows && gmem_col < n)
                {
                    //if (i == 2 && j  == 1 && block_col < 2)
                    smem_a[smem_idx] = matrix_a[gmem_row * n + gmem_col];

                }else
                    smem_a[smem_idx] = I(0);
            }
        }
    }

    // Pack B matrix data into 8x8 blocks stored sequentially (transposed)
    // B: n rows x (WMMA_N*4) cols = n rows x 64 cols -> transposed to 64 rows x n cols
    const std::size_t num_blocks_b_row = 128 / BLOCK_SIZE_B;            // n/8 blocks vertically
    const std::size_t num_blocks_b_col = num_b_cols / BLOCK_SIZE_B;  // 8 blocks horizontally
    const std::size_t total_b_blocks = num_blocks_b_row * num_blocks_b_col;
    
    for (std::size_t block_idx = tid; block_idx < total_b_blocks; block_idx += num_threads)
    {
        std::size_t block_row = block_idx / num_blocks_b_col;
        std::size_t block_col = block_idx % num_blocks_b_col;
        
        // Each thread copies one 8x8 block, transposed
        for (std::size_t i = 0; i < BLOCK_SIZE_B; i++)
        {
            for (std::size_t j = 0; j < BLOCK_SIZE_B; j++)
            {
                std::size_t gmem_row = block_row * BLOCK_SIZE_B + i;
                std::size_t gmem_col = block_b_col_start + block_col * BLOCK_SIZE_B + j;
                // Transpose: swap i and j in the smem_idx calculation
                std::size_t smem_idx = block_idx * BLOCK_SIZE_B * BLOCK_SIZE_B + j * BLOCK_SIZE_B + i;
                
                //if (block_idx== 0 && tid  == 0)
                //printf("bidx %d bidy %d tid %ld bid %ld i %ld j %ld smem_idx: %ld, max %d\n", blockIdx.x, blockIdx.y, tid, block_idx, i, j, smem_idx, 4*256*WMMA_M);
                if (gmem_row < n && gmem_col < n)
                    //if (j == 1 )

                    //if (block_row == 2 && i == 0 && j < 4)
                    {
                    smem_b[smem_idx] = matrix_b[gmem_row * n + gmem_col];
                    //else
                    //smem_b[smem_idx] = 0;
                    }
                else
                    smem_b[smem_idx] = I(0);
                //if (j > 3)
                //    smem_b[smem_idx] = 1; //I(0);
                //else
                //    smem_b[smem_idx] = I(0);
                    
            }
        }
    }

    __syncthreads();

    // Note that threadblocks are a 4x8 2D grid of warps
    std::size_t a_col = 0; 
    const std::size_t a_row = (threadIdx.y) * WMMA_M;

    const std::size_t b_col = ((threadIdx.x) / WARP_SZ) * WMMA_N;
    std::size_t b_row = 0;

    const std::size_t c_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_N;
    const std::size_t c_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_M;

    uint32_t warp_x = (threadIdx.x % WARP_SZ) / WMMA_K;
    uint32_t warp_y = (threadIdx.x % WARP_SZ) % WMMA_K;

    //if (a_row >= num_rows || b_col >= n) return;

    I *ap = smem_a + a_row * 128;
    ap += (threadIdx.x % WARP_SZ > 15) ? ( ((threadIdx.x % WARP_SZ) - 16) * BLOCK_SIZE_A + 8) : ((threadIdx.x % WARP_SZ)  * BLOCK_SIZE_A);
    //(warp_x / WMMA_M) * BLOCK_SIZE_A * BLOCK_SIZE_A + (warp_x % WMMA_M) * BLOCK_SIZE_A;
    // B matrix needs to be transposed before we want finish this...
    I *bp = smem_b + b_col * BLOCK_SIZE_B;
    //if (tid == 0)
    //{
    //    for (int idx = 0; idx < 128*WMMA_N*8; ++idx)
    //        printf("%f ", (float)smem_b[idx]);
    //}
    bp += (threadIdx.x % WARP_SZ > 7) ? ( ((threadIdx.x % WARP_SZ) - 8) * BLOCK_SIZE_B + BLOCK_SIZE_B * 64) : ((threadIdx.x % WARP_SZ) * BLOCK_SIZE_B);
    //+ (warp_x / WMMA_N) * BLOCK_SIZE_B * BLOCK_SIZE_B + (warp_x % WMMA_N) * BLOCK_SIZE_B;
    R *cp = res + (c_row + (threadIdx.x % 32) / 4) * n + (c_col + (threadIdx.x % 4)*2);
    //R *cp = res;
    gemm_helper_mma((uint64_t)ap, (uint64_t)bp, (uint64_t)cp, (uint32_t) n);
    //*res= 2.0;
#if 0 
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, R> acc;
    wmma::fill_fragment(acc, R(0));

    for (std::size_t k = 0; k < n; k += WMMA_K)
    {
        a_col = k;
        b_row = k;
        
        // Calculate which 8x8 blocks we need to read from shared memory
        // For A: need to find the block at (threadIdx.y * WMMA_M, k)
        // For B: need to find the block at (k, threadIdx.x / WARP_SZ * WMMA_N)
        
        // Load from shared memory (data is packed in 8x8 blocks)
        // For now, we'll use a temporary buffer to reorganize for WMMA
        // This is a placeholder - actual implementation depends on WMMA layout requirements
        wmma::load_matrix_sync(afrag, matrix_a + a_row * n + a_col, n);
        wmma::load_matrix_sync(bfrag, matrix_b + b_row * n + b_col, n);
        wmma::mma_sync(acc, afrag, bfrag, acc);
    }

    wmma::store_matrix_sync(res + c_row * n + c_col, acc, n, wmma::mem_row_major);
#endif
}

/* 2:4 sparsity SpMM */
template<typename I, typename R>
__global__
void spmm_24(I *matrix_a, I *matrix_b, R *res, uint32_t *idx_mat, std::size_t n, std::size_t superblock_sz=0, std::size_t k_offset=0)
{
    constexpr std::size_t BLOCK_SIZE_A = 8;
    constexpr std::size_t BLOCK_SIZE_B = 8;
    
    // Declare shared memory arrays with fragment packing layout
    // smem_a stores the A matrix fragment (64 rows x n cols) packed in 16x16 blocks
    // smem_b stores the B matrix fragment (n rows x 64 cols) packed in 8x8 blocks
    //__shared__ I smem_a[WMMA_M*4*n];
    //__shared__ I smem_b[n*WMMA_N*8];
    __shared__ I smem_a[WMMA_M*4*128];
    __shared__ I smem_b[128*WMMA_N*8];

    // Safe as this will be consistent for an entire kernel launch
    const std::size_t num_rows = (superblock_sz) ? superblock_sz : n; 

    // Calculate thread ID and total threads in block
    const std::size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const std::size_t num_threads = blockDim.x * blockDim.y;
    
    // Calculate starting rows/cols for this threadblock
    const std::size_t block_a_row_start = blockIdx.y * blockDim.y * WMMA_M;
    const std::size_t block_b_col_start = blockIdx.x * blockDim.x / WARP_SZ * WMMA_N;
    
    const std::size_t num_a_rows = WMMA_M * 4;  // 64 rows
    const std::size_t num_b_cols = WMMA_N * 8;  // 64 cols
    
    // Pack A matrix data into 16x16 blocks stored sequentially
    // A: (WMMA_M*4) rows x n cols = 64 rows x n cols
    const std::size_t num_blocks_a_row = num_a_rows / BLOCK_SIZE_A;  // 4 blocks vertically
    const std::size_t num_blocks_a_col = 128 / BLOCK_SIZE_A;           // n/16 blocks horizontally
    const std::size_t total_a_blocks = num_blocks_a_row * num_blocks_a_col;

    for (std::size_t block_idx = tid; block_idx < total_a_blocks; block_idx += num_threads)
    {
        std::size_t block_row = block_idx / num_blocks_a_col;
        std::size_t block_col = block_idx % num_blocks_a_col;
        
        // Each thread copies one 16x16 block
        for (std::size_t i = 0; i < BLOCK_SIZE_A; i++)
        {
            for (std::size_t j = 0; j < BLOCK_SIZE_A; j++)
            {
                std::size_t gmem_row = block_a_row_start + block_row * BLOCK_SIZE_A + i;
                std::size_t gmem_col = block_col * BLOCK_SIZE_A + j;
                std::size_t smem_idx = block_idx * BLOCK_SIZE_A * BLOCK_SIZE_A + i * BLOCK_SIZE_A + j;
                
                if (gmem_row < num_rows && gmem_col < n)
                {
                    //if (i == 2 && j  == 1 && block_col < 2)
                    smem_a[smem_idx] = matrix_a[gmem_row * n + gmem_col];

                }else
                    smem_a[smem_idx] = I(0);
            }
        }
    }

    // Pack B matrix data into 8x8 blocks stored sequentially (transposed)
    // B: n rows x (WMMA_N*4) cols = n rows x 64 cols -> transposed to 64 rows x n cols
    const std::size_t num_blocks_b_row = 128 / BLOCK_SIZE_B;            // n/8 blocks vertically
    const std::size_t num_blocks_b_col = num_b_cols / BLOCK_SIZE_B;  // 8 blocks horizontally
    const std::size_t total_b_blocks = num_blocks_b_row * num_blocks_b_col;
    
    for (std::size_t block_idx = tid; block_idx < total_b_blocks; block_idx += num_threads)
    {
        std::size_t block_row = block_idx / num_blocks_b_col;
        std::size_t block_col = block_idx % num_blocks_b_col;
        
        // Each thread copies one 8x8 block, transposed
        for (std::size_t i = 0; i < BLOCK_SIZE_B; i++)
        {
            for (std::size_t j = 0; j < BLOCK_SIZE_B; j++)
            {
                std::size_t gmem_row = block_row * BLOCK_SIZE_B + i;
                std::size_t gmem_col = block_b_col_start + block_col * BLOCK_SIZE_B + j;
                // Transpose: swap i and j in the smem_idx calculation
                std::size_t smem_idx = block_idx * BLOCK_SIZE_B * BLOCK_SIZE_B + j * BLOCK_SIZE_B + i;
                
                //if (block_idx== 0 && tid  == 0)
                //printf("bidx %d bidy %d tid %ld bid %ld i %ld j %ld smem_idx: %ld, max %d\n", blockIdx.x, blockIdx.y, tid, block_idx, i, j, smem_idx, 4*256*WMMA_M);
                if (gmem_row < n && gmem_col < n)
                    //if (j == 1 )

                    //if (block_row == 2 && i == 0 && j < 4)
                    {
                    smem_b[smem_idx] = matrix_b[gmem_row * n + gmem_col];
                    //else
                    //smem_b[smem_idx] = 0;
                    }
                else
                    smem_b[smem_idx] = I(0);
                //if (j > 3)
                //    smem_b[smem_idx] = 1; //I(0);
                //else
                //    smem_b[smem_idx] = I(0);
                    
            }
        }
    }


    __syncthreads();

    // Note that threadblocks are a 4x8 2D grid of warps
    std::size_t a_col = 0; 
    const std::size_t a_row = (threadIdx.y) * WMMA_M;

    const std::size_t b_col = ((threadIdx.x) / WARP_SZ) * WMMA_N;
    std::size_t b_row = 0;

    const std::size_t c_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_N;
    const std::size_t c_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_M;

    const std::size_t idx_row = c_row + (threadIdx.x % WARP_SZ) / 4;
    const std::size_t idx_col = k_offset;
    uint32_t *idx_vec = idx_mat + (idx_row*n + idx_col)/32; // 32 = 1/2 as many elements, 1/16 bytes/element vs uint32

    uint32_t warp_x = (threadIdx.x % WARP_SZ) / WMMA_K;
    uint32_t warp_y = (threadIdx.x % WARP_SZ) % WMMA_K;

    //if (a_row >= num_rows || b_col >= n) return;

    I *ap = smem_a + a_row * 128;
    ap += (threadIdx.x % WARP_SZ > 15) ? ( ((threadIdx.x % WARP_SZ) - 16) * BLOCK_SIZE_A + 8) : ((threadIdx.x % WARP_SZ)  * BLOCK_SIZE_A);
    //(warp_x / WMMA_M) * BLOCK_SIZE_A * BLOCK_SIZE_A + (warp_x % WMMA_M) * BLOCK_SIZE_A;
    // B matrix needs to be transposed before we want finish this...
    I *bp = smem_b + b_col * BLOCK_SIZE_B;
    //if (tid == 0)
    //{
    //    for (int idx = 0; idx < 128*WMMA_N*8; ++idx)
    //        printf("%f ", (float)smem_b[idx]);
    //}
    bp += (threadIdx.x % WARP_SZ > 7) ? ( ((threadIdx.x % WARP_SZ) - 8) * BLOCK_SIZE_B + BLOCK_SIZE_B * 64) : ((threadIdx.x % WARP_SZ) * BLOCK_SIZE_B);
    //+ (warp_x / WMMA_N) * BLOCK_SIZE_B * BLOCK_SIZE_B + (warp_x % WMMA_N) * BLOCK_SIZE_B;
    R *cp = res + (c_row + (threadIdx.x % 32) / 4) + (c_col + (threadIdx.x % 4)*2) * n;
    //R *cp = res;
    spmm_helper_mma((uint64_t)ap, (uint64_t)bp, (uint64_t)idx_vec, (uint64_t) cp, (uint32_t) n);
    //*res= 2.0;
#if 0 
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, R> acc;
    wmma::fill_fragment(acc, R(0));

    for (std::size_t k = 0; k < n; k += WMMA_K)
    {
        a_col = k;
        b_row = k;
        
        // Calculate which 8x8 blocks we need to read from shared memory
        // For A: need to find the block at (threadIdx.y * WMMA_M, k)
        // For B: need to find the block at (k, threadIdx.x / WARP_SZ * WMMA_N)
        
        // Load from shared memory (data is packed in 8x8 blocks)
        // For now, we'll use a temporary buffer to reorganize for WMMA
        // This is a placeholder - actual implementation depends on WMMA layout requirements
        wmma::load_matrix_sync(afrag, matrix_a + a_row * n + a_col, n);
        wmma::load_matrix_sync(bfrag, matrix_b + b_row * n + b_col, n);
        wmma::mma_sync(acc, afrag, bfrag, acc);
    }

    wmma::store_matrix_sync(res + c_row * n + c_col, acc, n, wmma::mem_row_major);
#endif
}

/** gemm_wrapper
  * @brief High-level interinterface for gemm designed to be called from general-purpose code
  */
template<typename I, typename R>
void gemm_wrapper(I *matrix_a, I *matrix_b, R *res, std::size_t n,
                  std::size_t superblock_sz=0, std::size_t first_superblock=0, std::size_t tile_size=0)
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
    if (tile_size > 0)
        assert( n % tile_size == 0 && "n must be a multiple of tile_size" );

    // i+=2 because two superblocks are computed in separate streams concurrently
    for (std::size_t i = first_superblock; i < n/superblock_sz; i+=2)
    {
      cudaMemcpyAsync( d_matrix_a, h_matrix_a+superblock_sz*(i-first_superblock)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[0] );
      cudaMemcpyAsync( d_matrix_a + superblock_sz*n, h_matrix_a+superblock_sz*(i-first_superblock+1)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[1] );

      const dim3 blockDim { WARP_SZ * 4, 4, 1 };
      dim3 gridDim;
      gridDim.x = (n + (WMMA_N * blockDim.x / WARP_SZ - 1)) / (WMMA_N * blockDim.x / WARP_SZ);
      gridDim.y = (superblock_sz + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y);

      if (tile_size > 0) {
          for (std::size_t k_tile = 0; k_tile < n; k_tile += tile_size) {
              gemm<I, R><<< gridDim, blockDim, 0, streams[0] >>>(d_matrix_a + k_tile, d_matrix_b + k_tile*n, d_matrix_c, n, superblock_sz);
              gemm<I, R><<< gridDim, blockDim, 0, streams[1] >>>(d_matrix_a+superblock_sz*n + k_tile, d_matrix_b + k_tile*n, d_matrix_c+superblock_sz*n, n, superblock_sz);
          }
      } else {
          gemm<I, R><<< gridDim, blockDim, 0, streams[0] >>>(d_matrix_a, d_matrix_b, d_matrix_c, n, superblock_sz);
          gemm<I, R><<< gridDim, blockDim, 0, streams[1] >>>(d_matrix_a+superblock_sz*n, d_matrix_b, d_matrix_c+superblock_sz*n, n, superblock_sz);
      }

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
