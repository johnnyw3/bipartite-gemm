#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <string>
#include <functional>
#include <mma.h>
#include <cstdlib>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

#include "bench/sparse_utils.h"

#if __cplusplus >= 202002L
#define CXX20
#endif

#define US_PER_S 1000000
#define GIGA     1000000000
#define FIXED_EPSILON 0.00001

namespace bipartite {

template < typename I, typename R >
class GemmExperiment {

public:

    // Member Variables
    const std::size_t n;

    //const std::vector<I> matrix_a;
    std::vector<I> matrix_a;
    I* d_matrix_a;
    I* h_matrix_a;

    //const std::vector<I> matrix_b;
    std::vector<I> matrix_b;
    I* d_matrix_b;

    std::vector<R> matrix_c;
    R* d_matrix_c;
    R* h_matrix_c;

    bool print_result;

    const std::size_t superblock_sz;
    std::vector<cudaStream_t> streams;

    // Sparse matrix support (2:4 sparsity)
    bool is_sparse;
    std::vector<I> matrix_b_compressed;   // compressed B (non-zero values), type I (half)
    std::vector<int> matrix_b_indices;    // packed index matrix


    // Member Functions
    GemmExperiment( std::size_t input_n, unsigned int upper_bound, std::size_t multiple, uint32_t seed, bool print_result ):
                                                        n{get_padded_sz(input_n, multiple)},
                                                        matrix_a{generate_matrix<I>(upper_bound, input_n, n, seed)},
                                                        matrix_b{generate_matrix<I>(upper_bound, input_n, n, seed)},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        h_matrix_a{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{n},
                                                        streams{nullptr},
                                                        is_sparse{false},
                                                        fixed_seed{seed},
                                                        print_result{print_result} {}

    GemmExperiment( std::size_t input_n, unsigned int upper_bound, std::size_t multiple, uint32_t seed, bool print_result, std::size_t superblock_sz ):
                                                        n{get_padded_sz(input_n, multiple)},
                                                        matrix_a{generate_matrix<I>(upper_bound, input_n, n, seed)},
                                                        matrix_b{generate_matrix<I>(upper_bound, input_n, n, seed)},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        h_matrix_a{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{superblock_sz},
                                                        streams{std::vector<cudaStream_t>(2)},
                                                        is_sparse{false},
                                                        fixed_seed{seed},
                                                        print_result{print_result} 
     {

         cudaStreamCreate(&streams[0]);
         cudaStreamCreate(&streams[1]);
     }

    // Constructor for file-loaded matrices with 2:4 sparse B
    // Reads A and B from files, compresses B using 2:4 sparsity (n_blk=2, m=4, l=1),
    // and creates the packed index matrix.
    GemmExperiment( const char *file_a, const char *file_b, std::size_t multiple, bool print_result ):
                                                        n{[&]() -> std::size_t {
                                                            int n_raw;
                                                            float *tmp;
                                                            read_mat(file_a, &n_raw, &tmp);
                                                            free(tmp);
                                                            return get_padded_sz(n_raw, multiple);
                                                        }()},
                                                        matrix_a{},
                                                        matrix_b{},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        h_matrix_a{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{n},
                                                        streams{nullptr},
                                                        is_sparse{true},
                                                        fixed_seed{0},
                                                        print_result{print_result}
     {
         constexpr int n_blk = 2, m = 4, l = 1;

         // Read matrices from files
         int n_a, n_b;
         float *raw_a, *raw_b;
         read_mat(file_a, &n_a, &raw_a);
         read_mat(file_b, &n_b, &raw_b);

         // Convert A from float to I (half) with padding
         const_cast<std::vector<I>&>(matrix_a).resize(n * n, I(0));
         for (int row = 0; row < n_a; ++row)
             for (int col = 0; col < n_a; ++col)
                 const_cast<std::vector<I>&>(matrix_a)[row * n + col] = (I)raw_a[row * n_a + col];

         // Store original B (float->half) for verification
         const_cast<std::vector<I>&>(matrix_b).resize(n * n, I(0));
         for (int row = 0; row < n_b; ++row)
             for (int col = 0; col < n_b; ++col)
                 const_cast<std::vector<I>&>(matrix_b)[row * n + col] = (I)raw_b[row * n_b + col];

         // Compress B using 2:4 sparsity
         std::size_t compressed_cols = (std::size_t)n_b * n_blk / m;
         float *mat_b_compressed_f = (float*)aligned_alloc(64, sizeof(float) * n_b * compressed_cols);
         int *idx_mat = (int*)aligned_alloc(64, sizeof(int) * n_b * compressed_cols);
         memset(mat_b_compressed_f, 0, sizeof(float) * n_b * compressed_cols);
         memset(idx_mat, 0, sizeof(int) * n_b * compressed_cols);

         squash_matrix(raw_b, mat_b_compressed_f, idx_mat, n_blk, m, l, n_b);
         cpu_transpose(mat_b_compressed_f, n_b, (int)compressed_cols);
         cpu_transpose(idx_mat, n_b / l, (int)compressed_cols);
         pack_mat(idx_mat, (int)compressed_cols, n_b / l);

         // Convert compressed B from float to I (half)
         std::size_t compressed_size = compressed_cols * n_b;
         matrix_b_compressed.resize(compressed_size);
         for (std::size_t i = 0; i < compressed_size; ++i)
             matrix_b_compressed[i] = (I)mat_b_compressed_f[i];

         // Store packed index matrix
         std::size_t packed_cols = compressed_cols / 16;
         std::size_t idx_size = packed_cols * (n_b / l);
         matrix_b_indices.resize(idx_size);
         memcpy(matrix_b_indices.data(), idx_mat, sizeof(int) * idx_size);

         free(raw_a);
         free(raw_b);
         free(mat_b_compressed_f);
         free(idx_mat);
     }
    
    ~GemmExperiment(){
        if (d_matrix_a != nullptr) cudaFree(d_matrix_a);
        if (d_matrix_b != nullptr) cudaFree(d_matrix_b);
        if (d_matrix_c != nullptr) cudaFree(d_matrix_c);
        if (h_matrix_a != nullptr) cudaFree(h_matrix_a);
    }

    void prepare_device(){
        assert( n * n == matrix_a.size() && "GemmExperiment needs to be of size n x n" );

        // Allocate space on device
        cudaMalloc( &d_matrix_a, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_c, sizeof( R ) * n * n );
 
        // Create pinned memory buffers for matricies we will be accessing
        // multiple times 
        size_t a_size = sizeof( I ) * matrix_a.size();
        size_t c_size = sizeof( R ) * matrix_c.size();
        size_t pinned_sz = (c_size > a_size) ? c_size : a_size;
        cudaMallocHost((void**) &h_matrix_a, pinned_sz);

        // Copy b to device using the pinned buffer we created for a 
        // (needs to be done first since b is row-major)
        memcpy(h_matrix_a, matrix_b.data(),  sizeof( I ) * matrix_b.size());
        cudaMemcpy( d_matrix_b, h_matrix_a,  sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

        // Now we can actually use a's pinned buffer for a
        memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());
        cudaMemcpy( d_matrix_a, h_matrix_a, sizeof( I ) * matrix_a.size(), cudaMemcpyHostToDevice );

        // Set contents of matrix_c to zero on device
        cudaMemset(d_matrix_c, 0x0, sizeof(R) * matrix_c.size() );
    }

    void unprepare_device(){
        if (d_matrix_a != nullptr) cudaFree(d_matrix_a);
        if (d_matrix_b != nullptr) cudaFree(d_matrix_b);
        if (d_matrix_c != nullptr) cudaFree(d_matrix_c);
        if (h_matrix_a != nullptr) cudaFree(h_matrix_a);
    }

    void get_product_from_device(){
        cudaMemcpy( matrix_c.data(), d_matrix_c, sizeof(R) * matrix_c.size(), cudaMemcpyDeviceToHost );
    }

    std::size_t get_n(){
        return n;
    }

    std::size_t get_superblk_sz(){
        return superblock_sz;
    }

    void run_experiment( std::function<void(I*, I*, R*)> kernel_wrapper, std::string title, int num_runs=1, std::size_t tile_size=0)
    {
        assert( num_runs && "Cannot call run_experiment with 0 runs." );

        std::size_t time_sum = 0;
        for (int i = 0; i < num_runs; i++)
        {
            auto const start = std::chrono::high_resolution_clock::now();
            prepare_device();

            if (tile_size > 0) {
                assert( n % tile_size == 0 && "n must be a multiple of tile_size" );
                for (std::size_t k_tile = 0; k_tile < n; k_tile += tile_size) {
                    kernel_wrapper(
                        d_matrix_a + k_tile,
                        d_matrix_b + k_tile * n,
                        d_matrix_c
                    );
                    //cudaDeviceSynchronize();
                }
            } else {
                kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c);
                cudaDeviceSynchronize();
            }

            get_product_from_device();
            auto const end = std::chrono::high_resolution_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            unprepare_device();
        }

        get_results(title, time_sum/num_runs, num_runs);
    }

    void run_experiment_streams( std::function<void(I*, I*, R*, cudaStream_t)> kernel_wrapper, std::string title, int num_runs=1, std::size_t tile_size=0)
    {
        assert( num_runs && "Cannot call run_experiment with 0 runs.");

        std::size_t time_sum = 0;
        for (int i = 0; i < num_runs; i++)
        {
            auto const start = std::chrono::high_resolution_clock::now();
            cudaMalloc( &d_matrix_a, sizeof( I ) * 2 * superblock_sz * n );
            cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
            cudaMalloc( &d_matrix_c, sizeof( R ) * 2 * superblock_sz * n);

            // Create pinned memory buffers for matricies we will be accessing
            // multiple times
            size_t a_size = sizeof( I ) * matrix_a.size();
            size_t c_size = sizeof( R ) * matrix_c.size();
            size_t pinned_sz = (c_size > a_size) ? c_size : a_size;
            cudaMallocHost((void**) &h_matrix_c, pinned_sz);
            h_matrix_a = (I*)((uint8_t*)h_matrix_c + c_size - a_size); 

            // Copy b to device using the pinned buffer we created for a
            // (needs to be done first since b is row-major)
            memcpy(h_matrix_c, matrix_b.data(),  sizeof( I ) * matrix_b.size());
            cudaMemcpy( d_matrix_b, h_matrix_c,  sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

            // Now we can actually use a's pinned buffer for a
            memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());

            assert( n * n == matrix_a.size() && "GemmExperiment need to be of size n x n" );
            assert( n % superblock_sz ==0 && "superblock_sz must be a factor of n" );

            // i+=2 because two superblocks are computed in separate streams concurrently
            for (std::size_t i = 0; i < n/superblock_sz; i+=2)
            {
              cudaMemcpyAsync( d_matrix_a, h_matrix_a+superblock_sz*i*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[0] );
              cudaMemcpyAsync( d_matrix_a + superblock_sz*n, h_matrix_a+superblock_sz*(i+1)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[1] );

              if (tile_size > 0) {
                  assert( n % tile_size == 0 && "n must be a multiple of tile_size" );
                  for (std::size_t k_tile = 0; k_tile < n; k_tile += tile_size) {
                      kernel_wrapper(d_matrix_a + k_tile, d_matrix_b + k_tile * n, d_matrix_c, streams[0]);
                      kernel_wrapper(d_matrix_a + superblock_sz*n + k_tile, d_matrix_b + k_tile * n, d_matrix_c + superblock_sz*n, streams[1]);
                    cudaDeviceSynchronize();
                  }
              } else {
                  kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c, streams[0]);
                  kernel_wrapper(d_matrix_a+superblock_sz*n, d_matrix_b, d_matrix_c+superblock_sz*n, streams[1]);
              }

              cudaMemcpyAsync(  h_matrix_c + superblock_sz*i*n, d_matrix_c, sizeof( R ) * superblock_sz*n, cudaMemcpyDeviceToHost, streams[0] );
              cudaMemcpyAsync(  h_matrix_c + superblock_sz*(i+1)*n, d_matrix_c + superblock_sz*n, sizeof( R ) * superblock_sz*n, cudaMemcpyDeviceToHost, streams[1] );
            }

            cudaDeviceSynchronize();
            memcpy(matrix_c.data(), h_matrix_c,  sizeof( R ) * matrix_c.size());
            auto const end = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaFree( &d_matrix_a );
            cudaFree( &d_matrix_b );
            cudaFree( &d_matrix_c );
            cudaFreeHost( (void*) h_matrix_c );
        }

        get_results(title, time_sum/num_runs, num_runs);
    }

    void run_experiment_sparse(
        std::function<void(I*, I*, R*, uint32_t*, std::size_t)> kernel_wrapper,
        std::string title, int num_runs=1, std::size_t tile_size=0)
    {
        assert(is_sparse && "run_experiment_sparse requires sparse data");
        assert(num_runs && "Cannot call run_experiment_sparse with 0 runs.");

        // Transpose matrix_a (n×n row-major → column-major for kernel's matrix_b param)
        std::vector<I> matrix_a_t(n * n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                matrix_a_t[c * n + r] = matrix_a[r * n + c];

        // Transpose matrix_b_compressed for kernel's matrix_a param (sparse operand)
        // matrix_b_compressed: n rows × (n/2) cols → (n/2) rows × n cols
        std::size_t b_comp_cols = matrix_b_compressed.size() / n;
        std::vector<I> matrix_b_comp_t(matrix_b_compressed.size());
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < b_comp_cols; ++c)
                matrix_b_comp_t[c * n + r] = matrix_b_compressed[r * b_comp_cols + c];

        I* d_sparse_a;      // compressed B (transposed) → kernel's matrix_a
        I* d_dense_b;       // A (transposed) → kernel's matrix_b
        R* d_res;
        uint32_t* d_idx;

        std::size_t time_sum = 0;
        for (int run = 0; run < num_runs; run++)
        {
            auto const start = std::chrono::high_resolution_clock::now();

            cudaMalloc(&d_sparse_a, sizeof(I) * matrix_b_comp_t.size());
            cudaMalloc(&d_dense_b, sizeof(I) * n * n);
            cudaMalloc(&d_res, sizeof(R) * n * n);
            cudaMalloc(&d_idx, sizeof(uint32_t) * matrix_b_indices.size());

            cudaMemcpy(d_sparse_a, matrix_b_comp_t.data(), sizeof(I) * matrix_b_comp_t.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dense_b, matrix_a_t.data(), sizeof(I) * n * n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, matrix_b_indices.data(), sizeof(uint32_t) * matrix_b_indices.size(), cudaMemcpyHostToDevice);
            cudaMemset(d_res, 0x0, sizeof(R) * n * n);

            if (tile_size > 0) {
                assert(n % tile_size == 0 && "n must be a multiple of tile_size");
                for (std::size_t k_tile = 0; k_tile < n; k_tile += tile_size) {
                    kernel_wrapper(
                        d_sparse_a + k_tile,
                        d_dense_b + k_tile * n,
                        d_res,
                        d_idx,
                        k_tile
                    );
                }
            } else {
                kernel_wrapper(d_sparse_a, d_dense_b, d_res, d_idx, 0);
            }

            cudaMemcpy(matrix_c.data(), d_res, sizeof(R) * n * n, cudaMemcpyDeviceToHost);
            auto const end = std::chrono::high_resolution_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaFree(d_sparse_a);
            cudaFree(d_dense_b);
            cudaFree(d_res);
            cudaFree(d_idx);
        }

        get_results(title, time_sum/num_runs, num_runs);
    }

    void get_results( std::string title, std::size_t time_us, int num_runs=1 ) {
#ifdef CXX20
        std::cout << std::format("--------{} (runs: {})--------", title, num_runs) << std::endl;
#else
        printf("--------%s (runs: %d)--------\n", title.c_str(), num_runs);
#endif

#ifdef NO_OPENBLAS
        std::vector<R> matrix_c_expected = naive_cpu_matmul();
#else
        const std::vector<R> matrix_c_expected = cblas_cpu_matmul();
#endif

        if (print_result) std::cout << "Expected Result:" << std::endl;
        print_matrix<R>( matrix_c_expected, n, print_result );
        if (print_result) std::cout << "Actual Result:" << std::endl;
        print_matrix<R>( matrix_c, n, print_result );

        std::cout << "Correct output: " << matrices_equal( matrix_c, matrix_c_expected ) << std::endl;

        double gflops = get_gflops(time_us, NUM_SMS);

#ifdef CXX20
        std::cout << std::format("Time: {} us", time_us) << std::endl
                  << std::format("Estimated GFLOPs/SM: {}", gflops) << std::endl;
#else
        printf("Time: %lu us\n", time_us);
        printf("Estimated GFLOPs/SM: %lf\n", gflops);
#endif
    }

    double get_gflops(std::size_t us, std::size_t num_sms)
    {
        double s = us*1.0 / US_PER_S;
        return 2 * (n*n*n) / s / GIGA / num_sms;
    }


    // Static Functions
    template< typename T >
    void print_matrix( const std::vector<T>& matrix, std::size_t side_length, bool enabled = false )
    {
        assert( side_length*side_length == matrix.size() && "matrix must be of length n*n");

        if (enabled)
        {
            for (std::size_t idx = 0; idx < n; ++idx)
            {
                std::cout << idx << ": ";
                for (std::size_t jdx = 0; jdx < n; ++jdx)
                {   
                    /* Note: Certain types like half do not have an << overload defined, so 
                       we cast to a float to ensure we can print.
                       May need to come up with a better/safer option later.
                    */
                    std::cout << static_cast<float> (matrix[idx*n + jdx]) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

private:

    // Static Functions
    template< typename T >
    static std::vector<T> generate_matrix( unsigned int upper_bound, std::size_t n, std::size_t padded_n, uint32_t seed )
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::mt19937::result_type> distribution(0, upper_bound);

        std::vector<T> matrix;
        for ( std::size_t idx = 0; idx < padded_n * padded_n; ++idx ){
            std::size_t row = idx / padded_n;
            std::size_t col = idx % padded_n;
            T val = 0;
            std::size_t count = 0;
            if ((count++) < n*n && row < n && col < n){
                /* first convert to int as CUDA < 12 can't do this conversion
                 * implicitly
                 */
                int val_int = distribution(rng);
                val = (T)val_int;
            }
            if (!seed)
                matrix.push_back((T)1);
            else if (seed == 1)
            {
                if (row == 16 && col < 4)
                    matrix.push_back((T)1);
                else
                    matrix.push_back((T)0);
            }
            else
                matrix.push_back( val );
        }

        return matrix;
    }

    // Static Member Variables
    const uint32_t fixed_seed;

    constexpr std::size_t get_padded_sz( std::size_t n, std::size_t multiple)
    {
        return n%multiple ? n + (multiple - n%multiple) : n;
    }

    template <typename T>
    bool matrices_equal( const std::vector<T>& matrix_actual, const std::vector<T>& matrix_expected )
    {
        assert( matrix_actual.size() == matrix_expected.size() && "The given matrices must have the same size");

        for ( std::size_t idx = 0; idx < matrix_actual.size(); ++idx)
        {
            if (fabs ( (float) (matrix_actual[idx] - matrix_expected[idx] ) ) >= FIXED_EPSILON )
                return false;
        }
        return true;
    }

    // Member Functions
    #ifdef NO_OPENBLAS

    std::vector<R> naive_cpu_matmul()
    {   
        // Create a zero-initialized result matrix of size nxn.
        std::vector<T> result( n*n, T(0) );

        for (std::size_t idx = 0; idx < n; idx++)
            for (std::size_t jdx = 0; jdx < n; jdx++)
                for (std::size_t kdx = 0; kdx < n; kdx++)
                result[idx*n + jdx] += matrix_a[idx*n + kdx] * matrix_b[kdx*n + jdx];
        
        return result;
    }

    #else

    std::vector<R> cblas_cpu_matmul()
    {   
        // Generate vectors of floats to be compatible with cblas
        const std::vector<float> matrix_a_float( matrix_a.begin(), matrix_a.end() );
        const std::vector<float> matrix_b_float( matrix_b.begin(), matrix_b.end() );

        // Create a zero-initialized result matrix of size nx n.
        std::vector<float> result( n * n, 0 );
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    matrix_a_float.data(), n, matrix_b_float.data(), n, 1.0, result.data(), n );
        
        std::vector<R> result_converted( result.begin(), result.end() ); 
        
        return result_converted;
    }

    #endif

}; // class GemmExperiment


} // namepace bipartite
