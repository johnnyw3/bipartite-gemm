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

    const std::vector<I> matrix_a;
    I* d_matrix_a;
    I* h_matrix_a;

    const std::vector<I> matrix_b;
    I* d_matrix_b;
    I* h_matrix_b;

    std::vector<R> matrix_c;
    R* d_matrix_c;
    R* h_matrix_c;

    bool print_result;

    const std::size_t superblock_sz;
    std::vector<cudaStream_t> streams;


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
                                                        h_matrix_b{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{n},
                                                        streams{nullptr},
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
                                                        h_matrix_b{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{superblock_sz},
                                                        streams{std::vector<cudaStream_t>(2)},
                                                        fixed_seed{seed},
                                                        print_result{print_result} 
     {

         cudaStreamCreate(&streams[0]);
         cudaStreamCreate(&streams[1]);
     }
    
    ~GemmExperiment(){
        if (d_matrix_a != nullptr) cudaFree(d_matrix_a);
        if (d_matrix_b != nullptr) cudaFree(d_matrix_b);
        if (d_matrix_c != nullptr) cudaFree(d_matrix_c);
        if (h_matrix_a != nullptr) cudaFree(h_matrix_a);
        if (h_matrix_b != nullptr) cudaFree(h_matrix_b);
        if (h_matrix_c != nullptr) cudaFree(h_matrix_c);
    }

    void prepare_device(){
        assert( n * n == matrix_a.size() && "GemmExperiment needs to be of size n x n" );

        // Allocate space on device
        cudaMalloc( &d_matrix_a, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_c, sizeof( R ) * n * n );
 
        // Copy contents of matrix_a and matrix_b to pinned memory
        cudaMallocHost((void**) &h_matrix_a, sizeof( I ) * matrix_a.size());
        cudaMallocHost((void**) &h_matrix_b, sizeof( I ) * matrix_b.size());
        cudaMallocHost((void**) &h_matrix_b, sizeof( R ) * matrix_c.size());
        memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());
        memcpy(h_matrix_b, matrix_b.data(),  sizeof( I ) * matrix_b.size());

        cudaMemcpy( d_matrix_a, h_matrix_a, sizeof( I ) * matrix_a.size(), cudaMemcpyHostToDevice );
        cudaMemcpy( d_matrix_b, h_matrix_b, sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

        // Set contents of matrix_c to zero on device
        cudaMemset(d_matrix_c, 0x0, sizeof(R) * matrix_c.size() );
    }

    void unprepare_device(){
        if (d_matrix_a != nullptr) cudaFree(d_matrix_a);
        if (d_matrix_b != nullptr) cudaFree(d_matrix_b);
        if (d_matrix_c != nullptr) cudaFree(d_matrix_c);
        if (h_matrix_a != nullptr) cudaFree(h_matrix_a);
        if (h_matrix_b != nullptr) cudaFree(h_matrix_b);
        if (h_matrix_c != nullptr) cudaFree(h_matrix_c);
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

    void run_experiment( std::function<void(I*, I*, R*)> kernel_wrapper, std::string title, int num_runs=1)
    {
        assert( num_runs && "Cannot call run_experiment with 0 runs." );

        std::size_t time_sum = 0;
        for (int i = 0; i < num_runs; i++)
        {
            auto const start = std::chrono::high_resolution_clock::now();
            prepare_device();
            kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c);
            cudaDeviceSynchronize();
            get_product_from_device();
            auto const end = std::chrono::high_resolution_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            unprepare_device();
        }

        get_results(title, time_sum/num_runs, num_runs);
    }

    void run_experiment_streams( std::function<void(I*, I*, R*, cudaStream_t)> kernel_wrapper, std::string title, int num_runs=1)
    {
        assert( num_runs && "Cannot call run_experiment with 0 runs.");

        std::size_t time_sum = 0;
        for (int i = 0; i < num_runs; i++)
        {
            auto const start = std::chrono::high_resolution_clock::now();
            cudaMalloc( &d_matrix_a, sizeof( I ) * 2 * superblock_sz * n );
            cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
            cudaMalloc( &d_matrix_c, sizeof( R ) * 2 * superblock_sz * n);

            // Copy contents of matrix_a and matrix_b to pinned memory
            cudaMallocHost((void**) &h_matrix_a, sizeof( I ) * matrix_a.size());
            cudaMallocHost((void**) &h_matrix_b, sizeof( I ) * matrix_b.size());
            cudaMallocHost((void**) &h_matrix_c, sizeof( R ) * matrix_b.size());
            memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());
            memcpy(h_matrix_b, matrix_b.data(),  sizeof( I ) * matrix_b.size());

            // Copy b to device (needs to be done first since b is row-major)
            cudaMemcpy( d_matrix_b, h_matrix_b,  sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

            assert( n * n == matrix_a.size() && "GemmExperiment need to be of size n x n" );
            assert( n % superblock_sz ==0 && "superblock_sz must be a factor of n" );

            // i+=2 because two superblocks are computed in separate streams concurrently
            for (std::size_t i = 0; i < n/superblock_sz; i+=2)
            {
              cudaMemcpyAsync( d_matrix_a, h_matrix_a+superblock_sz*i*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[0] );
              cudaMemcpyAsync( d_matrix_a + superblock_sz*n, h_matrix_a+superblock_sz*(i+1)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[1] );

              kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c, streams[0]);
              kernel_wrapper(d_matrix_a+superblock_sz*n, d_matrix_b, d_matrix_c+superblock_sz*n, streams[1]);

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
            cudaFreeHost( (void*) h_matrix_a );
            cudaFreeHost( (void*) h_matrix_b );
            cudaFreeHost( (void*) h_matrix_c );
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
