#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace nvcuda;

// -------------------- Constants --------------------
#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

// Warp layout within a thread block: 2x2 warps
#define WARPS_X 2
#define WARPS_Y 2
#define WARPS_PER_BLOCK (WARPS_X * WARPS_Y)  // 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)  // 128

// Block-level tile dimensions (each warp covers one 16x16 sub-tile)
#define BLOCK_M (WARPS_Y * MMA_M)  // 32
#define BLOCK_N (WARPS_X * MMA_N)  // 32

inline __host__ __device__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// -------------------- Kernel --------------------
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void matmul_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    size_t M, size_t N, size_t K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int warp_row = warp_id / WARPS_X;
    const int warp_col = warp_id % WARPS_X;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int tile_row = block_row + warp_row * MMA_M;
    const int tile_col = block_col + warp_col * MMA_N;

    // Accumulator fragment (FP32 for precision)
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Dynamic shared memory: reused for loading (half) and storing (float)
    extern __shared__ char smem_raw[];
    half* smemA = reinterpret_cast<half*>(smem_raw);      // BLOCK_M x MMA_K
    half* smemB = smemA + BLOCK_M * MMA_K;                // MMA_K x BLOCK_N

    // ---- Main K-loop: accumulate via tensor core MMA ----
    for (size_t kb = 0; kb < K; kb += MMA_K) {
        // Cooperative load A tile (BLOCK_M x MMA_K) with zero-padding for edges
        for (int idx = tid; idx < BLOCK_M * MMA_K; idx += THREADS_PER_BLOCK) {
            int a_row = idx / MMA_K;
            int a_col = idx % MMA_K;
            int gr = block_row + a_row;
            int gc = (int)kb + a_col;
            half val = __float2half(0.0f);
            if ((size_t)gr < M && (size_t)gc < K)
                val = A[gr * K + gc];
            smemA[idx] = val;
        }

        // Cooperative load B tile (MMA_K x BLOCK_N) with zero-padding for edges
        for (int idx = tid; idx < MMA_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int b_row = idx / BLOCK_N;
            int b_col = idx % BLOCK_N;
            int gr = (int)kb + b_row;
            int gc = block_col + b_col;
            half val = __float2half(0.0f);
            if ((size_t)gr < K && (size_t)gc < N)
                val = B[gr * N + gc];
            smemB[idx] = val;
        }

        __syncthreads();

        // Each warp loads its 16x16 sub-tile from shared memory into fragments
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::row_major> b_frag;

        // A fragment: rows [warp_row*16 .. warp_row*16+15], stride = MMA_K
        wmma::load_matrix_sync(a_frag, smemA + warp_row * MMA_M * MMA_K, MMA_K);
        // B fragment: cols [warp_col*16 .. warp_col*16+15], stride = BLOCK_N
        wmma::load_matrix_sync(b_frag, smemB + warp_col * MMA_N, BLOCK_N);

        // Tensor core MMA: replaces the entire scalar TILE_K FMA loop
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ---- Store: accumulator (float) -> shared memory -> global (half) ----
    float* smem_store = reinterpret_cast<float*>(smem_raw);
    float* warp_tile = smem_store + warp_id * MMA_M * MMA_N;

    wmma::store_matrix_sync(warp_tile, c_frag, MMA_N, wmma::mem_row_major);
    __syncwarp();

    // Convert FP32 -> FP16 and write to global memory with bounds checking
    for (int idx = lane_id; idx < MMA_M * MMA_N; idx += WARP_SIZE) {
        int r = idx / MMA_N;
        int c = idx % MMA_N;
        int gr = tile_row + r;
        int gc = tile_col + c;
        if ((size_t)gr < M && (size_t)gc < N) {
            C[gr * N + gc] = __float2half_rn(warp_tile[r * MMA_N + c]);
        }
    }
}

// -------------------- Host launcher --------------------
void launch_gpu_implementation(
    void* output,
    void* A,
    void* B,
    int64_t N)
{
    const size_t M = (size_t)N;
    const size_t K = (size_t)N;
    const size_t NN = (size_t)N;

    dim3 grid((unsigned)div_ceil(NN, (size_t)BLOCK_N),
              (unsigned)div_ceil(M, (size_t)BLOCK_M),
              1);

    dim3 block(THREADS_PER_BLOCK, 1, 1);

    // Shared memory: max of K-loop (half tiles) and store phase (float accumulators)
    size_t smem_load = (BLOCK_M * MMA_K + MMA_K * BLOCK_N) * sizeof(half);    // 2048 bytes
    size_t smem_store = WARPS_PER_BLOCK * MMA_M * MMA_N * sizeof(float);       // 4096 bytes
    size_t smem_bytes = (smem_load > smem_store) ? smem_load : smem_store;

    matmul_wmma_kernel<<<grid, block, smem_bytes>>>(
        static_cast<const half*>(A),
        static_cast<const half*>(B),
        static_cast<half*>(output),
        M, NN, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
