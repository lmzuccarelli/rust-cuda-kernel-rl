#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <algorithm>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / 32)
#define WARPS_M 4
#define WARPS_N 2
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define WARP_M_TILES (WM / WMMA_TILE_M)
#define WARP_N_TILES (WN / WMMA_TILE_N)
#define CHUNK_K_CNT (BK / WMMA_TILE_K)

#define A_PAD 8
#define B_PAD 8
#define A_STRIDE (BK + A_PAD)
#define B_STRIDE (BN + B_PAD)

#define K_STAGE 4

#define A_STAGE_ELEMS (BM * A_STRIDE)
#define B_STAGE_ELEMS (BK * B_STRIDE)
#define SMEM_ELEMS ((A_STAGE_ELEMS + B_STAGE_ELEMS) * K_STAGE)

#define A_COPIES_PER_ROW (BK / 8)
#define A_TOTAL_COPIES (BM * A_COPIES_PER_ROW)
#define A_ITERS (A_TOTAL_COPIES / BLOCK_SIZE)

#define B_COPIES_PER_ROW (BN / 8)
#define B_TOTAL_COPIES (BK * B_COPIES_PER_ROW)
#define B_ITERS (B_TOTAL_COPIES / BLOCK_SIZE)

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void async_load_A(
    half *smem_A, const half *__restrict__ A,
    int stage, size_t k_base, size_t block_m,
    size_t M, size_t K, int tid)
{
    #pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
        int ci = it * BLOCK_SIZE + tid;
        int row = ci / A_COPIES_PER_ROW;
        int col = (ci % A_COPIES_PER_ROW) * 8;

        size_t gr = block_m + row;
        size_t gc = k_base + col;

        uint32_t dst = __cvta_generic_to_shared(
            &smem_A[stage * A_STAGE_ELEMS + row * A_STRIDE + col]);

        bool ok = (gr < M) && (gc + 8 <= K);
        const half *src = ok ? &A[gr * K + gc] : A;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
            :: "r"(dst), "l"(src), "n"(16), "r"(ok ? 16 : 0));
    }
}

__device__ __forceinline__ void async_load_B(
    half *smem_B, const half *__restrict__ B,
    int stage, size_t k_base, size_t block_n,
    size_t K, size_t N, int tid)
{
    #pragma unroll
    for (int it = 0; it < B_ITERS; it++) {
        int ci = it * BLOCK_SIZE + tid;
        int row = ci / B_COPIES_PER_ROW;
        int col = (ci % B_COPIES_PER_ROW) * 8;

        size_t gr = k_base + row;
        size_t gc = block_n + col;

        uint32_t dst = __cvta_generic_to_shared(
            &smem_B[stage * B_STAGE_ELEMS + row * B_STRIDE + col]);

        bool ok = (gr < K) && (gc + 8 <= N);
        const half *src = ok ? &B[gr * N + gc] : B;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
            :: "r"(dst), "l"(src), "n"(16), "r"(ok ? 16 : 0));
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
mmaAsyncStage4Kernel(
    const half *__restrict__ A,
    const half *__restrict__ B,
    half *__restrict__ C,
    size_t M, size_t N, size_t K)
{
    extern __shared__ half smem[];
    half *smem_A = smem;
    half *smem_B = smem + A_STAGE_ELEMS * K_STAGE;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const size_t block_m = (size_t)blockIdx.y * BM;
    const size_t block_n = (size_t)blockIdx.x * BN;

    const int num_k_tiles = (int)div_ceil(K, (size_t)BK);

    wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float>
        c_frag[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // ====== PROLOGUE: prefetch first K_STAGE-1 tiles ======
    #pragma unroll
    for (int s = 0; s < K_STAGE - 1; s++) {
        if (s < num_k_tiles) {
            size_t kb = (size_t)s * BK;
            async_load_A(smem_A, A, s, kb, block_m, M, K, tid);
            async_load_B(smem_B, B, s, kb, block_n, K, N, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // ====== MAIN LOOP: overlap load and compute ======
    int rd = 0;
    int wr = K_STAGE - 1;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_STAGE - 2));
        __syncthreads();

        int nk = kt + K_STAGE - 1;
        if (nk < num_k_tiles) {
            size_t kb = (size_t)nk * BK;
            async_load_A(smem_A, A, wr, kb, block_m, M, K, tid);
            async_load_B(smem_B, B, wr, kb, block_n, K, N, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // Tensor core compute on current read stage
        #pragma unroll
        for (int kk = 0; kk < CHUNK_K_CNT; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K,
                           half, wmma::row_major> a_frag[WARP_M_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K,
                           half, wmma::row_major> b_frag[WARP_N_TILES];

            #pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
                int sr = warp_m * WM + m * WMMA_TILE_M;
                int sc = kk * WMMA_TILE_K;
                wmma::load_matrix_sync(a_frag[m],
                    &smem_A[rd * A_STAGE_ELEMS + sr * A_STRIDE + sc],
                    A_STRIDE);
            }

            #pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                int sk = kk * WMMA_TILE_K;
                int sn = warp_n * WN + n * WMMA_TILE_N;
                wmma::load_matrix_sync(b_frag[n],
                    &smem_B[rd * B_STAGE_ELEMS + sk * B_STRIDE + sn],
                    B_STRIDE);
            }

            #pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++)
                #pragma unroll
                for (int n = 0; n < WARP_N_TILES; n++)
                    wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
        }

        rd = (rd + 1) % K_STAGE;
        wr = (wr + 1) % K_STAGE;
    }

    // ====== STORE C: stage through shared memory for float→half ======
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    float *c_stage = (float *)smem;

    #pragma unroll
    for (int m = 0; m < WARP_M_TILES; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_N_TILES; n++) {
            size_t cr = block_m + warp_m * WM + m * WMMA_TILE_M;
            size_t cc = block_n + warp_n * WN + n * WMMA_TILE_N;

            if (cr >= M || cc >= N) continue;

            float *wbuf = &c_stage[warp_id * WMMA_TILE_M * WMMA_TILE_N];
            wmma::store_matrix_sync(wbuf, c_frag[m][n],
                                    WMMA_TILE_N, wmma::mem_row_major);
            __syncwarp();

            bool full = (cr + WMMA_TILE_M <= M) && (cc + WMMA_TILE_N <= N);
            #pragma unroll
            for (int i = lane_id; i < WMMA_TILE_M * WMMA_TILE_N; i += 32) {
                int r = i / WMMA_TILE_N;
                int c = i % WMMA_TILE_N;
                if (full || (cr + r < M && cc + c < N)) {
                    C[(cr + r) * N + cc + c] = __float2half_rn(wbuf[i]);
                }
            }
            __syncwarp();
        }
    }
}

// Host launcher
size_t get_smem_max_size(size_t N) {
    return SMEM_ELEMS * sizeof(half);
}

void launch_gpu_implementation(
    void* output, void* A, void* B, int64_t N)
{
    size_t smem_size = SMEM_ELEMS * sizeof(half);

    cudaFuncSetAttribute(mmaAsyncStage4Kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    dim3 block(BLOCK_SIZE);
    dim3 grid((unsigned int)div_ceil(N, BN),
              (unsigned int)div_ceil(N, BM));

    mmaAsyncStage4Kernel<<<grid, block, smem_size>>>(
        static_cast<const half*>(A),
        static_cast<const half*>(B),
        static_cast<half*>(output),
        (size_t)N, (size_t)N, (size_t)N);

    cudaDeviceSynchronize();
}
