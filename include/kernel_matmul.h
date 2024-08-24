#pragma once

template <typename T>
__global__ void kmatmulNaive(T* L, T* R, T* O, size_t M, size_t K, size_t N)
{
    int rid = blockIdx.y * blockDim.y + threadIdx.y;
    int cid = blockIdx.x * blockDim.x + threadIdx.x;

    if (rid < M && cid < N) {
        T tmp = 0;
        for (size_t i = 0; i < K; ++i) {
            tmp += L[rid * K + i] * R[i * N + cid];
        }
        O[rid * N + cid] = tmp;
    }
}

template<typename T>
__global__ void kmatmulSharedMem(T* A, T* B, T* C, size_t M, size_t K, size_t N) {
    /*
        一个线程负责计算一个 C 中的结果
    */
    auto BM = blockDim.x, BN = blockDim.y;
    auto BK = BM;    // BK == BM == BN
    int num_shared_block = (K + BK - 1) / BK;

    extern __shared__ T sharedArr[];
    T* As = &sharedArr[0];
    T* Bs = &sharedArr[BM * (BK + 1)];

    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];

    T tmp = 0.;
    for (int i = 0; i < num_shared_block; ++i) {
        // copy data to shared mem
        int A_row = threadIdx.y;  
        int A_col = threadIdx.x;
        if (blockIdx.y * BM + A_row < M && i * BK + A_col < K) {
            // As[A_row][A_col] = A[A_row * K + A_col];
            As[A_row * (BK + 1) + A_col] = A[A_row * K + A_col];
            // As[A_row * BK + A_col] = A[A_row * K + A_col];
        } else {
            // As[A_row][A_col] = 0;
            As[A_row * (BK + 1) + A_col] = 0;
            // As[A_row * BK + A_col] = 0;
        }

        int B_row = threadIdx.y;  
        int B_col = threadIdx.x;
        if (i * BK + B_row < K && blockIdx.x * BN + B_col < N) {
            // Bs[B_row][B_col] = B[B_row * N + B_col];
            Bs[B_row * (BN + 1) + B_col] = B[B_row * N + B_col];
            // Bs[B_row * BN + B_col] = B[B_row * N + B_col];
        } else {
            // Bs[B_row][B_col] = 0;
            Bs[B_row * (BN + 1) + B_col] = 0;
            // Bs[B_row * BN + B_col] = 0;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        
        for (int k = 0; k < BK; ++k) {
            // tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            tmp = tmp + As[threadIdx.y * (BK + 1) + k] * Bs[k * (BN + 1) + threadIdx.x];
            // tmp = tmp + As[threadIdx.y * BK + k] * Bs[k * BN + threadIdx.x];
        }
        __syncthreads();
    }
    
    int C_row = threadIdx.y;
    int C_col = threadIdx.x;
    if (blockIdx.y * (BM + 1) + C_row < M && blockIdx.x * (BN + 1) + C_col < N) {
        C[C_row * N + C_col] = tmp;
    }
}


template<typename T>
__global__ void kmatmulSharedMemTile(T* A, T* B, T* C, size_t M, size_t K, size_t N) {
    int size = 4;   // one thread calculate 4 * 4 results in C
    auto TM = size, TN = size;
    auto BM = blockDim.x * size, BN = blockDim.y * size;  // blockDim.x = blockDim.y = 16
    auto BK = blockDim.x;    // BK = 16
    int num_shared_block = (K + BK - 1) / BK;

    extern __shared__ T sharedArr[];
    T* As = &sharedArr[0];          // BM * BK
    T* Bs = &sharedArr[BM * BK];    // BK * BN

    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];

    // T val[TM][TN] = {0.};
    T val[4][4] = {0.};

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row * BK + A_col] = A[A_row * K + A_col];
            } else {
                As[A_row * BK + A_col] = 0.;
            }
        }
        for (int n = 0; n < TN; ++n) {
            int B_row = threadIdx.y;
            int B_col = threadIdx.x * TN + n;
            if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
                Bs[B_row * BN + B_col] = B[B_row * N + B_col];
            } else {
                Bs[B_row * BN + B_col] = 0.;
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                int A_row = threadIdx.y * TM + m;
                for (int n = 0; n < TN; ++n) {
                    int B_col = threadIdx.x * TN + n;
                    val[m][n] += As[A_row * BK + k] * Bs[k * BN + B_col];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        
        for (int n = 0; n < TN; ++n) {
            int C_col = threadIdx.x * TN + n;
            if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
                C[C_row * N + C_col] = val[m][n];
            }
        }
    }
}


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define sizef4 4
#define BNF4 (16 * sizef4)
#define BMF4 (16 * sizef4)
#define BKF4 (4 * sizef4)
#define TMF4 (sizef4)
#define TNF4 (sizef4)
#define OFFSET(row, col, stride) ((row) * (stride) + (col))


__global__ void kmatmulSharedMemTileF4(float* A, float* B, float* C, size_t M, size_t K, size_t N) {
    const int block_row_thread = BNF4 / TNF4;
    const int block_col_thread = BMF4 / TMF4;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = (K - BKF4) / BKF4;

    __shared__ float As[BKF4][BMF4];    // transpose shared A for avoid bank conflict
    __shared__ float Bs[BKF4][BNF4];

    float accum[TMF4][TNF4] = {0.};

    const int load_a_cache_time = (BKF4 * BMF4) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BKF4 * BNF4) / thread_num / 4;  // Each thread load 4 float


    float load_a_cache[4 * load_a_cache_time];

    A = &A[OFFSET(blockIdx.y * BMF4, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BNF4, N)];
    C = &C[OFFSET(blockIdx.y * BMF4, blockIdx.x * BNF4, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BKF4 / 4);
    int a_tile_col = thread_id % (BKF4 / 4) * 4;
    int a_tile_stride = BMF4 / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BNF4 / 4);
    int b_tile_col = thread_id % (BNF4 / 4) * 4;
    int b_tile_stride = BKF4 / load_b_cache_time;

    float As_cache[TMF4] = {0.};
    float Bs_cache[TNF4] = {0.};

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
#pragma unroll
        for (int m = 0; m < BMF4; m += a_tile_stride) {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            As[a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BKF4; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BKF4;    // Start position of next tile block to be processed
        B += BKF4 * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BKF4; ++k) {
#pragma unroll
            for (int m = 0; m < TMF4; m += 4) {
                int A_row = threadIdx.y * TMF4 + m;
                FETCH_FLOAT4(As_cache[m]) = FETCH_FLOAT4(As[k][A_row]);
            }
#pragma unroll
            for (int n = 0; n < TNF4; n += 4) {
                int B_col = threadIdx.x * TNF4 + n;
                FETCH_FLOAT4(Bs_cache[n]) = FETCH_FLOAT4(Bs[k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TMF4; ++m) {
#pragma unroll
                for (int n = 0; n < TNF4; ++n) {
                    accum[m][n] += As_cache[m] * Bs_cache[n];
                }
            }
        }
        __syncthreads();
    }

    float TMF4p[4] = {0.};
#pragma unroll
    for (int m = 0; m < TMF4; ++m) {
        int C_row = threadIdx.y * TMF4 + m;
#pragma unroll
        for (int n = 0; n < TNF4; n += 4) {
            int C_col = threadIdx.x * TNF4 + n;
            FETCH_FLOAT4(TMF4p) = FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]);
            TMF4p[0] = accum[m][n];
            TMF4p[1] = accum[m][n + 1];
            TMF4p[2] = accum[m][n + 2];
            TMF4p[3] = accum[m][n + 3];
            FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]) = FETCH_FLOAT4(TMF4p);
        }
    }
}
