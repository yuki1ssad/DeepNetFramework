#pragma once

template <typename T>
__global__ void ktranspose(T* In, T* Out, size_t M, size_t N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < N && y < M) {
        Out[x * M + y] = In[y * N + x];
    }
}

template <typename T>
__global__ void ktransposeSharedMem(T* In, T* Out, size_t M, size_t N) {
    // printf("run ktransposeSharedMem\n");
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    extern __shared__ T tile[];
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    tile[threadIdx.y * blockDim.x + threadIdx.x] = (x < N && y < M) ? In[y * N + x] : 0;
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < M && y < N) {
        Out[y * M + x] = tile[threadIdx.x * blockDim.y + threadIdx.y];
    }
}


template <typename T>
__device__ int ktranspose_minbkcft_index(int index, int group_size) {
    int row_index = index * (sizeof(T) / 4) / 32;   // one bank: 4B
    int stride = warpSize / group_size;
    int group_index = index / group_size;
    int index_in_group = index % group_size;
    int index_in_group_minbkcft = (index_in_group + stride * row_index) % group_size;
    int minbkcft_index =  group_index * group_size + index_in_group_minbkcft;
    return minbkcft_index;
}

// block only (8, 8)(16, 16)(32, 32)
template <typename DATA_TYPE>
__global__ void ktransposeSharedMemMinbkcft(DATA_TYPE *In, DATA_TYPE *Out, size_t M, size_t N) {
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    // extern __shared__ DATA_TYPE tile[]; // [TILE_DIM, TILE_DIM + 1]
    // __shared__ DATA_TYPE tile[16][17];
    __shared__ DATA_TYPE tile[8][8 + 1];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // int minbkcft_index = ktranspose_minbkcft_index<DATA_TYPE>(
    //     threadIdx.x * blockDim.y + threadIdx.y,
    //     blockDim.x
    // );
    // int minbkcft_index = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    // tile[minbkcft_index] = (x < N && y < M) ? In[y * N + x] : 0;
    tile[threadIdx.y][threadIdx.x] = (x < N && y < M) ? In[y * N + x] : 0;

    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < M && y < N) {
        // minbkcft_index = ktranspose_minbkcft_index<DATA_TYPE>(
        //     threadIdx.y * blockDim.x + threadIdx.x,
        //     blockDim.x
        // );
        // minbkcft_index = threadIdx.x * (blockDim.y) + threadIdx.y;
        // minbkcft_index = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
        // Out[y * M + x] = tile[minbkcft_index];
        Out[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template <typename DATA_TYPE = int>
__device__ int transpose_smem_4xvec4_minbkcft_index(int index) {
    int bank_row = index / 32;
    int bank = index % 32;
    int frag_in_smem_Idx_y = bank_row / 4;
    int frag_in_smem_Idx_x = bank / 4;
    int frag_group_in_smem_Idx_y = (frag_in_smem_Idx_y / 4) * 4;
    int ele_in_frag_Idx_x = bank % 4;
    int frag_in_smem_minbkcft_Idx_x = (frag_in_smem_Idx_x + frag_group_in_smem_Idx_y) % 8;
    int minbkcft_index =\
        bank_row * 32 +
        frag_in_smem_minbkcft_Idx_x * 4 +
        (ele_in_frag_Idx_x + frag_in_smem_Idx_y) % 4;
    return minbkcft_index;
}


// frag & vec & minbkcft & STG coalesced, float only
template <typename DATA_TYPE, typename DATA_TYPE_4>
__global__ void ktransposeSharedMem4xvec4Minbkcft(DATA_TYPE *In, DATA_TYPE *Out, size_t M, size_t N) {
    assert(M % 32 == 0 && N % 32 == 0);
    assert(blockDim.x == 8 && blockDim.y == 8 && blockDim.z == 1);

    extern __shared__ DATA_TYPE tile[]; // 16 * blocksize

    int TILE_DIM = blockDim.x;
    size_t x_in_block = threadIdx.x;
    size_t y_in_block = threadIdx.y;
    size_t x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
    size_t y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;

    DATA_TYPE_4 row0 = reinterpret_cast<DATA_TYPE_4*>(In)[(4 * y_in_kernel + 0) * (N / 4) + x_in_kernel];
    DATA_TYPE_4 row1 = reinterpret_cast<DATA_TYPE_4*>(In)[(4 * y_in_kernel + 1) * (N / 4) + x_in_kernel];
    DATA_TYPE_4 row2 = reinterpret_cast<DATA_TYPE_4*>(In)[(4 * y_in_kernel + 2) * (N / 4) + x_in_kernel];
    DATA_TYPE_4 row3 = reinterpret_cast<DATA_TYPE_4*>(In)[(4 * y_in_kernel + 3) * (N / 4) + x_in_kernel];

    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.x;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.y;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.z;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.w;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.x;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.y;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.z;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.w;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.x;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.y;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.z;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.w;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.x;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.y;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.z;
    tile[transpose_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.w;
    __syncthreads();

    x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
    y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;

    for (int r = 0; r < 4; r++) {
        DATA_TYPE_4 vecOut = make_float4(
            tile[transpose_smem_4xvec4_minbkcft_index((4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 0))],
            tile[transpose_smem_4xvec4_minbkcft_index((4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 1))],
            tile[transpose_smem_4xvec4_minbkcft_index((4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 2))],
            tile[transpose_smem_4xvec4_minbkcft_index((4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 3))]
        );
        reinterpret_cast<DATA_TYPE_4*>(Out)[(4 * y_in_kernel + r) * (M / 4) + x_in_kernel] = vecOut;
    }
}