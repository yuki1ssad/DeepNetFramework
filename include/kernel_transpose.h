#pragma once

template <typename T>
__global__ void ktranspose(T* In, T* Out, size_t M, size_t N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < N && y < M) {
        Out[x * M + y] = In[y * N + x];
    }
}