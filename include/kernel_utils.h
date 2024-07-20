#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename T>
__global__ void kmemset(size_t N, T* I, T val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        I[tid] = val;
    }
}

template <typename T>
__global__ void kmemset_d(size_t N, T* I, T alpha, T* val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        I[tid] = alpha * (*val);
    }
}

template <typename T>
__global__ void kinitializeRandom(T* data, size_t N, T lower_bound, T upper_bound)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        data[tid] = curand_uniform(&state) * (upper_bound - lower_bound) - lower_bound;
    }
}


