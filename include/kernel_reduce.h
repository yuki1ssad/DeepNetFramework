#pragma once


template <typename T>
__device__ void warpReduceSum(volatile T* shmemPtr, int t)
{
    shmemPtr[t] += shmemPtr[t + 16];
    shmemPtr[t] += shmemPtr[t + 8];
    shmemPtr[t] += shmemPtr[t + 4];
    shmemPtr[t] += shmemPtr[t + 2];
    shmemPtr[t] += shmemPtr[t + 1];
}

template <typename T>
__global__ void kreduceSum(size_t totalNum, size_t currentNum, T* In, T* Out)
{
    extern __shared__ T partial[];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T front = 0, back = 0;
    if (i < currentNum) {
        front = In[i];
    }
    if (i + blockDim.x < currentNum) {
        back = In[i + blockDim.x];
    }

    partial[threadIdx.x] = front + back;
    __syncthreads();

    if (threadIdx.x < 16) {
        warpReduceSum(partial, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        Out[blockIdx.x] = partial[0];
    }
}

template <class DATATYPE>
__inline__ __device__
DATATYPE warpReduceSum(DATATYPE val)
{
    for (unsigned int step = warpSize / 2; step > 0; step >> 1) {
        val += __shfl_down_sync(0xffffffff, val, step);
    }
    return val;
}

template <class DATATYPE>
__inline__ __device__
DATATYPE blockReduceSum(DATATYPE val)
{
    val = warpReduceSum(val);

    __shared__ DATATYPE warpAns[32];
    int wid = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) {
        warpAns[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (lane < (blockDim.x + warpSize - 1) / warpSize ? warpAns[lane] : 0);
        val = warpReduceSum(val);
    }
    return val;
}

template<class DATATYPE>
__global__ void kreduceSumShfl(DATATYPE* In, DATATYPE* Out, size_t N)
{
    DATATYPE val = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        val += (i < N) ? In[i] : 0;
    }
    val = blockReduceSum(val);
    if (threadIdx.x == 0) {
        Out[blockIdx.x] = val;
    }
}

template<class DATATYPE>
__global__ void kreduceSumWarpshflAtom(DATATYPE* In, DATATYPE* Out, size_t N)
{
    DATATYPE val = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        val += (i < N) ? In[i] : 0;
    }
    val = warpReduceSum(val);
    if ((threadIdx.x & (warpSize - 1)) == 0) {  // 每个 warp 的 0 号线程
        atomicAdd(Out, val);
    }
}

template<class DATATYPE>
__global__ void kreduceSumBlockshflAtom(DATATYPE* In, DATATYPE* Out, size_t N)
{
    DATATYPE val = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        val += (i < N) ? In[i] : 0;
    }
    val = blockReduceSum(val);
    if (threadIdx.x == 0) { // 每个 block 的 0 号线程
        atomicAdd(Out, val);
    }
}

template<class DATATYPE, class DATATYPE4>
__global__ void kreduceSumVec4BlockshflAtom(DATATYPE* In, DATATYPE* Out, size_t N)
{
    DATATYPE val = 0;
    DATATYPE4 frag;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N / 4; i += gridDim.x * blockDim.x) {
        frag = reinterpret_cast<DATATYPE4>(In)[i];
        val += frag.w + frag.x + frag.y + frag.z;
    }

    if (threadIdx.x < N % 4 && blockIdx.x == 0) {
        val += In[N - 1 - threadIdx.x];
    }

    val = blockReduceSum(val);
    if (threadIdx.x == 0) {
        atomicAdd(Out, val);
    }
}

template<class DATATYPE, class DATATYPE4>
__global__ void kreduceSumVec4WarpshflAtom(DATATYPE* In, DATATYPE* Out, size_t N)
{
    DATATYPE val = 0;
    DATATYPE4 frag;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N / 4; i += gridDim.x * blockDim.x) {
        frag = reinterpret_cast<DATATYPE4>(In)[i];
        val += frag.w + frag.x + frag.y + frag.z;
    }

    if (threadIdx.x < N % 4 && blockIdx.x == 0) {
        val += In[N - threadIdx.x - 1];
    }

    val = warpReduceSum(val);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(Out, val);
    }
}

template<class DATATYPE>
DATATYPE kreduceSumCpu(DATATYPE* In, size_t N) {
    if (N == 1) {
        return *In;
    }
    size_t stride = N / 2;

    // #pragma omp parallel for
    for (int i = 0; i < stride; ++i) {
        In[i] += In[i + stride];
    }

    if (N % 2) {
        In[0] += In[N - 1];
    }
    return kreduceSumCpu(In, stride);
}
