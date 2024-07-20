#pragma once
#include <iostream>
#include <cmath>

enum class MAP_OP
{
    ADD = 0,
    MULTIPLY,
    POW,
    LOG,
    ABS,
    SIGN
};

std::ostream& operator<<(std::ostream& os, MAP_OP op);

template <typename T>
__global__ void kmap(size_t N, T* In, T operand, T* Out, MAP_OP op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op)
        {
            case MAP_OP::ADD:
                Out[tid] = In[tid] + operand;
                break;

            case MAP_OP::MULTIPLY:
                Out[tid] = In[tid] * operand;
                break;
            
            case MAP_OP::POW:
                Out[tid] = powf(In[tid], operand);
                break;
            
            case MAP_OP::LOG:
                Out[tid] = logf(In[tid]);
                break;
            
            case MAP_OP::ABS:
                Out[tid] = fabsf(In[tid]);
                break;
            
            case MAP_OP::SIGN:
                Out[tid] = In[tid] > 0 ? 1 : -1;
                break;
            
            default:
                break;
        }
    }
}

template <typename T>
__global__ void kmapInplace(size_t N, T* In, T operand, MAP_OP op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op)
        {
            case MAP_OP::ADD:
                In[tid] = In[tid] + operand;
                break;

            case MAP_OP::MULTIPLY:
                In[tid] = In[tid] * operand;
                break;
            
            case MAP_OP::POW:
                In[tid] = powf(In[tid], operand);
                break;
            
            case MAP_OP::LOG:
                In[tid] = logf(In[tid]);
                break;
            
            case MAP_OP::ABS:
                In[tid] = fabsf(In[tid]);
                break;
            
            case MAP_OP::SIGN:
                In[tid] = In[tid] > 0 ? 1 : -1;
                break;
            
            default:
                break;
        }
    }
}