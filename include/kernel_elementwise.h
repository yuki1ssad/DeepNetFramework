#pragma once
#include <iostream>

enum class ELE_OP
{
    ADD = 0,
    SUB,
    MULTIPLY,
    DIVIDE
};

std::ostream& operator<<(std::ostream& os, ELE_OP op);

template <typename T>
__global__ void kelementwise(size_t N, T* In, T alpha, T* operand, T* Out, ELE_OP op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op)
        {
        case ELE_OP::ADD:
            Out[tid] = In[tid] + (alpha * operand[tid]);
            break;

        case ELE_OP::SUB:
            Out[tid] = In[tid] - (alpha * operand[tid]);
            break;
        
        case ELE_OP::MULTIPLY:
            Out[tid] = In[tid] * (alpha * operand[tid]);
            break;

        case ELE_OP::DIVIDE:
            Out[tid] = In[tid] / (alpha * operand[tid]);
            break;
        
        default:
            break;
        }
    }
}

template <typename T>
__global__ void kelementwiseInplace(size_t N, T* In, T alpha, T* operand, ELE_OP op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op)
        {
        case ELE_OP::ADD:
            In[tid] = In[tid] + (alpha * operand[tid]);
            break;

        case ELE_OP::SUB:
            In[tid] = In[tid] - (alpha * operand[tid]);
            break;
        
        case ELE_OP::MULTIPLY:
            In[tid] = In[tid] * (alpha * operand[tid]);
            break;

        case ELE_OP::DIVIDE:
            In[tid] = In[tid] / (alpha * operand[tid]);
            break;
        
        default:
            break;
        }
    }
}
