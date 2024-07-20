#pragma once

#include <vector>
#include <iostream>
#include <cstring>  // memcpy
#include <random>   // std::random_device ...
#include <iomanip>  // 包含setprecision

#include <cuda_runtime.h>

#include "Operators.h"

class Operators;

class Tensor
{
public:
    static std::vector<size_t> show_elements;

    cudaMemoryType _dataMemType = cudaMemoryTypeHost;
    std::vector<size_t> _shape = {};
    size_t _elementCount = 0;
    size_t _totalSize = 0;
    float* _pdata = nullptr;
    float* _pgradient = nullptr;

    Operators *_pfrom = nullptr;
    std::vector<Operators*> _to = {};
    
public:

    Tensor();
    Tensor(std::vector<size_t> shape, cudaMemoryType memType=cudaMemoryTypeHost, float* data=nullptr);
    Tensor(const Tensor& tensor);
    Tensor(Tensor &&tensor);
    Tensor &operator=(const Tensor& tensor);    // Copy Assignment Operator
    Tensor &operator=(Tensor&& tensor);         // Move Assignment Operator
    Tensor &operator==(const Tensor& tensor) const;
    ~Tensor();

    Tensor grad();
    void setShape(std::vector<size_t> shape);
    void allocMem();
    void to(cudaMemoryType targetMemType);
    void fillDataRandom(float lower_bound, float upper_bound);
    void updateWeights(float alpha, cudaStream_t cudastream);
    friend std::ostream &operator<<(std::ostream& os, Tensor& tensor);
};


