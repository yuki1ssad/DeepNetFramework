#pragma once

#include <cassert>
#include "compute_graph.h"

class Network : public ComputeGraph
{
public:
    static Network* _trainer;
    cudaStream_t _cudaStream;
public:
    Network();
    Network(ComputeGraph* cptGraph, cudaStream_t cudaStream);
    ~Network();

    void to(cudaMemoryType type);
    std::vector<Tensor*> init(const std::vector<Tensor*>& inputs, const std::string& weightPath);
    std::vector<Tensor*> forward(std::vector<Tensor*>& inputs);
    void backward();
    void updateWeights(float lr);
};
