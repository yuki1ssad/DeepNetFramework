#pragma once

#include <list>
#include <unordered_set>
#include <map>

#include "Tensor.h"
#include "Operators.h"

class ComputeGraph
{
public:
    std::vector<Tensor*> _inputTensors;
    std::vector<Tensor*> _weightTensors;
    std::list<Operators*> _opSeqs;
    std::vector<Tensor*> _outputTensors;

    
public:
    ComputeGraph();
    ~ComputeGraph();

    std::list<Operators*>& getOpSeqs();
    std::vector<Tensor*> getOutputTensors();

    void copy(std::vector<Tensor*>& inputTensors, std::vector<Tensor*>& weightTensors);
};

