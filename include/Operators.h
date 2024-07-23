#pragma once

#include "Tensor.h"
#include <string>
#include <map>

class Tensor;

class Operators
{
public:
    std::string _name;
    bool _endOfGraph = false;
    cudaStream_t _cudaStream = cudaStreamDefault;

    std::map<Operators*, bool> _preOps = {};    // for topologicalSort
    std::map<Operators*, bool> _nextOps = {};

    std::vector<Tensor*> _intensors;
    std::vector<Tensor*> _outtensors;
    
public:
    Operators(/* args */){}
    Operators(bool endOfGraph) : _endOfGraph(endOfGraph) {}
    Operators(std::vector<Tensor*> intensors, std::vector<Tensor*> outtensors);
    Operators(const Operators& op) : _name(op._name) {}
    virtual ~Operators()
    {
        for (auto t : _outtensors) {
            delete t;
        }
    }

    virtual void mirror(const std::map<Tensor*, Tensor*>& tensorMap, const std::map<Operators*, Operators*>& opMap);
    virtual int indegree();
    virtual void setcudaStream(cudaStream_t cudaStream);

    virtual std::string typeStr() = 0;
    virtual Operators* copy() = 0;
    virtual void inferShape() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;


};
