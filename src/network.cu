#include "network.h"

Network::Network(){}

Network::Network(ComputeGraph* cptGraph, cudaStream_t cudaStream) : 
    _cudaStream(cudaStream)
{
    for (Operators* op : getOpSeqs()) {
        op->setcudaStream(cudaStream);
    }
    cptGraph->copy(_inputTensors, _weightTensors);
}

Network::~Network(){}

void Network::to(cudaMemoryType type)
{
    for (Tensor* in_t : _inputTensors) {
        in_t->to(type);
    }

    for (Tensor* w_t : _weightTensors) {
        w_t->to(type);
    }

    for (Operators* op : this->getOpSeqs()) {
        for (Tensor* out_t : op->_outtensors) {
            out_t->to(type);
        }
    }
}

std::vector<Tensor*> Network::init(const std::vector<Tensor*>& inputs, const std::string& weightPath)
{
    assert(_inputTensors.size() == inputs.size());
    for (int i = 0; i < _inputTensors.size(); ++i) {
        *_inputTensors[i] = *inputs[i];
        _inputTensors[i]->allocMem();
    }

    for (Tensor* weight : _weightTensors) {
        weight->allocMem();
    }

    int i = 0;
    for (Operators* op : getOpSeqs()) {
        op->_name = op->typeStr() + "_" + std::to_string(i++);
        op->inferShape();
        for (Tensor* t : op->_outtensors) {
            t->allocMem();
        }
    }
    return getOutputTensors();
}

std::vector<Tensor*> Network::forward(std::vector<Tensor*>& inputs)
{
    assert(_inputTensors.size() == inputs.size());
    for (int i = 0; i < _inputTensors.size(); ++i) {
        // inputs[i]->to(cudaMemoryTypeDevice);
        *_inputTensors[i] = *inputs[i];
    }
    
    for (Operators* op : getOpSeqs()) {
        op->forward();
    }
    cudaDeviceSynchronize();
    return getOutputTensors();
}

void Network::backward()
{
    for (auto it : getOpSeqs()) {
        it->backward();
    }
}

void Network::updateWeights(float lr)
{
    for (auto w : _weightTensors) {
        w->updateWeights(lr, _cudaStream);
    }
}
