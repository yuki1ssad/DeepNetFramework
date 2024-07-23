#include "compute_graph.h"

ComputeGraph::ComputeGraph(){}

ComputeGraph::~ComputeGraph(){}

std::list<Operators*>& ComputeGraph::getOpSeqs()    // acquire op sequence sorted topologically
{
    if (_opSeqs.empty()) {
        std::unordered_set<Operators*> initOps;
        for (Tensor* t : _inputTensors) {
            for (Operators* op : t->_to) {
                if (initOps.find(op) == initOps.end()) {
                    _opSeqs.push_back(op);
                    initOps.insert(op);
                }
            }
        }

        for (Operators* op : _opSeqs) {
            for (std::pair<Operators* const, bool>& pairNextOp : op->_nextOps) {
                Operators* nextOp = pairNextOp.first;
                nextOp->_preOps[op] = false;    // for updating indegree of topological sorting
                if (nextOp->indegree() == 0) {
                    _opSeqs.push_back(nextOp);
                }
            }
        }

        for (auto op : _opSeqs) {
            for (auto& p : op->_preOps) {
                p.second = true;
            }
        }
    }
    return _opSeqs;
}

std::vector<Tensor*> ComputeGraph::getOutputTensors()
{
    if (_outputTensors.empty()) {
        for (Operators* op : getOpSeqs()) {
            for (Tensor* t : op->_outtensors) {
                if (t->_to.empty()) {
                    _outputTensors.push_back(t);
                }
            }
        }
    }
    return _outputTensors;
}

void ComputeGraph::copy(std::vector<Tensor*>& inputTensors, std::vector<Tensor*>& weightTensors)
{
    std::map<Tensor*, Tensor*> tensorMapOn;
    std::map<Operators*, Operators*> opMapOn;
    for (Tensor* in_t : _inputTensors) {
        Tensor* copyed_t = new Tensor(*in_t);
        inputTensors.push_back(copyed_t);
        tensorMapOn[in_t] = copyed_t;
    }

    for (Tensor* w_t : _weightTensors) {
        Tensor* copyed_t = new Tensor(*w_t);
        inputTensors.push_back(copyed_t);
        tensorMapOn[w_t] = copyed_t;
    }

    for (Operators* op : getOpSeqs()) {
        opMapOn[op] = op->copy();
        for (Tensor* out_t : op->_outtensors) {
            Tensor* copyed_t = new Tensor(*out_t);
            tensorMapOn[out_t] = copyed_t;
        }
    }

    for (std::pair<Tensor*, Tensor*> p : tensorMapOn) {
        p.first->mirror(tensorMapOn, opMapOn);
    }

    for (std::pair<Operators*, Operators*> p : opMapOn) {
        p.first->mirror(tensorMapOn, opMapOn);
    }
}


