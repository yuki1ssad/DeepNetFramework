#pragma once

#include "Operators.h"


Operators::Operators(std::vector<Tensor*> intensors, std::vector<Tensor*> outtensors) :
    _intensors(intensors),
    _outtensors(outtensors)
{
    for (auto t : intensors) {
        t->to.push_back(this);
        if (t->pfrom) {
            t->_pfrom->_nextOps[this] = true;
            _preOps[t->_pfrom] = true;
        }
    }

    for (auto t : outtensors) {
        t->_pfrom = this;
    }
}


virtual void Operators::mirror(const std::map<Tensor*, Tensor*>& tensorMap, const std::map<Operators*, Operators*>& opMap)
{
    for (auto t : _intensors) {
        opMap.at(this)->_intensors.push_back(tensorMap.at(t));
    }

    for (auto t : _outtensors) {
        opMap.at(this)->_outtensors.push_back(tensorMap.at(t));
    }

    for (std::pair<Operators*, bool> p : _preOps) {
        Operators* op = p.first;
        opMap.at(this)->_preOps[opMap.at(op)] = true;
    }

    for (std::pair<Operators*, bool> p : _nextOps) {
        Operators* op = p.first;
        opMap.at(this)->_nextOps[opMap.at(op)] = true;
    }
}


virtual int Operators::indegree()
{
    int inds = 0;
    for (auto p : _preOps) {
        inds += p.second;
    }
    return inds;
}


