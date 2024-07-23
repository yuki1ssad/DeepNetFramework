#include "l1loss.h"

L1loss::L1loss(/* args */)
{
    _inputTensors.push_back(new Tensor());  // predict
    _inputTensors.push_back(new Tensor());  // target
    Operators* sub = new Op_elementwise(_inputTensors[0], _inputTensors[1], ELE_OP::SUB);
    Operators* abs = new Op_Map(sub->_outtensors[0], MAP_OP::ABS);
    Operators* reduceAdd = new Op_reduce(abs->_outtensors[0]);
    reduceAdd->_endOfGraph = true;
}

L1loss::~L1loss() {}


