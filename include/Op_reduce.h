#pragma once

#include "Operators.h"
#include "kernel_reduce.h"
#include "kernel_map.h"
#include "kernel_utils.h"

class Op_reduce : public Operators
{
public:
    Op_reduce(){}
    Op_reduce(bool endOfGraph);
    Op_reduce(Tensor* A);
    ~Op_reduce(){}

    virtual std::string typeStr();
    virtual Op_reduce* copy();
    virtual void inferShape();
    virtual void forward();
    virtual void backward();
};


