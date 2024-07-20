#pragma once

#include <cassert>
#include "Operators.h"
#include "kernel_elementwise.h"
#include "kernel_map.h"
#include "kernel_utils.h"

class Op_elementwise : public Operators
{
public:
    ELE_OP _eleOp;
public:
    Op_elementwise(){};
    Op_elementwise(ELE_OP op, bool endOfGraph);
    Op_elementwise(Tensor* A, Tensor* B, ELE_OP op);
    ~Op_elementwise(){};

    virtual std::string typeStr();
    virtual Op_elementwise* copy();
    virtual void inferShape();
    virtual void forward();
    virtual void backward();
};

