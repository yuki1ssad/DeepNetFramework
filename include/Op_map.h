#pragma once

#include "Operators.h"
#include "kernel_map.h"
#include "kernel_utils.h"
#include "kernel_elementwise.h"

class Op_Map : public Operators
{
public:
    MAP_OP _mapOp;
    float _operand;
public:
    Op_Map(){};
    Op_Map(MAP_OP op, float operand=0.f, bool endOfGraph=false);
    Op_Map(Tensor* A, MAP_OP op, float operand=0.f);
    ~Op_Map(){};

    virtual std::string typeStr();
    virtual Op_Map* copy();
    virtual void inferShape();
    virtual void forward();
    virtual void backward();
};

