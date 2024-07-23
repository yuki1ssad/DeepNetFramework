#pragma once

#include <cassert>
#include <string>
#include "Operators.h"
#include "kernel_matmul.h"
#include "kernel_transpose.h"
#include "kernel_utils.h"

class Op_matmul : public Operators
{
public:
    std::string _name = "Op_matmul";
public:
    Op_matmul(){}
    Op_matmul(bool endOfGraph) : Operators(endOfGraph) {}
    Op_matmul(Tensor* A, Tensor* B) : Operators({A, B}, {new Tensor()}) {}
    ~Op_matmul(){}

    virtual std::string typeStr();
    virtual Op_matmul* copy();
    virtual void inferShape();
    virtual void forward();
    virtual void backward();

    void setcudaStream(cudaStream_t cudaStream);
};
