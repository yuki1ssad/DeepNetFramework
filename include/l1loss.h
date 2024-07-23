#pragma once

#include "compute_graph.h"
#include "Op_elementwise.h"
#include "Op_map.h"
#include "Op_reduce.h"

class L1loss : public ComputeGraph
{
public:
    L1loss(/* args */);
    ~L1loss();
};

