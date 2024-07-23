#include "Op_matmul.h"

std::string Op_matmul::typeStr()
{
    return std::string("Op_matmul");
}

Op_matmul* Op_matmul::copy()
{
    return new Op_matmul(_endOfGraph);
}

void Op_matmul::inferShape()
{
    assert(_intensors.size() == 2 && "input should be two tensors.");
    assert(_intensors[0]->_shape.size() == 2 && "input tensor size should be 2.");
    assert(_intensors[1]->_shape.size() == 2 && "input tensor size should be 2.");
    assert(_intensors[0]->_shape[1] == _intensors[1]->_shape[0] && "input tensors shape should align.");
    _outtensors[0]->setShape({_intensors[0]->_shape[0], _intensors[1]->_shape[1]});
}

void Op_matmul::forward()
{
    // dim3 BLOCK(16, 16);
    dim3 BLOCK(2, 2);
    dim3 GRID(
        (_intensors[1]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_intensors[0]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    size_t sharedMem = BLOCK.x * BLOCK.y * sizeof(float) * 2;
    // _intensors[0]->to(cudaMemoryTypeHost);
    kmatmulNaive<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        _intensors[0]->_pdata,
        _intensors[1]->_pdata,
        _outtensors[0]->_pdata,
        _intensors[0]->_shape[0],
        _intensors[0]->_shape[1],
        _intensors[1]->_shape[1]
    );
    cudaDeviceSynchronize();
    std::cout << "Op_matmul forward" << std::endl;
}

void Op_matmul::backward()
{
    if (_endOfGraph) {
        dim3 BLOCK(_outtensors[0]->_elementCount < 1024 ? _outtensors[0]->_elementCount : 1024);
        dim3 GRID(ceil(max((int)_outtensors[0]->_elementCount, 1024) / 1024));
        kmemset<<<GRID, BLOCK, 0, _cudaStream>>>(
            _outtensors[0]->_elementCount,
            _outtensors[0]->_pgradient,
            1.f
        );
    }

    dim3 BLOCK, GRID;
    size_t sharedMem;

    // A @ B = C
    // cal grad of A
    BLOCK = dim3(16, 16);
    GRID = dim3(
        (_intensors[0]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_intensors[0]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    sharedMem = BLOCK.x * BLOCK.y * sizeof(float) * 2;

    auto transB = _intensors[1]->_pdata;
    ktranspose<<<GRID, BLOCK, sharedMem, _cudaStream>>>(_intensors[1]->_pdata, transB, _intensors[1]->_shape[0], _intensors[1]->_shape[1]);
    kmatmulNaive<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        _outtensors[0]->_pgradient,
        transB,
        _intensors[0]->_pgradient,
        _intensors[0]->_shape[0],
        _intensors[1]->_shape[1],
        _intensors[1]->_shape[0]
    );
    cudaDeviceSynchronize();

    // cal grad of B
    BLOCK = dim3(16, 16);
    GRID = dim3(
        (_intensors[1]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_intensors[1]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    sharedMem = BLOCK.x * BLOCK.y * sizeof(float) * 2;

    auto transA = _intensors[0]->_pdata;
    ktranspose<<<GRID, BLOCK, sharedMem, _cudaStream>>>(_intensors[0]->_pdata, transA, _intensors[0]->_shape[0], _intensors[0]->_shape[1]);
    kmatmulNaive<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        transA,
        _outtensors[0]->_pgradient,
        _intensors[1]->_pgradient,
        _intensors[0]->_shape[1],
        _intensors[0]->_shape[0],
        _intensors[1]->_shape[1]
    );
    cudaDeviceSynchronize();

    Tensor s1 = _intensors[0]->grad();
    Tensor s2 = _intensors[1]->grad();
    std::cout << "matmul backward get input tensor[0] grad: " << s1 <<std::endl;
    std::cout << "matmul backward get input tensor[1] grad: " << s2 <<std::endl;
}

void Op_matmul::setcudaStream(cudaStream_t cudaStream)
{
    _cudaStream = cudaStream;
}
