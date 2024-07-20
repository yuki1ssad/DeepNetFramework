#include "Op_reduce.h"
    
Op_reduce::Op_reduce(bool endOfGraph) : Operators(endOfGraph) {}

Op_reduce::Op_reduce(Tensor* A) : Operators({A}, {new Tensor()}) {}

std::string Op_reduce::typeStr()
{
    return std::string("Op_reduce");
}

Op_reduce* Op_reduce::copy()
{
    return new Op_reduce(_endOfGraph);
}

void Op_reduce::inferShape()
{
    _outtensors[0]->setShape({});

}

void Op_reduce::forward()
{
    dim3 BLOCK(512);
    dim3 GRID;
    size_t sharedMem = BLOCK.x * sizeof(float);

    std::cout << "Op_reduce forward input tensor: " << *_intensors[0] << std::endl;
    size_t workNum = _intensors[0]->_elementCount;
    float* workSpace = nullptr;
    cudaMalloc(&workSpace, _intensors[0]->_totalSize);
    cudaMemcpyAsync(workSpace, _intensors[0]->_pdata, _intensors[0]->_totalSize, cudaMemcpyDeviceToDevice, _cudaStream);
    while (workNum != 1) {
        GRID = ceil(max(workNum, BLOCK.x * 2) / (BLOCK.X * 2));
        kreduceSum<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
            _intensors[0]->_elementCount,
            workNum,
            workSpace,
            workSpace
        );
        workNum = GRID.x;
    }
    cudaMemcpyAsync(_outtensors[0]->_pdata, workSpace, sizeof(float), cudaMemcpyDeviceToDevice, _cudaStream);
    cudaFreeAsync(workSpace, _cudaStream);
    std::cout << "Op_reduce forward output tensor: " << *_outtensors[0] << std::endl;
}

void Op_reduce::backward()
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

    dim3 BLOCK(32);
    dim3 GRID(ceil(max((int)_outtensors[0]->_elementCount, 32) / 32));
    size_t sharedMem = 0;

    kmemset_d<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        _intensors[0]->_elementCount,
        _intensors[0]->_pgradient,
        1.f,
        _outtensors[0]->_pgradient
    );
    cudaDeviceSynchronize();

    Tensor s = _intensors[0]->grad();
    s.to(cudaMemoryTypeHost);
}



