#include "Op_elementwise.h"

Op_elementwise::Op_elementwise(ELE_OP op, bool endOfGraph) : Operators(endOfGraph), _eleOp(op) {}

Op_elementwise::Op_elementwise(Tensor* A, Tensor* B, ELE_OP op) : Operators({A, B}, {new Tensor()}), _eleOp(op) {}

std::string Op_elementwise::typeStr()
{
    return std::string("Op_elementwise");
}

Op_elementwise* Op_elementwise::copy()
{
    return new Op_elementwise(_eleOp, _endOfGraph);
}

void Op_elementwise::inferShape()
{
    assert((_intensors.size() == 2) && "There should be two inputs.");
    assert((_intensors[0]->_shape.size() == _intensors[1]->_shape.size()) && "Two inputs should keep the same dim.");
    for (size_t i = 0; i < _intensors[0]->_shape.size(); ++i) {
        assert((_intensors[0]->_shape[i] == _intensors[1]->_shape[i]) && "Two inputs should keep the same subdim.");
    }
    _outtensors[0]->setShape(_intensors[0]->_shape);
}

void Op_elementwise::forward()
{
    dim3 BLOCK(32);
    dim3 GRID((_intensors[0]->_elementCount + BLOCK.x - 1) / BLOCK.x);

    size_t sharedMem = 0;
    kelementwise<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        _intensors[0]->_elementCount,
        _intensors[0]->_pdata,
        1.f,
        _intensors[1]->_pdata,
        _outtensors[0]->_pdata,
        _eleOp
    );
    cudaDeviceSynchronize();
    std::cout << "Op_elementwise::" << _eleOp << " forward output tensor " << *_outtensors[0] << std::endl;
}

void Op_elementwise::backward()
{
    if (_endOfGraph) {
        dim3 BLOCK(_outtensors[0]->_elementCount < 1024 ? _outtensors[0]->_elementCount : 1024);
        dim3 GRID(max((int)_outtensors[0]->_elementCount, 1024) / 1024 + 1);
        kmemset<<<GRID, BLOCK, 0, _cudaStream>>>(
            _outtensors[0]->_elementCount,
            _outtensors[0]->_pgradient,
            1.f
        );
    }

    dim3 BLOCK(32);
    dim3 GRID((_intensors[0]->_elementCount + BLOCK.x - 1) / BLOCK.x);
    size_t sharedMem = 0;
    switch (_eleOp)
    {
        case ELE_OP::ADD:
            cudaMemcpyAsync(_intensors[0]->_pgradient, _outtensors[0]->_pgradient, _outtensors[0]->_totalSize, cudaMemcpyDeviceToDevice, _cudaStream);
            cudaMemcpyAsync(_intensors[1]->_pgradient, _outtensors[0]->_pgradient, _outtensors[0]->_totalSize, cudaMemcpyDeviceToDevice, _cudaStream);
            cudaDeviceSynchronize();
            break;
        
        case ELE_OP::SUB:
            cudaMemcpyAsync(_intensors[0]->_pgradient, _outtensors[0]->_pgradient, _outtensors[0]->_totalSize, cudaMemcpyDeviceToDevice, _cudaStream);
            kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _outtensors[0]->_totalSize,
                _outtensors[0]->_pdata,
                -1.f,
                _intensors[1]->_pgradient,
                MAP_OP::MULTIPLY
            );
            cudaDeviceSynchronize();
            break;

        case ELE_OP::MULTIPLY:
            kelementwise<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _outtensors[0]->_totalSize,
                _intensors[1]->_pdata,
                1.f,
                _outtensors[0]->_pgradient,
                _intensors[0]->_pgradient,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();

            kelementwise<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _outtensors[0]->_totalSize,
                _intensors[0]->_pdata,
                1.f,
                _outtensors[0]->_pgradient,
                _intensors[1]->_pgradient,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();
            break;

        case ELE_OP::DIVIDE:
            // cal dividend grad
            kelementwise<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _outtensors[0]->_totalSize,
                _outtensors[0]->_pgradient,
                1.f,
                _intensors[1]->_pdata,
                _intensors[0]->_pgradient,
                ELE_OP::DIVIDE
            );
            cudaDeviceSynchronize();

            // cal divisor grad
            kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_totalSize,
                _intensors[1]->_pdata,
                -2.f,
                _intensors[1]->_pgradient,
                MAP_OP::POW
            );
            cudaDeviceSynchronize();

            kmapInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[1]->_totalSize,
                _intensors[1]->_pgradient,
                -1.f,
                MAP_OP::MULTIPLY
            );
            cudaDeviceSynchronize();

            kelementwiseInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[1]->_totalSize,
                _intensors[1]->_pgradient,
                1.f,
                _intensors[0]->_pdata,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();

            kelementwiseInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[1]->_totalSize,
                _intensors[1]->_pgradient,
                1.f,
                _outtensors[0]->_pgradient,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();
            break;
        default:
            break;
    }
}


