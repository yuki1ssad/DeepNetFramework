#include "Op_map.h"

Op_Map::Op_Map(MAP_OP op, float operand, bool endOfGraph) : Operators(endOfGraph), _mapOp(op), _operand(operand) {}

Op_Map::Op_Map(Tensor* A, MAP_OP op, float operand) : Operators({A}, {new Tensor()}), _mapOp(op), _operand(operand) {}

std::string Op_Map::typeStr()
{
    return std::string("Op_Map");
}

Op_Map* Op_Map::copy()
{
    return new Op_Map(_mapOp, _operand, _endOfGraph);
}

void Op_Map::inferShape()
{
    _outtensors[0]->setShape(_intensors[0]->_shape);
}

void Op_Map::forward()
{
    dim3 BLOCK(32);
    dim3 GRID((_intensors[0]->_elementCount + BLOCK.x - 1) / BLOCK.x);
    size_t sharedMem = 0;

    kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
        _intensors[0]->_elementCount,
        _intensors[0]->_pdata,
        _operand,
        _outtensors[0]->_pdata,
        _mapOp
    );
    cudaDeviceSynchronize();
    std::cout << "Op_Map::" << _mapOp << " forward. Output tensor: " << *_outtensors[0] << std::endl;
}

void Op_Map::backward()
{
    if (_endOfGraph) {
        dim3 BLOCK(_outtensors[0]->_elementCount < 1024 ? _outtensors[0]->_elementCount : 1024);
        dim3 GRID(max((int(_outtensors[0]->_elementCount), 1024)) / 1024 + 1);
        kmemset<<<GRID, BLOCK, 0, _cudaStream>>>(
            _outtensors[0]->_elementCount,
            _outtensors[0]->_pgradient,
            1.f
        );
    }

    dim3 BLOCK(32);
    dim3 GRID((_intensors[0]->_elementCount + BLOCK.x - 1) / BLOCK.x);
    size_t sharedMem = 0;

    switch (_mapOp)
    {
        case MAP_OP::ADD:
            cudaMemcpy(_intensors[0]->_pgradient, _outtensors[0]->_pgradient, _outtensors[0]->_totalSize, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            break;

        case MAP_OP::MULTIPLY:
            kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _outtensors[0]->_pgradient,
                _operand,
                _intensors[0]->_pgradient,
                MAP_OP::MULTIPLY
            );
            cudaDeviceSynchronize();
            break;
        
        case MAP_OP::POW:
            kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _intensors[0]->_pdata,
                _operand - 1,
                _intensors[0]->_pgradient,
                MAP_OP::POW
            );
            cudaDeviceSynchronize();

            kmapInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _intensors[0]->_pgradient,
                _operand,
                MAP_OP::MULTIPLY
            );
            cudaDeviceSynchronize();

            kelementwiseInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _intensors[0]->_pgradient,
                1.f,
                _outtensors[0]->_pgradient,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();
            break;

        case MAP_OP::ABS:
            kmap<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _intensors[0]->_pdata,
                0.f,
                _intensors[0]->_pgradient,
                MAP_OP::SIGN
            );
            cudaDeviceSynchronize();

            kelementwiseInplace<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _intensors[0]->_pgradient,
                1.f,
                _outtensors[0]->_pgradient,
                ELE_OP::MULTIPLY
            );
            cudaDeviceSynchronize();

            break;
            
        case MAP_OP::LOG:
            kelementwise<<<GRID, BLOCK, sharedMem, _cudaStream>>>(
                _intensors[0]->_elementCount,
                _outtensors[0]->_pgradient,
                1.f,
                _intensors[0]->_pdata,
                _intensors[0]->_pgradient,
                ELE_OP::DIVIDE
            );
            break;
            
        default:
            break;
    }
    Tensor s = _intensors[0]->grad();
    std::cout << _name << _mapOp << " backward get input tensor[0] grad:" << s << std::endl;
}

