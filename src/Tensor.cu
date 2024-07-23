#include "Tensor.h"

std::vector<size_t> Tensor::show_elements = {64, 64, 3, 1};

Tensor::Tensor()
{
    std::cout << "Tensor constructed by default!" << std::endl;
}

Tensor::Tensor(std::vector<size_t> shape, cudaMemoryType memType, float* data) :
    _dataMemType(memType),
    _shape(shape)
{
    auto cnt = 1;
    for (auto num : shape) {
        cnt *= num;
    }
    _elementCount = cnt;
    _totalSize = _elementCount * sizeof(float);

    if (memType == cudaMemoryTypeHost) {
        _pdata = new float[_elementCount];
        _pgradient = new float[_elementCount];
        if (data) {
            memcpy(_pdata, data, _totalSize);
        }
    } else if (memType == cudaMemoryTypeDevice) {
        checkCudaErrors(cudaMalloc(&_pdata, _totalSize));
        checkCudaErrors(cudaMalloc(&_pgradient, _totalSize));
        if (data) {
            cudaMemcpy(_pdata, data, _totalSize, cudaMemcpyHostToDevice);
        }
    } else {
        std::cout << "Data type error!" << std::endl;
    }
    std::cout << "Tensor constructed by shape!" << std::endl;
}

Tensor::Tensor(const Tensor& tensor) :
    _dataMemType(tensor._dataMemType),
    _shape(tensor._shape),
    _elementCount(tensor._elementCount),
    _totalSize(tensor._totalSize)
{
    if (tensor._pdata) {
        if (_dataMemType == cudaMemoryTypeHost) {
            _pdata = new float[_elementCount];
            memcpy(_pdata, tensor._pdata, _totalSize);
        } else if (_dataMemType == cudaMemoryTypeDevice) {
            cudaMalloc(&_pdata, _totalSize);
            cudaMemcpy(_pdata, tensor._pdata, _totalSize, cudaMemcpyHostToDevice);
        } else {
            std::cout << "Data type error!" << std::endl;
        }
    }

    if (tensor._pgradient) {
        if (_dataMemType == cudaMemoryTypeHost) {
            _pgradient = new float[_elementCount];
            memcpy(_pgradient, tensor._pgradient, _totalSize);
        } else if (_dataMemType == cudaMemoryTypeDevice) {
            cudaMalloc(&_pgradient, _totalSize);
            cudaMemcpy(_pgradient, tensor._pgradient, _totalSize, cudaMemcpyHostToDevice);
        } else {
            std::cout << "Data type error!" << std::endl;
        }
    }
    std::cout << "Tensor constructed by copy constructor!" << std::endl;
}

Tensor::Tensor(Tensor &&tensor) :
    _dataMemType(tensor._dataMemType),
    _shape(tensor._shape),
    _elementCount(tensor._elementCount),
    _totalSize(tensor._totalSize)
{
    _pdata = tensor._pdata;
    tensor._pdata = nullptr;
    _pgradient = tensor._pgradient;
    tensor._pgradient = nullptr;
}

Tensor& Tensor::operator=(const Tensor& tensor)    // Copy Assignment Operator
{
    if (this != &tensor) {
        if (_dataMemType == cudaMemoryTypeHost) {
            delete[] _pdata;
            delete[] _pgradient;
        } else if (_dataMemType == cudaMemoryTypeDevice) {
            cudaFree(_pdata);
            cudaFree(_pgradient);
        } else {
            std::cout << "Data type error!" << std::endl;
        }
        _pdata = nullptr;
        _pgradient = nullptr;

        _shape = tensor._shape;
        _elementCount = tensor._elementCount;
        _totalSize = tensor._totalSize;

        cudaMemcpyKind direct;
        switch (_dataMemType) {
            case cudaMemoryTypeHost:
                if (tensor._pdata) {
                    _pdata = new float[_elementCount];
                }
                if (tensor._pgradient) {
                    _pgradient = new float[_elementCount];
                }

                if (tensor._dataMemType == cudaMemoryTypeHost) {
                    direct = cudaMemcpyHostToHost;
                } else if (tensor._dataMemType == cudaMemoryTypeHost) {
                    direct = cudaMemcpyDeviceToHost;
                }

                break;
            
            case cudaMemoryTypeDevice:
                if (tensor._pdata) {
                    cudaMalloc(&_pdata, _totalSize);
                }
                if (tensor._pgradient) {
                    cudaMalloc(&_pgradient, _totalSize);
                }

                if (tensor._dataMemType == cudaMemoryTypeHost) {
                    direct = cudaMemcpyHostToDevice;
                } else if (tensor._dataMemType == cudaMemoryTypeHost) {
                    direct = cudaMemcpyDeviceToDevice;
                }

                break;
            default:
                break;
        }

        if (tensor._pdata) {
            cudaMemcpy(_pdata, tensor._pdata, _totalSize, direct);
        }
        if (tensor._pgradient) {
            cudaMemcpy(_pgradient, tensor._pgradient, _totalSize, direct);
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& tensor)         // Move Assignment Operator
{
    if (this != &tensor) {
        if (_dataMemType != tensor._dataMemType) {
            std::cout << "Failed move cross-device!" << std::endl;
        }

        if (_dataMemType == cudaMemoryTypeHost) {
            delete[] _pdata;
            delete[] _pgradient;
        } else if (_dataMemType == cudaMemoryTypeDevice) {
            cudaFree(_pdata);
            cudaFree(_pgradient);
        } else {
            std::cout << "Data type error!" << std::endl;
        }

        _pdata = tensor._pdata;
        _pgradient = tensor._pgradient;
        tensor._pdata = nullptr;
        tensor._pgradient = nullptr;

        _shape = tensor._shape;
        _elementCount = tensor._elementCount;
        _totalSize = tensor._totalSize;
    }
    return *this;
}

// Tensor::Tensor &operator==(const Tensor& tensor) const;

Tensor::~Tensor()
{
    if (_dataMemType == cudaMemoryTypeHost) {
        delete[] _pdata;
        delete[] _pgradient;
    } else if (_dataMemType == cudaMemoryTypeHost) {
        cudaFree(_pdata);
        cudaFree(_pgradient);
    }
    _pdata = nullptr;
    _pgradient = nullptr;
}

Tensor Tensor::grad()
{
    if (_dataMemType == cudaMemoryTypeHost) {
        // TODO
    } else if (_dataMemType == cudaMemoryTypeHost) {
        // TODO
    }
    return Tensor();
}

void Tensor::setShape(std::vector<size_t> shape)
{
    _shape = shape;
     auto cnt = 1;
    for (auto num : shape) {
        cnt *= num;
    }
    _elementCount = cnt;
    _totalSize = _elementCount * sizeof(float);
}

void Tensor::allocMem()
{
    if (_dataMemType == cudaMemoryTypeHost) {
        delete[] _pdata;
        delete[] _pgradient;
        _pdata = new float[_totalSize];
        _pgradient = new float[_totalSize];
        for (int i = 0; i < _elementCount; ++i) {
            _pgradient[i] = 0.f;
        }
    } else if (_dataMemType == cudaMemoryTypeDevice) {
        checkCudaErrors(cudaFree(_pdata));
        checkCudaErrors(cudaFree(_pgradient));
        checkCudaErrors(cudaMalloc(&_pdata, _totalSize));
        checkCudaErrors(cudaMalloc(&_pgradient, _totalSize));
        dim3 BLOCK(_elementCount < 1024 ? _elementCount : 1024);
        dim3 GRID(max((int)_elementCount, 1024) /  + 1);
        kmemset<<<GRID, BLOCK>>>(
            _elementCount,
            _pgradient,
            0.f
        );
    } else {
        std::cout << "Data type error!" << std::endl;
}
}

void Tensor::to(cudaMemoryType targetMemType)
{
    if (_dataMemType == targetMemType) {
        return;
    }

    float* tmp = nullptr;
    if (targetMemType == cudaMemoryTypeHost) {
        if (_pdata) {
            tmp = new float[_elementCount];
            checkCudaErrors(cudaMemcpy(tmp, _pdata, _totalSize, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(_pdata));
            _pdata = tmp;
        } else {
            std::cout << "Move tensor with pullptr _pdata!" << std::endl;
        }

        if (_pgradient) {
            tmp = new float[_elementCount];
            checkCudaErrors(cudaMemcpy(tmp, _pgradient, _totalSize, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(_pgradient));
            _pgradient = tmp;
        } else {
            std::cout << "Move tensor with pullptr _pgradient!" << std::endl;
        }
    } else if (targetMemType == cudaMemoryTypeDevice) {
        if (_pdata) {
            checkCudaErrors(cudaMalloc(&tmp, _totalSize));
            checkCudaErrors(cudaMemcpy(tmp, _pdata, _totalSize, cudaMemcpyHostToDevice));
            delete[] _pdata;
            _pdata = tmp;
        } else {
            std::cout << "Move tensor with pullptr _pdata!" << std::endl;
        }

        if (_pgradient) {
            checkCudaErrors(cudaMalloc(&tmp, _totalSize));
            checkCudaErrors(cudaMemcpy(tmp, _pgradient, _totalSize, cudaMemcpyHostToDevice));
            delete[] _pgradient;
            _pgradient = tmp;
        } else {
            std::cout << "Move tensor with pullptr _pgradient!" << std::endl;
        }
    }
    _dataMemType = targetMemType;
}

void Tensor::fillDataRandom(float lower_bound, float upper_bound)
{
    if (!_pdata) {
        return;
    }

    if (_dataMemType == cudaMemoryTypeHost) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(lower_bound, upper_bound);
        for (int i = 0; i < _elementCount; i++) _pdata[i] = dis(gen);
    } else if (_dataMemType == cudaMemoryTypeDevice) {
        kinitializeRandom<<<(_elementCount + 511) / 512, 512>>>(_pdata, _elementCount, lower_bound, upper_bound);
        cudaDeviceSynchronize();
        // printf("after kernel function : %s\n",cudaGetErrorString(cudaGetLastError()));
        // this->to(cudaMemoryTypeHost);
        // std::cout << *this << std::endl;
    }
}

void Tensor::mirror(const std::map<Tensor*, Tensor*>& tensorMap, const std::map<Operators*, Operators*>& opMap)
{
    if (_pfrom) {
        tensorMap.at(this)->_pfrom = opMap.at(_pfrom);
    }
    for (Operators *op : _to) {
        tensorMap.at(this)->_to.push_back(opMap.at(op));
    }
}

void Tensor::updateWeights(float lr, cudaStream_t cudaStream)
{
    if (!_pdata || !_pgradient) {
        return;
    }

    if (_dataMemType == cudaMemoryTypeDevice) {
        dim3 BLOCK(32);
        dim3 GRID((_elementCount + BLOCK.x - 1) / BLOCK.x);
        kelementwiseInplace<<<GRID, BLOCK, 0, cudaStream>>>(
            _elementCount,
            _pdata,
            lr,
            _pgradient,
            ELE_OP::SUB
        );
    } else {
        std::cout << "UpdateWeights should be done on device!" << std::endl;
    }

}

std::ostream &operator<<(std::ostream& os, Tensor& tensor)
{
    if (tensor._dataMemType != cudaMemoryTypeHost) {
        Tensor show;
        show = tensor;
        os << "[device] " << show;
        return os;
    } else {
        // CHECK_EQ(tensor._data_memorytype, cudaMemoryTypeHost);
        if ( tensor._pdata == nullptr ) {
            // os << "tensor " << tensor._name << " empty"<< std::endl;
            os << "tensor is empty"<< std::endl;
            return os;
        }

        if (tensor._shape.size() > 2) {
            // for (int i = 0; ; i++) {
            //     if (i >= tensor.show_elements[tensor._shape.size() - 1]) {
            //         for (int i = 0; i < tensor._shape.size(); i++) {
            //             for (int i = 0; i < tensor._shape.size(); i++) os << "-";
            //             for (int i = 0; i < tensor._shape.size(); i++) os << " ";
            //         }
            //         break;
            //     } else if (i >= tensor._shape[0]) {
            //         break;
            //     } else {
            //         os << tensor[i];
            //     }
            // }
            // os << std::endl;
            
            // TODO
            
        } else if (tensor._shape.size() == 2) {
            // os << "Tensor " << tensor._name << std::endl;
            for (int i = 0; ; i++) {
                if (i > tensor.show_elements[tensor._shape.size() - 1]) {
                    os << ".\n.\n.\n";
                    break;
                } else if (i >= tensor._shape[0]) {
                    break;
                } else {
                    // os << tensor[i];
                }
            }
        } else if (tensor._shape.size() == 1) {
            for (int i = 0; ; i++) {
                if (i > tensor.show_elements[tensor._shape.size() - 1]) {
                    os << "...";
                    break;
                } else if (i >= tensor._shape[0]) {
                    break;
                } else {
                    os << std::fixed << std::setprecision(1) << tensor._pdata[i] << " ";
                }
            }
            os << std::endl;
        } else {
            os << "scalar: " << tensor._pdata[0] << std::endl;
        }
        return os;
    }
}