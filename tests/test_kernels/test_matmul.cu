#include <random>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "Tensor.h"
#include "tools_cuda.h"
#include "kernel_matmul.h"
#include "tools_common.h"

class test_matmul:
    public testing::TestWithParam<
        std::tuple<
            int,  // m
            int,  // n
            int,  // k
            std::function<float(const std::vector<int>&)>,  // W gen
            std::function<float(const std::vector<int>&)>,  // X gen
            dim3  // block
        >
    >
{
public:
    int m ,n, k;
    std::function<float(const std::vector<int>&)> W_gen, X_gen;
    dim3 BLOCK;
    
    float alpha = 1.f, beta = 0.f;
    size_t W_size, X_size, Y_size;
    float *W_host, *X_host, *Y_ground_truth_host, *Y_predict_host, *W_device, *X_device, *Y_ground_truth_device, *Y_predict_device;
    dim3 GRID;
    size_t shared_mem;

    cublasHandle_t handle = nullptr;

    test_matmul();
    ~test_matmul();
};

test_matmul::test_matmul() {
    std::tie(
        m,
        n,
        k,
        W_gen,
        X_gen,
        BLOCK
    ) = GetParam();

    cublasCreate(&handle);
    W_size = m * k * sizeof(float);
    X_size = k * n * sizeof(float);
    Y_size = m * n * sizeof(float);

    W_host = (float*)malloc(W_size);
    X_host = (float*)malloc(X_size);
    Y_ground_truth_host = (float*)malloc(Y_size);
    Y_predict_host = (float*)malloc(Y_size);
    checkCudaErrors(cudaMalloc(&W_device, W_size));
    checkCudaErrors(cudaMalloc(&X_device, X_size));
    checkCudaErrors(cudaMalloc(&Y_ground_truth_device, Y_size));
    checkCudaErrors(cudaMalloc(&Y_predict_device, Y_size));

    #pragma omp parallel for
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < k; c++) {
            W_host[r * k + c] = W_gen({r, c});
            // W_host[r * k + c] = 1.;
        }
    }

    #pragma omp parallel for
    for (int r = 0; r < k; r++) {
        for (int c = 0; c < n; c++) {
            X_host[r * n + c] = X_gen({r, c});
            // X_host[r * n + c] = 1.;
        }
    }
    checkCudaErrors(cudaMemcpy(W_device, W_host, W_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(X_device, X_host, X_size, cudaMemcpyHostToDevice));


    GRID = dim3((n + BLOCK.x - 1)/BLOCK.x, (m + BLOCK.y - 1)/BLOCK.y);
    shared_mem = BLOCK.x * BLOCK.y * sizeof(float) * 2;
}

test_matmul::~test_matmul() {
    free(W_host);
    free(X_host);
    free(Y_ground_truth_host);
    free(Y_predict_host);
    checkCudaErrors(cudaFree(W_device));
    checkCudaErrors(cudaFree(X_device));
    checkCudaErrors(cudaFree(Y_ground_truth_device));
    checkCudaErrors(cudaFree(Y_predict_device));

    cublasDestroy(handle);
}


INSTANTIATE_TEST_SUITE_P(
    design,
    test_matmul,
    testing::Values(
        std::make_tuple(
            // 256,
            // 256,
            // 256,
            // 512,
            // 512,
            // 512,
            1024,
            1024,
            1024,
            get_rand_data_gen<float, std::uniform_real_distribution>(-1.f, 1.f),
            get_rand_data_gen<float, std::uniform_real_distribution>(-2.f, 2.f),
            dim3(16, 16)
        )
    )
);


INSTANTIATE_TEST_SUITE_P(
    exhaustive_combine_lite,
    test_matmul,
    testing::Combine(
        testing::Values(1, 2, 3, 8, 64),
        testing::Values(1, 2, 128),
        testing::Values(1, 2, 256),
        testing::Values(
            get_rand_data_gen<float, std::uniform_real_distribution>(-1.f, 1.f)
        ),
        testing::Values(
            get_rand_data_gen<float, std::uniform_real_distribution>(-1.f, 1.f)
        ),
        testing::Values(
            dim3(2, 2),
            dim3(8, 8),
            dim3(16, 16)
            // dim3(32, 32) todo: cudaErrorLaunchOutOfResources
        )
    )
);


INSTANTIATE_TEST_SUITE_P(
    exhaustive_combine,
    test_matmul,
    testing::Combine(
        testing::Values(1, 2, 3, 8, 64, 512, 4 * 1024),
        testing::Values(1, 2, 128, 2 * 1024),
        testing::Values(1, 2, 256, 3 * 1023),
        testing::Values(
            get_rand_data_gen<float, std::uniform_real_distribution>(1.f, 2.f)
        ),
        testing::Values(
            get_rand_data_gen<float, std::uniform_real_distribution>(2.f, 3.f)
        ),
        testing::Values(
            dim3(2, 2),
            dim3(8, 8),
            dim3(16, 16)
            // dim3(32, 32) todo: cudaErrorLaunchOutOfResources
        )
    )
);

TEST_P(test_matmul, positive){
    std::vector<size_t> W_shape = std::vector<size_t>{size_t(m), size_t(k)};
    Tensor show_W(W_shape, cudaMemoryTypeDevice, W_device);
    VLOG(8) << "show W \n" << show_W;

    std::vector<size_t> X_shape = std::vector<size_t>{size_t(k), size_t(n)};
    Tensor show_X(X_shape, cudaMemoryTypeDevice, X_device);
    VLOG(8) << "show X \n" << show_X;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        X_device,
        n,
        W_device,
        k,
        &beta,
        Y_ground_truth_device,
        n
    );
    cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, Y_size, cudaMemcpyDeviceToHost);

    Tensor gt({size_t(m), size_t(n)}, cudaMemoryTypeHost, Y_ground_truth_host);
    VLOG(8) << "show gt \n" << gt;

    kmatmulNaive<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        W_device,
        X_device,
        Y_predict_device,
        m,
        k,
        n
    );
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    cudaMemcpy(Y_predict_host, Y_predict_device, Y_size, cudaMemcpyDeviceToHost);

    Tensor pd({size_t(m), size_t(n)}, cudaMemoryTypeHost, Y_predict_host);
    VLOG(8) << "show pd \n" << pd;

    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            ASSERT_LE(
                // abs(Y_predict_host[r * n + c] - Y_ground_truth_host[r * n + c]) / Y_ground_truth_host[r * n + c],
                // 0.01
                abs(Y_predict_host[r * n + c] - Y_ground_truth_host[r * n + c]),
                1e-3
            ) << "\nm: " + std::to_string(m) +\
                 "\nn: " + std::to_string(n) +\
                 "\nk: " + std::to_string(k) +\
                 "\nGRID: " << GRID\
                 << "\nBLOCK: " << BLOCK\
                 << "\nat [" << std::to_string(r) << ", " << std::to_string(c) << "]"
                 << Y_predict_host[r * n + c] << " vs " << Y_ground_truth_host[r * n + c];
        }
    }
}