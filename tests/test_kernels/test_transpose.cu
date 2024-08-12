#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "Tensor.h"
#include "tools_cuda.h"
#include "tools_common.h"
#include "kernel_transpose.h"

class test_transpose:
    public testing::TestWithParam<
        std::tuple<
            uint,  // BLOCK
            size_t,  // m
            size_t  // n
        >
    >
{
public:
    uint TILE_DIM;
    size_t m, n;

    size_t ele_count, sz;
    float   *X_host, *X_device,\
            *Y_ground_truth_host, *Y_ground_truth_device,\
            *Y_predict_host, *Y_predict_device;

    cublasHandle_t handle;

    test_transpose();
    ~test_transpose();
};

test_transpose::test_transpose() {
    std::tie(TILE_DIM, m, n) = GetParam();

    ele_count = m * n ;
    sz = ele_count * sizeof(float);

    X_host = (float*)malloc(sz);
    checkCudaErrors(cudaMalloc(&X_device, sz));
    Y_ground_truth_host = (float*)malloc(sz);
    checkCudaErrors(cudaMalloc(&Y_ground_truth_device, sz));
    Y_predict_host = (float*)malloc(sz);
    checkCudaErrors(cudaMalloc(&Y_predict_device, sz));

    // auto X_gen = get_rand_data_gen<float, std::uniform_real_distribution>(1.f, 1.5f);
    auto X_gen = [](std::vector<int> i){return 1000 * (i[0] + 1) + (i[1] + 1);};
#pragma omp parallel for
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            X_host[r * n + c] = X_gen({r, c});
        }
    }

    checkCudaErrors(cudaMemcpy(X_device, X_host, sz, cudaMemcpyHostToDevice));
    cublasCreate(&handle);
}

test_transpose::~test_transpose() {
    free(X_host);
    checkCudaErrors(cudaFree(X_device));
    free(Y_ground_truth_host);
    checkCudaErrors(cudaFree(Y_ground_truth_device));
    free(Y_predict_host);
    checkCudaErrors(cudaFree(Y_predict_device));

    cublasDestroy(handle);
}


INSTANTIATE_TEST_SUITE_P(
    exhaustive_combine,
    test_transpose,
    testing::Combine(
        testing::Values(  // TILE.x == TILE.y
            8
        ),
        testing::Values(
            32,
            64,
            128,
            256,
            512,
            1024,
            2 * 1024,
            4 * 1024,
            8 * 1024,
            16 * 1024
        ),
        testing::Values(
            32,
            64,
            128,
            256,
            512,
            1024,
            2 * 1024,
            4 * 1024,
            8 * 1024,
            16 * 1024
        )
    )
);


INSTANTIATE_TEST_SUITE_P(
    design,
    test_transpose,
    testing::Combine(
        testing::Values(
            // 1
            // 4
            8
            // 16
            // 32
        ),
        testing::Values(
            // 32
            512
        ),
        testing::Values(
            // 32
            512
        )
    )
);


TEST_P(test_transpose, ktranspose){
    dim3 BLOCK(TILE_DIM, TILE_DIM);
    dim3 GRID(
        (n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );
    ktranspose<<<GRID, BLOCK, 0, cudaStreamDefault>>>(
        X_device,
        Y_predict_device,
        m,
        n
    );
    checkCudaErrors(cudaMemcpy(Y_predict_host, Y_predict_device, sz, cudaMemcpyDeviceToHost));

    float alpha = 1.f, beta = 0.f;
    cublasSgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        m, n,
        &alpha, X_device, n,
        &beta, X_device, n,
        Y_ground_truth_device, m
    );
    checkCudaErrors(cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, sz, cudaMemcpyDeviceToHost));

    for (int r = 0; r < n; r++)
        for (int c = 0; c < m; c++)
            ASSERT_LE(
                abs((Y_ground_truth_host[r * m + c] - Y_predict_host[r * m + c]) / Y_ground_truth_host[r * m + c]),
                0.0002
            )   << "\npos[" << r << ", " << c << "]:"
                << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * m + c]
                << "\nY_predict_host: " <<  Y_predict_host[r * m + c];
}


TEST_P(test_transpose, ktransposeSharedMem){
    dim3 BLOCK = dim3(TILE_DIM, TILE_DIM);
    dim3 GRID = dim3(
        (n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );
    size_t shared_mem = TILE_DIM * TILE_DIM * sizeof(float);
    ktransposeSharedMem<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        X_device,
        Y_predict_device,
        m,
        n
    );
    checkCudaErrors(cudaMemcpy(Y_predict_host, Y_predict_device, sz, cudaMemcpyDeviceToHost));

    float alpha = 1.f, beta = 0.f;
    cublasSgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        m, n,
        &alpha, X_device, n,
        &beta, X_device, n,
        Y_ground_truth_device, m
    );
    checkCudaErrors(cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, sz, cudaMemcpyDeviceToHost));

    for (int r = 0; r < n; r++)
        for (int c = 0; c < m; c++)
            ASSERT_LE(
                abs((Y_ground_truth_host[r * m + c] - Y_predict_host[r * m + c]) / Y_ground_truth_host[r * m + c]),
                0.0002
            )   << "\npos[" << r << ", " << c << "]:"
                << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * m + c]
                << "\nY_predict_host: " <<  Y_predict_host[r * m + c];
}


TEST_P(test_transpose, ktransposeSharedMemMinbkcft){
    dim3 BLOCK = dim3(TILE_DIM, TILE_DIM);
    dim3 GRID = dim3(
        (n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );
    size_t shared_mem = TILE_DIM * TILE_DIM * sizeof(float);
    ktransposeSharedMemMinbkcft<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        X_device,
        Y_predict_device,
        m,
        n
    );
    checkCudaErrors(cudaMemcpy(Y_predict_host, Y_predict_device, sz, cudaMemcpyDeviceToHost));

    // Tensor Y_P({n, m}, cudaMemoryTypeHost, Y_predict_host);
    // std::cout << Y_P;

    float alpha = 1.f, beta = 0.f;
    cublasSgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        m, n,
        &alpha, X_device, n,
        &beta, X_device, n,
        Y_ground_truth_device, m
    );
    checkCudaErrors(cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, sz, cudaMemcpyDeviceToHost));

    // Tensor Y_G({n, m}, cudaMemoryTypeHost, Y_ground_truth_host);
    // std::cout << Y_G;

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < m; c++) {
            ASSERT_LE(
                abs((Y_ground_truth_host[r * m + c] - Y_predict_host[r * m + c]) / Y_ground_truth_host[r * m + c]),
                0.0002
            )   << "\npos[" << r << ", " << c << "]:"
                << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * m + c]
                << "\nY_predict_host: " <<  Y_predict_host[r * m + c];
        }
    }
        
}


// TEST_P(test_transpose, ktranspose_smem_4xvec4){
//     int BLOCK_TILE_DIM = TILE_DIM;
//     int TILE_DIM = 4 * BLOCK_TILE_DIM;
//     dim3 BLOCK = dim3(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
//     dim3 GRID = dim3(
//         ceil(n, TILE_DIM) / TILE_DIM,
//         ceil(m, TILE_DIM) / TILE_DIM
//     );
//     size_t shared_mem = TILE_DIM * TILE_DIM * sizeof(float);
//     ktranspose_smem_4xvec4<float, float4><<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
//         m,
//         n,
//         X_device,
//         Y_predict_device
//     );
//     checkCudaErrors(
//         cudaMemcpy(
//             Y_predict_host,
//             Y_predict_device,
//             sz,
//             cudaMemcpyDeviceToHost
//         )
//     );

//     // Tensor Y_P({n, m}, cudaMemoryTypeHost, Y_predict_host);
//     // std::cout << Y_P;

//     float alpha = 1.f, beta = 0.f;
//     cublasSgeam(
//         handle,
//         CUBLAS_OP_T, CUBLAS_OP_T,
//         m, n,
//         &alpha, X_device, n,
//         &beta, X_device, n,
//         Y_ground_truth_device, m
//     );
//     checkCudaErrors(
//         cudaMemcpy(
//             Y_ground_truth_host,
//             Y_ground_truth_device,
//             sz,
//             cudaMemcpyDeviceToHost
//         )
//     );

//     // Tensor Y_G({n, m}, cudaMemoryTypeHost, Y_ground_truth_host);
//     // std::cout << Y_G;

//     for (int r = 0; r < n; r++)
//         for (int c = 0; c < m; c++)
//             ASSERT_LE(
//                 abs((Y_ground_truth_host[r * m + c] - Y_predict_host[r * m + c]) / Y_ground_truth_host[r * m + c]),
//                 0.0002
//             )   << "\npos[" << r << ", " << c << "]:"
//                 << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * m + c]
//                 << "\nY_predict_host: " <<  Y_predict_host[r * m + c];
// }


TEST_P(test_transpose, ktransposeSharedMem4xvec4Minbkcft){
    int BLOCK_TILE_DIM = TILE_DIM;
    int TILE_DIM = 4 * BLOCK_TILE_DIM;
    dim3 BLOCK = dim3(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
    dim3 GRID = dim3(
        (n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );
    size_t shared_mem = TILE_DIM * TILE_DIM * sizeof(float);
    ktransposeSharedMem4xvec4Minbkcft<float, float4><<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        X_device,
        Y_predict_device,
        m,
        n
    );
    checkCudaErrors(cudaMemcpy(Y_predict_host, Y_predict_device, sz, cudaMemcpyDeviceToHost));

    // Tensor Y_P({n, m}, cudaMemoryTypeHost, Y_predict_host);
    // std::cout << Y_P;

    float alpha = 1.f, beta = 0.f;
    cublasSgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        m, n,
        &alpha, X_device, n,
        &beta, X_device, n,
        Y_ground_truth_device, m
    );
    checkCudaErrors(cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, sz, cudaMemcpyDeviceToHost));

    // Tensor Y_G({n, m}, cudaMemoryTypeHost, Y_ground_truth_host);
    // std::cout << Y_G;

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < m; c++) {
            ASSERT_LE(
                abs((Y_ground_truth_host[r * m + c] - Y_predict_host[r * m + c]) / Y_ground_truth_host[r * m + c]),
                0.0002
            )   << "\npos[" << r << ", " << c << "]:"
                << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * m + c]
                << "\nY_predict_host: " <<  Y_predict_host[r * m + c];
        }
    }
}
