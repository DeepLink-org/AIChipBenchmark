#include "cuda_runtime.h"
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>

bool debug_print=false;

template<typename T, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void random_init(T *data, size_t size) {      
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand() / 10);
    }
}

template<typename T, typename std::enable_if<std::is_same<T, __half>::value>::type* = nullptr>
void random_init(T *data, size_t size) {      
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half_rn(float(rand() / 10));
    }
}

template <typename T>
void debug_init(T *data, size_t m, size_t n, bool trans) {
    float val = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t ind = i * n + j;
            if (trans)
                ind =j * m + i;
            val = val + 1;
            data[ind] = __float2half(val);
        }
    }
}

template <typename T>
void random_init(T *data, size_t m, size_t n, bool trans) {
    if (debug_print)
        debug_init(data, m, n, trans);
    else
        random_init(data, m*n);
}

template <typename T>
void print_mat(T* src, size_t size) {
    for (int i=0; i < size; i++) {
        std::cout << *(src+i) << " ";
    }
    std::cout << std::endl;
}

template <typename T, typename S>
void test_gemm(float& ms, int m, int k, int n, bool trans1, bool trans2, T *d_A, T *d_B, S *d_C, int algo, S *alpha, S *beta) {
    T *h_A, *h_B;
    S *h_C;
    cudaDataType_t AType, BType, CType, ComputeType;

    if(std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    }

    cudaMallocHost(&h_A, m * k * sizeof(T));
    cudaMallocHost(&h_B, k * n * sizeof(T));
    cudaMallocHost(&h_C, m * n * sizeof(S));

    random_init(h_A, m , k, trans1);
    random_init(h_B, k , n, trans2);
    if (debug_print) {
        print_mat(h_A, m*k);
        print_mat(h_B, n*k);
    }

    cudaMalloc(&d_A, m * k * sizeof(T));
    cudaMalloc(&d_B, k * n * sizeof(T));
    cudaMalloc(&d_C, m * n * sizeof(S));

    cudaMemcpy(d_A, h_A, m * k * sizeof(T), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // set the math mode to allow cuBLAS to use Tensor Cores
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; ++i) {
        cublasGemmEx(handle,
            trans1 ? CUBLAS_OP_T : CUBLAS_OP_N,
            trans2 ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            &alpha, 
            d_A, AType, trans1 ? k : m,
            d_B, BType, trans2 ? n : k,
            &beta,
            d_C, CType, m,
            ComputeType,
            static_cast<cublasGemmAlgo_t>(algo));
    }

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        cublasGemmEx(handle,
            trans1 ? CUBLAS_OP_T : CUBLAS_OP_N,
            trans2 ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            &alpha, 
            d_A, AType, trans1 ? k : m,
            d_B, BType, trans2 ? n : k,
            &beta,
            d_C, CType, m,
            ComputeType,
            static_cast<cublasGemmAlgo_t>(algo));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaMemcpy(h_C, d_C, m * n * sizeof(S), cudaMemcpyDefault);

    if (debug_print) {
        print_mat(h_C, m*n);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}

int main(int argc, char** argv) {
    if (argc == 8)
        debug_print = std::stoi(argv[7]);
    float ms;
    float f_alpha = 1, f_beta = 0;
    int algo = CUBLAS_GEMM_DEFAULT;
    if(std::stoi(argv[6]) == 16) {
        __half *hA, *hB, *hC;
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        test_gemm(ms, std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), hA, hB, hC, algo, &h_alpha, &h_beta);
    } 
    else if(std::stoi(argv[6]) == 32) {
        float *fA, *fB, *fC;
        test_gemm(ms, std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), fA, fB, fC, algo, &f_alpha, &f_beta);
    }
    std::cout << ms << std::endl;    
}