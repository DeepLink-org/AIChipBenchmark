#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include "cuda_runtime.h"
#include "cudnn.h"

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

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

template<typename T>
void test_conv(float& ms, int in_n, int in_c, int in_h, int in_w,
                int filt_k, int filt_w, int filt_h,
                int pad_w, int pad_h, int str_w, int str_h, cudnnDataType_t data_type, T *in_data, T *filt_data, T *out_data, T *ws_data, cudnnTensorFormat_t tensor_format) {
    const int dil_h = 1;
    const int dil_w = 1;
    int filt_c = in_c;

    // create a CuDNN handle:
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            in_desc, tensor_format, data_type,    
            in_n, in_c, in_h, in_w));

    CUDA_CALL(cudaMalloc(
            &in_data, in_n * in_c * in_h * in_w * sizeof(T)));

    cudnnFilterDescriptor_t filt_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
            filt_desc, data_type, tensor_format,     
            filt_k, filt_c, filt_h, filt_w));

    CUDA_CALL(cudaMalloc(
        &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(T)));   

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h, pad_w, str_h, str_w, dil_h, dil_w,
            CUDNN_CONVOLUTION, data_type));   

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc,
            &out_n, &out_c, &out_h, &out_w));

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            out_desc, tensor_format, data_type,    
            out_n, out_c, out_h, out_w));

    CUDA_CALL(cudaMalloc(
            &out_data, out_n * out_c * out_h * out_w * sizeof(T)));    

    // set the math type to allow cuDNN to use Tensor Cores:
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);  

    // algorithm
    cudnnConvolutionFwdAlgo_t algo;

    // if the type is float32, choose the best algo
    if(std::is_same<T, float>::value) {
        int requestedAlgoCount = 2;
        int returnedAlgoCount = 0;
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(
            requestedAlgoCount);
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn,
                in_desc, filt_desc, conv_desc, out_desc,
                requestedAlgoCount, &returnedAlgoCount, perfResults.data()));
        bool algoFound = false;

        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);

        // choose best algo : every fwd algo is deterministic
        for (int i = 0; i < returnedAlgoCount; i++) {
            if (perfResults[i].status == CUDNN_STATUS_SUCCESS
                && (perfResults[i].memory < freeMem)) {
                algo = perfResults[i].algo;
                algoFound = true;
                break;
            }
        }

        if(algoFound == false) {
            std::cout << "algo not found";
            return;
        }
    } 
    else if(std::is_same<T, __half>::value) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

    CUDA_CALL(cudaMalloc(&ws_data, ws_size));

    // perform
    float alpha = 1.f;
    float beta = 0.f;

    T* h_in, *h_filt;
    size_t data_size = in_n*in_c*in_h*in_w;
    size_t filt_size = filt_k*filt_c*filt_h*filt_w;
    cudaMallocHost(&h_in, data_size * sizeof(T));
    cudaMallocHost(&h_filt, filt_size * sizeof(T));
    random_init(h_in, data_size);
    random_init(h_filt, filt_size);
    cudaMemcpy(in_data, h_in, data_size * sizeof(T), cudaMemcpyDefault);
    cudaMemcpy(filt_data, h_filt,  filt_size * sizeof(T), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i) {
            cudnnConvolutionForward(
                cudnn,
                &alpha, in_desc, in_data, filt_desc, filt_data,
                conv_desc, algo, ws_data, ws_size,
                &beta, out_desc, out_data);

    }
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
            cudnnConvolutionForward(
                cudnn,
                &alpha, in_desc, in_data, filt_desc, filt_data,
                conv_desc, algo, ws_data, ws_size,
                &beta, out_desc, out_data);

    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);

    // finalizing
    cudaFreeHost(h_in);
    cudaFreeHost(h_filt);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    CUDA_CALL(cudaFree(ws_data));
    CUDA_CALL(cudaFree(out_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDA_CALL(cudaFree(filt_data));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDA_CALL(cudaFree(in_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));
}

int main(int argc, char** argv) {

    float ms;
    cudnnTensorFormat_t  tensor_format = CUDNN_TENSOR_NHWC;
    // cudnnTensorFormat_t  tensor_format = CUDNN_TENSOR_NCHW;

    cudnnDataType_t data_type;
    if (std::stoi(argv[12]) == 32)  {
       data_type =  CUDNN_DATA_FLOAT;
       float *f_in_data, *f_filt_data, *f_out_data, *f_ws_data;
       test_conv(ms,
        std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
        std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
        std::stoi(argv[8]), std::stoi(argv[9]), std::stoi(argv[10]),  std::stoi(argv[11]), data_type, f_in_data, f_filt_data, f_out_data, f_ws_data, tensor_format);
    } 
    else if (std::stoi(argv[12]) == 16)  {
        data_type = CUDNN_DATA_HALF;
        __half *h_in_data, *h_filt_data, *h_out_data, *h_ws_data;
        test_conv(ms,
        std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
        std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
        std::stoi(argv[8]), std::stoi(argv[9]), std::stoi(argv[10]),  std::stoi(argv[11]), data_type, h_in_data, h_filt_data, h_out_data, h_ws_data, tensor_format);
    }
    std::cout << ms << std::endl;
    return 0;
}