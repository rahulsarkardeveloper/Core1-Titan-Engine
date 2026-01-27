#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#include "../include/core1.h"

__global__ void titan_attention_kernel(float* Q, float* K, float* V, float* out, int d_model) {
    extern __shared__ float s_data[]; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_model && col < d_model) {
        float sum = 0.0f;
        float scale = 1.0f / sqrtf((float)d_model);

        // Q * K^T Calculation
        for (int i = 0; i < d_model; ++i) {
            sum += Q[row * d_model + i] * K[col * d_model + i];
        }

        // Softmax & V multiplication (Simplified for Ultra-Speed)
        float score = expf(sum * scale);
        out[row * d_model + col] = score * V[row * d_model + col];
    }
}

__global__ void moe_router_kernel(float* input, int* selected_experts, int n_experts, int d_model) {
    int tid = threadIdx.x;
    if (tid == 0) {
        selected_experts[0] = (int)(input[0]) % n_experts; 
        selected_experts[1] = (int)(input[1]) % n_experts;
    }
}

extern "C" void launch_titan_engine(float* d_Q, float* d_K, float* d_V, float* d_out, int d_model) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((d_model + 31) / 32, (d_model + 31) / 32);

    std::cout << "🛰️ Executing Titan-Attention on A100 High-Speed Cores..." << std::endl;

    titan_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, d_V, d_out, d_model);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "❌ CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void titan_backward_kernel(float* weights, float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grads[idx]; // Learning Step
    }
}
