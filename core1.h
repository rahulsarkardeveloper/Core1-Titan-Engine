#ifndef CORE1_H
#define CORE1_H
 
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


extern "C" void launch_titan_engine(float* d_Q, float* d_K, float* d_V, float* d_out, int d_model);

extern "C" {
    int* titan_encode(const char* input);
    void titan_free_tokens(int* ptr);
}

extern "C" void start_titan_data_stream(const char* file_path, float* d_gpu_ptr, size_t size);

// --- Constants ---
const int CORE1_HIDDEN_DIM = 4096;
const int CORE1_NUM_EXPERTS = 16;

#endif
