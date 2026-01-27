5#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include "../include/core1.h"

class TitanDataLoader {
private:
    float* d_pinned_buffer; 
    size_t current_buffer_size;

public:
    TitanDataLoader(size_t size) : current_buffer_size(size) {
        
        cudaError_t err = cudaMallocHost((void**)&d_pinned_buffer, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "❌ Memory Pinning Failed: " << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "💾 Pinned Memory allocated for Ultra-Fast Streaming." << std::endl;
    }

    // ২. Zero-Latency Data Streaming Logic
    void stream_to_vram(const std::string& dataset_path, float* d_gpu_target) {
        int fd = open(dataset_path.c_str(), O_RDONLY | O_DIRECT); 
        if (fd == -1) {
            fd = open(dataset_path.c_str(), O_RDONLY);
        }

        if (fd != -1) {
            ssize_t bytesRead = read(fd, d_pinned_buffer, current_buffer_size * sizeof(float));
            
            if (bytesRead > 0) {
                cudaMemcpyAsync(d_gpu_target, d_pinned_buffer, bytesRead, cudaMemcpyHostToDevice);
                std::cout << "🚀 Streamed " << bytesRead / (1024 * 1024) << " MB directly to A100." << std::endl;
            }
            close(fd);
        } else {
            std::cerr << "❌ Failed to access dataset at: " << dataset_path << std::endl;
        }
    }

    ~TitanDataLoader() {
        cudaFreeHost(d_pinned_buffer);
        std::cout << "🧹 Loader buffer cleared." << std::endl;
    }
};

// --- C++ Interface for Main Controller ---
extern "C" void start_titan_data_stream(const char* file_path, float* d_gpu_ptr, size_t size) {
    TitanDataLoader loader(size);
    loader.stream_to_vram(file_path, d_gpu_ptr);
}
