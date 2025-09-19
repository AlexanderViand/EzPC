// Example demonstrating gpu_ptr usage and migration path

#include "utils/gpu_ptr.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include <iostream>

// Example kernel
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void example_old_style() {
    std::cout << "=== Old style (manual management) ===" << std::endl;

    const int N = 1000;

    // Manual allocation
    float* d_a = (float*)gpuMalloc(N * sizeof(float));
    float* d_b = (float*)gpuMalloc(N * sizeof(float));
    float* d_c = (float*)gpuMalloc(N * sizeof(float));

    // Use the pointers
    addKernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    checkCudaErrors(cudaDeviceSynchronize());

    // Manual cleanup - easy to forget!
    gpuFree(d_a);
    gpuFree(d_b);
    gpuFree(d_c);
}

void example_new_style() {
    std::cout << "=== New style (RAII) ===" << std::endl;

    const int N = 1000;

    // Automatic allocation
    gpu_ptr<float> d_a(N);
    gpu_ptr<float> d_b(N);
    gpu_ptr<float> d_c(N);

    // Works exactly the same with kernels (implicit conversion)
    addKernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    checkCudaErrors(cudaDeviceSynchronize());

    // No manual cleanup needed!
}

void example_migration_path() {
    std::cout << "=== Migration example ===" << std::endl;

    const int N = 1000;

    // Can mix old and new style during migration
    gpu_ptr<float> d_a(N);  // New style
    float* d_b = (float*)gpuMalloc(N * sizeof(float));  // Old style
    gpu_ptr<float> d_c(N);  // New style

    // All work together
    addKernel<<<(N + 255) / 256, 256>>>(d_a.get(), d_b, d_c, N);
    checkCudaErrors(cudaDeviceSynchronize());

    // Only need to manually free the old-style allocation
    gpuFree(d_b);
    // d_a and d_c are automatically freed
}

void example_move_semantics() {
    std::cout << "=== Move semantics ===" << std::endl;

    const int N = 1000;

    // Create a gpu_ptr
    gpu_ptr<float> d_a(N);

    // Move it to another variable (no copy, just transfer ownership)
    gpu_ptr<float> d_b = std::move(d_a);
    // Now d_a is empty, d_b owns the memory

    // Can also use in containers
    std::vector<gpu_ptr<float>> buffers;
    buffers.push_back(gpu_ptr<float>(N));  // Moved into vector
    buffers.emplace_back(N);  // Constructed in-place
}

void example_release_pattern() {
    std::cout << "=== Release pattern for legacy APIs ===" << std::endl;

    const int N = 1000;

    // Start with RAII
    gpu_ptr<float> d_a(N);

    // Need to pass to legacy function that takes ownership?
    float* raw_ptr = d_a.release();  // Release ownership

    // Now responsible for manual cleanup
    gpuFree(raw_ptr);
}

// Example of refactoring a typical function
template<typename T>
T* old_function(int N) {
    T* d_output = (T*)gpuMalloc(N * sizeof(T));
    // ... do computation ...
    return d_output;  // Caller must remember to free!
}

template<typename T>
gpu_ptr<T> new_function(int N) {
    gpu_ptr<T> d_output(N);
    // ... do computation ...
    return d_output;  // Automatically moved, no leak possible
}

int main() {
    // Initialize GPU memory pool
    initGPUMemPool();

    example_old_style();
    example_new_style();
    example_migration_path();
    example_move_semantics();
    example_release_pattern();

    std::cout << "All examples completed successfully!" << std::endl;
    return 0;
}