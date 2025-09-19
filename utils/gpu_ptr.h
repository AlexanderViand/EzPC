// RAII wrapper for GPU memory management
// Provides automatic cleanup and move semantics for CUDA allocations

#pragma once

#include "gpu_mem.h"
#include <utility>
#include <cassert>

// Minimal RAII wrapper for GPU memory
// Maintains compatibility with existing code through implicit conversion
template<typename T>
class gpu_ptr {
private:
    T* ptr = nullptr;
    size_t num_elements = 0;

public:
    // Default constructor
    gpu_ptr() = default;

    // Allocate n elements of type T
    explicit gpu_ptr(size_t n) : num_elements(n) {
        if (n > 0) {
            ptr = reinterpret_cast<T*>(gpuMalloc(n * sizeof(T)));
        }
    }

    // Destructor - automatic cleanup
    ~gpu_ptr() {
        if (ptr) {
            gpuFree(ptr);
        }
    }

    // Move constructor
    gpu_ptr(gpu_ptr&& other) noexcept
        : ptr(other.ptr), num_elements(other.num_elements) {
        other.ptr = nullptr;
        other.num_elements = 0;
    }

    // Move assignment
    gpu_ptr& operator=(gpu_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                gpuFree(ptr);
            }
            ptr = other.ptr;
            num_elements = other.num_elements;
            other.ptr = nullptr;
            other.num_elements = 0;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment
    gpu_ptr(const gpu_ptr&) = delete;
    gpu_ptr& operator=(const gpu_ptr&) = delete;

    // Access raw pointer
    T* get() const { return ptr; }

    // Release ownership (caller responsible for deallocation)
    T* release() {
        T* tmp = ptr;
        ptr = nullptr;
        num_elements = 0;
        return tmp;
    }

    // Reset with new allocation
    void reset(size_t n = 0) {
        if (ptr) {
            gpuFree(ptr);
            ptr = nullptr;
        }
        num_elements = n;
        if (n > 0) {
            ptr = reinterpret_cast<T*>(gpuMalloc(n * sizeof(T)));
        }
    }

    // Swap with another gpu_ptr
    void swap(gpu_ptr& other) noexcept {
        std::swap(ptr, other.ptr);
        std::swap(num_elements, other.num_elements);
    }

    // Get number of elements
    size_t size() const { return num_elements; }

    // Get size in bytes
    size_t size_bytes() const { return num_elements * sizeof(T); }

    // Check if pointer is valid
    explicit operator bool() const { return ptr != nullptr; }

    // Implicit conversion to T* for backward compatibility
    operator T*() const { return ptr; }

    // Array subscript operator
    T& operator[](size_t idx) {
        assert(idx < num_elements);
        return ptr[idx];
    }

    const T& operator[](size_t idx) const {
        assert(idx < num_elements);
        return ptr[idx];
    }
};

// Helper function similar to std::make_unique
template<typename T>
gpu_ptr<T> make_gpu_ptr(size_t n) {
    return gpu_ptr<T>(n);
}

// Secure version using secure allocation functions
template<typename T>
class gpu_ptr_secure {
private:
    T* ptr = nullptr;
    size_t num_elements = 0;

public:
    gpu_ptr_secure() = default;

    explicit gpu_ptr_secure(size_t n) : num_elements(n) {
        if (n > 0) {
            ptr = reinterpret_cast<T*>(gpuMallocSecure(n * sizeof(T)));
        }
    }

    ~gpu_ptr_secure() {
        if (ptr) {
            gpuFreeSecure(ptr, num_elements * sizeof(T));
        }
    }

    // Move constructor
    gpu_ptr_secure(gpu_ptr_secure&& other) noexcept
        : ptr(other.ptr), num_elements(other.num_elements) {
        other.ptr = nullptr;
        other.num_elements = 0;
    }

    // Move assignment
    gpu_ptr_secure& operator=(gpu_ptr_secure&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                gpuFreeSecure(ptr, num_elements * sizeof(T));
            }
            ptr = other.ptr;
            num_elements = other.num_elements;
            other.ptr = nullptr;
            other.num_elements = 0;
        }
        return *this;
    }

    // Delete copy operations
    gpu_ptr_secure(const gpu_ptr_secure&) = delete;
    gpu_ptr_secure& operator=(const gpu_ptr_secure&) = delete;

    // Same interface as gpu_ptr
    T* get() const { return ptr; }
    T* release() {
        T* tmp = ptr;
        ptr = nullptr;
        num_elements = 0;
        return tmp;
    }

    size_t size() const { return num_elements; }
    size_t size_bytes() const { return num_elements * sizeof(T); }
    explicit operator bool() const { return ptr != nullptr; }
    operator T*() const { return ptr; }
};

// Specialization for void* to handle generic byte allocations
template<>
class gpu_ptr<void> {
private:
    void* ptr = nullptr;
    size_t size_bytes_val = 0;

public:
    gpu_ptr() = default;

    explicit gpu_ptr(size_t bytes) : size_bytes_val(bytes) {
        if (bytes > 0) {
            ptr = gpuMalloc(bytes);
        }
    }

    ~gpu_ptr() {
        if (ptr) {
            gpuFree(ptr);
        }
    }

    gpu_ptr(gpu_ptr&& other) noexcept
        : ptr(other.ptr), size_bytes_val(other.size_bytes_val) {
        other.ptr = nullptr;
        other.size_bytes_val = 0;
    }

    gpu_ptr& operator=(gpu_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                gpuFree(ptr);
            }
            ptr = other.ptr;
            size_bytes_val = other.size_bytes_val;
            other.ptr = nullptr;
            other.size_bytes_val = 0;
        }
        return *this;
    }

    gpu_ptr(const gpu_ptr&) = delete;
    gpu_ptr& operator=(const gpu_ptr&) = delete;

    void* get() const { return ptr; }
    void* release() {
        void* tmp = ptr;
        ptr = nullptr;
        size_bytes_val = 0;
        return tmp;
    }

    size_t size_bytes() const { return size_bytes_val; }
    explicit operator bool() const { return ptr != nullptr; }
    operator void*() const { return ptr; }
    operator uint8_t*() const { return reinterpret_cast<uint8_t*>(ptr); }
};

// Type alias for common use cases
using gpu_ptr_u8 = gpu_ptr<uint8_t>;
using gpu_ptr_u32 = gpu_ptr<uint32_t>;
using gpu_ptr_u64 = gpu_ptr<uint64_t>;