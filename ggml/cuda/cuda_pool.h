#pragma once
#include <assert.h>
#include <stdint.h>

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void* alloc(size_t size, size_t* actual_size) = 0;
    virtual void free(void* ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool* pool = nullptr;
    T* ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool& pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool& pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T* alloc(size_t size) {
        assert(pool != nullptr);
        assert(ptr == nullptr);
        ptr = (T*)pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T* alloc(ggml_cuda_pool& pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T* get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc&) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc&&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc&) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc&&) = delete;
};
