module;
#include <cstdlib>
#include <memory>
#include <span>

export module ggml:alloc;
import :buffer_type;

export namespace internal
{
    void* aligned_alloc(size_t alignment, size_t size)
    {
#if defined(_MSC_VER) || defined(__MINGW32__)
        return _aligned_malloc(size, alignment);
#else
        return std::aligned_alloc(alignment, size);
#endif
    }
    void free(void* ptr)
    {
#if defined(_MSC_VER) || defined(__MINGW32__)
        return _aligned_free(ptr);
#else
        return std::free(ptr);
#endif
    }
}

ggml_gallocr::ggml_gallocr(std::span<ggml_backend_buffer_type_t> bufts)
    : bufts(bufts.begin(), bufts.end()), buffers(bufts.size()), buf_tallocs(bufts.size()) {
    for (size_t i = 0; i < bufts.size(); i++) {
        // check if the same buffer type is used multiple times and reuse the same allocator
        for (size_t j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                buf_tallocs[i] = buf_tallocs[j];
                break;
            }
        }

        if (buf_tallocs[i] == nullptr) {
            size_t alignment = bufts[i]->get_alignment();
            buf_tallocs[i] = std::make_shared<ggml_dyn_tallocr>(alignment);
        }
    }
}

ggml_dyn_tallocr::ggml_dyn_tallocr(size_t alignment)
    : alignment(alignment)
{
    reset();
}

void ggml_dyn_tallocr::reset()
{
    n_free_blocks = 1;
    free_blocks[0].offset = 0;
    free_blocks[0].size = SIZE_MAX / 2; // restrict maximum size of a measure allocator to half size_t max to avoid overflows
    max_size = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        allocated_tensors[i].tensor = nullptr;
    }
#endif
}
