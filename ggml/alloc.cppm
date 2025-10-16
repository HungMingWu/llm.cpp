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

ggml_gallocr::ggml_gallocr(std::span<ggml_backend_buffer_type*> bufts)
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

size_t ggml_dyn_tallocr::max_size(int chunk) const {
    return chunk < chunks.size() ? chunks[chunk].max_size : 0;
}

void ggml_dyn_tallocr::reset()
{
    chunks.clear();

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}
