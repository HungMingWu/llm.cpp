module;
#include <memory>
#include <vector>

export module ggml:buffer_type;
import :ds;

export
{
    // Forward declaration
    struct ggml_backend_buffer;

    struct ggml_backend_buffer_type {
    protected:
        // allocate a buffer of this type
        virtual std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) = 0;
    public:
        virtual ~ggml_backend_buffer_type() = default;
        virtual const char* get_name() = 0;

        std::unique_ptr<ggml_backend_buffer> alloc_buffer(size_t size);
        std::unique_ptr<ggml_backend_buffer> alloc_tensors(const ggml_context* ctx);

        // tensor alignment
        virtual size_t get_alignment() = 0;
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        virtual size_t get_max_size() { return SIZE_MAX; }
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        virtual size_t get_alloc_size(const ggml_tensor* tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        virtual bool is_host() { return false; }

        // (optional) check whether support op
        virtual bool supports_op(const ggml_tensor* op) { return false; }
        // (optional) get tensor_traits for  op
        virtual tensor_traits* get_tensor_traits(const struct ggml_tensor* op) { return nullptr; }
    };
}