module;
#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

export module ggml:buffer;
import :buffer_type;
import :ds;

export
{
    struct ggml_backend_buffer {
    protected:
        ggml_backend_buffer_type* buft;
        size_t size;
        ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_ANY;
        // base address of the buffer
        virtual void* get_base_impl() { return nullptr; }
        // clear the entire buffer
        virtual void clear_impl(uint8_t value) = 0;
    public:
        ggml_backend_buffer(ggml_backend_buffer_type* buft,
            size_t size) : buft(buft), size(size) {
        }
        virtual ~ggml_backend_buffer() = default;
        void* get_base();
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        virtual ggml_status init_tensor(ggml_tensor* tensor) {
            return GGML_STATUS_SUCCESS;
        }
        // tensor data access
        virtual void memset_tensor(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {};
        virtual void set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {};
        virtual void get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {};
        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        virtual bool cpy_tensor(const ggml_tensor* src, ggml_tensor* dst) { return false; }
        void clear(uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        virtual void reset() {};
        size_t get_size() const { return size; }
        void setUsage(ggml_backend_buffer_usage usage) { this->usage = usage; }
        ggml_backend_buffer_usage getUsage() const { return usage; }
        // helper function
        constexpr ggml_backend_buffer_type* get_type() const { return buft; }
        constexpr bool is_host() const { return buft->is_host(); }
        constexpr size_t get_alignment() const { return buft->get_alignment(); }
        constexpr size_t get_alloc_size(const ggml_tensor* tensor) {
            return buft->get_alloc_size(tensor);
        }
        ggml_status alloc(ggml_tensor* tensor, void* addr);
    };

    struct multi_backend_buffer : public ggml_backend_buffer {
        std::vector<std::unique_ptr<ggml_backend_buffer>> buffers;
    protected:
        void clear_impl(uint8_t value) override;
    public:
        multi_backend_buffer(
            ggml_backend_buffer_type* buft, size_t size, std::vector<std::unique_ptr<ggml_backend_buffer>> buffers);
    };
}
