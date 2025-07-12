module;
#include <stdint.h>
#include <memory>

module ggml:rpc.buffer;
import :buffer;
import :ds;
import :rpc.socket;

struct rpc_backend_buffer : public ggml_backend_buffer {
    std::shared_ptr<socket_t> sock;
    void* base_ptr;
    uint64_t remote_ptr;
protected:
    void* get_base_impl() override;
    void clear_impl(uint8_t value) override;
public:
    rpc_backend_buffer(ggml_backend_buffer_type_t type, std::shared_ptr<socket_t> sock, void* base_ptr, uint64_t remote_ptr, size_t size)
        : ggml_backend_buffer(type, size), sock(std::move(sock)), base_ptr(base_ptr), remote_ptr(remote_ptr) {
    }
    ~rpc_backend_buffer() override;
    ggml_status init_tensor(ggml_tensor* tensor) override;
    void set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override;
    void get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override;
    bool cpy_tensor(const ggml_tensor* src, ggml_tensor* dst) override;
};