module;
#include <stdint.h>
#include <memory>
#include <string.h>
#include <string>
#include <vector>

module ggml:rpc.backend;
import :ds;
import :rpc.socket;

struct ggml_backend_rpc : public ggml_backend {
	std::shared_ptr<rpc_dispatcher> dispatcher;
	int device;
	std::string name;
protected:
	ggml_status graph_compute_impl(ggml_cgraph* cgraph) override;
	void set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override;
	void get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override;
public:
	ggml_backend_rpc(ggml_backend_device* device, int deviceID, std::shared_ptr<rpc_dispatcher> dispatcher, std::string name) :
		ggml_backend(device), device(deviceID), dispatcher(dispatcher), name(std::move(name))
	{
	}
	const char* get_name() override { return name.c_str(); }
	void synchronize() override;
	void event_record(ggml_backend_event* event) override;
	void event_wait(ggml_backend_event* event) override;
};
