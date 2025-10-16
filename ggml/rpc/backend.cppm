module;
#include <string>

module ggml:rpc.backend;
import :ds;

struct ggml_backend_rpc : public ggml_backend {
	std::string endpoint;
	int device;
	std::string name;
protected:
	ggml_status graph_compute_impl(ggml_cgraph* cgraph) override;
public:
	ggml_backend_rpc(ggml_backend_device* device, int deviceID, std::string endpoint, std::string name) : 
		ggml_backend(device), device(deviceID), endpoint(std::move(endpoint)), name(std::move(name))
	{
	}
	const char* get_name() override { return name.c_str(); }
	void synchronize() override;
};