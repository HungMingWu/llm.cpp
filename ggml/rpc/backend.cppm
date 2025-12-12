module;
#include <string.h>
#include <string>
#include <vector>

module ggml:rpc.backend;
import :ds;

struct graph_cache {

    bool is_cached(const ggml_cgraph* cgraph) {
        if ((int)last_graph.size() != cgraph->nodes.size()) {
            return false;
        }
        for (int i = 0; i < cgraph->nodes.size(); i++) {
            if (memcmp(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor)) != 0) {
                return false;
            }
        }
        return true;
    }

    void add(const ggml_cgraph* cgraph) {
        last_graph.resize(cgraph->nodes.size());
        for (int i = 0; i < cgraph->nodes.size(); i++) {
            memcpy(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor));
        }
    }

    std::vector<ggml_tensor> last_graph;
};

struct ggml_backend_rpc : public ggml_backend {
	std::string endpoint;
	int device;
	std::string name;
    graph_cache gc;
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
