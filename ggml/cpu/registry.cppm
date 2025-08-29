module;
#include <array>
#include <span>
#include <string_view>
#define GGML_ASSERT(...)
#define GGML_BACKEND_API_VERSION 1

export module ggml:cpu.registry;
import :ds;
import :cpu.device;

class backend_cpu_reg : public ggml_backend_reg {
public:
	using ggml_backend_reg::ggml_backend_reg;

	std::string_view get_name() override { return "CPU"; }

	std::span<ggml_backend_dev_t> get_devices() override {
		static ggml_backend_cpu_device cpu_device(this);
		static std::array<ggml_backend_dev_t, 1> backends{ &cpu_device };
		return backends;
	}

	void* get_proc_address(std::string_view name) override {
#if 0
		if (strcmp(name, "ggml_backend_get_features") == 0) {
			return (void*)ggml_backend_cpu_get_features;
		}
		if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
			return (void*)ggml_backend_cpu_set_abort_callback;
		}
		if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
			return (void*)ggml_numa_init;
		}
		if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
			return (void*)ggml_is_numa;
		}

		// threadpool - TODO:  move to ggml-base
		if (strcmp(name, "ggml_threadpool_new") == 0) {
			return (void*)ggml_threadpool_new;
		}
		if (strcmp(name, "ggml_threadpool_free") == 0) {
			return (void*)ggml_threadpool_free;
		}
		if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
			return (void*)ggml_backend_cpu_set_threadpool;
		}

		return NULL;

		GGML_UNUSED(reg);
#else
		return nullptr;
#endif
	}
};

export
{
	ggml_backend_reg_t ggml_backend_cpu_reg() {
		// init CPU feature detection
		//ggml_cpu_init();

		static backend_cpu_reg ggml_backend_cpu_reg = {
			/* .api_version = */ GGML_BACKEND_API_VERSION,
			/* .context     = */ nullptr,
		};

		return &ggml_backend_cpu_reg;
	}
}