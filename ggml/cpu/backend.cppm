module;
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <vector>

export module ggml:cpu.backend;
import :ds;
import :func;
import :cpu.registry;

export
{
	struct ggml_cpu_backend : public ggml_backend {
	protected:
		enum ggml_status graph_compute_impl(ggml_cgraph* cgraph) override;
	public:
		using ggml_backend::ggml_backend;
		size_t n_threads = GGML_DEFAULT_N_THREADS;

		ggml_abort_callback abort_callback = nullptr;

		const char* get_name() override {
			return "CPU";
		}

		void set_n_threads(size_t threads) {
			n_threads = threads;
		}
	};

	std::unique_ptr<ggml_cpu_backend> ggml_backend_cpu_init() {
		return std::make_unique<ggml_cpu_backend>(
			ggml_backend_cpu_reg()->get_device(0)
		);
	}
}
