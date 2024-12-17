module;
#include <vector>
#define GGML_USE_CPU

module ggml;

ggml_backend_registry::ggml_backend_registry() {
#ifdef GGML_USE_CUDA
	register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
	register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
	register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
	register_backend(ggml_backend_vk_reg());
#endif
#ifdef GGML_USE_OPENCL
	register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_CANN
	register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
	register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
	register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_KOMPUTE
	register_backend(ggml_backend_kompute_reg());
#endif
#ifdef GGML_USE_CPU
	register_backend(ggml_backend_cpu_reg());
#endif
}

ggml_backend_registry::~ggml_backend_registry() {
	// FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
	// since backend threads may still be running and accessing resources from the dynamic library
	for (auto& entry : backends) {
		if (entry.handle) {
			entry.handle.release(); // NOLINT
		}
	}
}