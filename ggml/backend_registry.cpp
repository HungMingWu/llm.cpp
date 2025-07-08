module;
#include <algorithm>
#include <string>
#include <vector>

#define GGML_USE_CPU
#define GGML_LOG_DEBUG(...)
#define GGML_LOG_INFO(...)
#define GGML_LOG_ERROR(...)

#define GGML_BACKEND_API_VERSION 1

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

void ggml_backend_registry::register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle) {
	if (!reg) {
		return;
	}

#ifndef NDEBUG
	GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
		__func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
	backends.push_back({ reg, std::move(handle) });
	for (auto dev : reg->get_devices())
		register_device(dev);
}

void ggml_backend_registry::register_device(ggml_backend_dev_t device) {
#ifndef NDEBUG
	GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
	devices.push_back(device);
}

ggml_backend_reg_t ggml_backend_registry::load_backend(const std::wstring& path, bool silent)
{
	dl_handle_ptr handle{ dl_load_library(path) };
	if (!handle) {
		if (!silent) {
			GGML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(path).c_str());
		}
		return nullptr;
	}

	auto score_fn = (ggml_backend_score_t)dl_get_sym(handle.get(), "ggml_backend_score");
	if (score_fn && score_fn() == 0) {
		if (!silent) {
			GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, utf16_to_utf8(path).c_str());
		}
		return nullptr;
	}

	auto backend_init_fn = (ggml_backend_init_t)dl_get_sym(handle.get(), "ggml_backend_init");
	if (!backend_init_fn) {
		if (!silent) {
			GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, utf16_to_utf8(path).c_str());
		}
		return nullptr;
	}

	ggml_backend_reg_t reg = backend_init_fn();
	if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
		if (!silent) {
			if (!reg) {
				GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n", __func__, utf16_to_utf8(path).c_str());
			}
			else {
				GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
					__func__, utf16_to_utf8(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
			}
		}
		return nullptr;
	}

	GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), utf16_to_utf8(path).c_str());

	register_backend(reg, std::move(handle));

	return reg;
}

void ggml_backend_registry::unload_backend(ggml_backend_reg_t reg, bool silent) {
	auto it = std::find_if(backends.begin(), backends.end(),
		[reg](const ggml_backend_reg_entry& entry) { return entry.reg == reg; });

	if (it == backends.end()) {
		if (!silent) {
			GGML_LOG_ERROR("%s: backend not found\n", __func__);
		}
		return;
	}

	if (!silent) {
		GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
	}

	// remove devices
	devices.erase(
		std::remove_if(devices.begin(), devices.end(),
			[reg](ggml_backend_dev_t dev) { return dev->get_backend_reg() == reg; }),
		devices.end());

	// remove backend
	backends.erase(it);
}
