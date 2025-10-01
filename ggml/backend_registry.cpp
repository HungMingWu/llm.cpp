module;
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#define GGML_USE_CPU

#define GGML_BACKEND_API_VERSION 1

static std::string path_str(const std::filesystem::path& path) {
	std::string u8path;
	try {
#if defined(__cpp_lib_char8_t)
		// C++20 and later: u8string() returns std::u8string
		std::u8string u8str = path.u8string();
		u8path = std::string(reinterpret_cast<const char*>(u8str.c_str()));
#else
		// C++17: u8string() returns std::string
		u8path = path.u8string();
#endif
	}
	catch (...) {
	}
	return u8path;
}

module ggml;
import :log;

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
#ifdef GGML_USE_WEBGPU
	register_backend(ggml_backend_webgpu_reg());
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
	GGML_LOG_DEBUG("{}: registered backend {} ({} devices)",
		__func__, reg->get_name(), reg->get_device_count());
#endif
	backends.push_back({ reg, std::move(handle) });
	for (auto dev : reg->get_devices())
		register_device(dev);
}

void ggml_backend_registry::register_device(ggml_backend_device* device) {
#ifndef NDEBUG
	GGML_LOG_DEBUG("{}: registered device {} ({})", __func__, device->get_name(), device->get_description());
#endif
	devices.push_back(device);
}

// For right now, std::fomrmat doesn't support print filesystem path
ggml_backend_reg_t ggml_backend_registry::load_backend(const std::filesystem::path& path, bool silent)
{
	dl_handle_ptr handle{ dl_load_library(path) };
	if (!handle) {
		if (!silent) {
			GGML_LOG_ERROR("{}: failed to load {}: {}", __func__, path_str(path), dl_error());
		}
		return nullptr;
	}

	auto score_fn = (ggml_backend_score_t)dl_get_sym(handle.get(), "ggml_backend_score");
	if (score_fn && score_fn() == 0) {
		if (!silent) {
			GGML_LOG_INFO("{}: backend {} is not supported on this system", __func__, path_str(path));
		}
		return nullptr;
	}

	auto backend_init_fn = (ggml_backend_init_t)dl_get_sym(handle.get(), "ggml_backend_init");
	if (!backend_init_fn) {
		if (!silent) {
			GGML_LOG_ERROR("{}: failed to find ggml_backend_init in {}", __func__, path_str(path));
		}
		return nullptr;
	}

	ggml_backend_reg_t reg = backend_init_fn();
	if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
		if (!silent) {
			if (!reg) {
				GGML_LOG_ERROR("{}: failed to initialize backend from {}: ggml_backend_init returned NULL", __func__, path_str(path));
			}
			else {
				GGML_LOG_ERROR("{}: failed to initialize backend from {}: incompatible API version (backend: {}, current: {})\n",
					__func__, path_str(path), reg->api_version, GGML_BACKEND_API_VERSION);
			}
		}
		return nullptr;
	}

	GGML_LOG_INFO("{}: loaded {} backend from {}", __func__, reg->get_name(), path_str(path));

	register_backend(reg, std::move(handle));

	return reg;
}

void ggml_backend_registry::unload_backend(ggml_backend_reg_t reg, bool silent) {
	auto it = std::find_if(backends.begin(), backends.end(),
		[reg](const ggml_backend_reg_entry& entry) { return entry.reg == reg; });

	if (it == backends.end()) {
		if (!silent) {
			GGML_LOG_ERROR("{}: backend not found", __func__);
		}
		return;
	}

	if (!silent) {
		GGML_LOG_DEBUG("{}: unloading {} backend\n", __func__, reg->get_name());
	}

	// remove devices
	devices.erase(
		std::remove_if(devices.begin(), devices.end(),
			[reg](ggml_backend_device* dev) { return dev->get_backend_reg() == reg; }),
		devices.end());

	// remove backend
	backends.erase(it);
}
