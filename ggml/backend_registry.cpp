module;
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

static std::string path_str(const std::filesystem::path& path) {
	try {
#if defined(__cpp_lib_char8_t)
		// C++20 and later: u8string() returns std::u8string
		const std::u8string u8str = path.u8string();
		return std::string(reinterpret_cast<const char*>(u8str.data()), u8str.size());
#else
		// C++17: u8string() returns std::string
		return path.u8string();
#endif
	}
	catch (...) {
		return std::string();
	}
}

module ggml;
import :log;

ggml_backend_registry::ggml_backend_registry() {
#ifdef GGML_USE_METAL
	register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
	register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
	// Add runtime disable check
	if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
		register_backend(ggml_backend_vk_reg());
	}
	else {
		GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
	}
#endif
#ifdef GGML_USE_WEBGPU
	register_backend(ggml_backend_webgpu_reg());
#endif
#ifdef GGML_USE_ZDNN
	register_backend(ggml_backend_zdnn_reg());
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
	register_backend(ggml_backend_virtgpu_reg());
#endif

#ifdef GGML_USE_OPENCL
	register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_ZENDNN
	register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
	register_backend(ggml_backend_hexagon_reg());
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
	for (size_t i = 0; i < reg->get_device_count(); i++) {
		register_device(reg->get_device(i));
	}
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
	if (!reg || reg->api_version != ggml_backend_reg::API_VERSION) {
		if (!silent) {
			if (!reg) {
				GGML_LOG_ERROR("{}: failed to initialize backend from {}: ggml_backend_init returned NULL", __func__, path_str(path));
			}
			else {
				GGML_LOG_ERROR("{}: failed to initialize backend from {}: incompatible API version (backend: {}, current: {})\n",
					__func__, path_str(path), reg->api_version, ggml_backend_reg::API_VERSION);
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

namespace fs = std::filesystem;
static ggml_backend_reg* ggml_backend_load_best(std::u8string name_path, bool silent, std::optional<fs::path> user_search_path) {
	// enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
	const fs::path file_prefix = backend_filename_prefix() + name_path + u8"-";
	const fs::path file_extension = backend_filename_extension();

	std::vector<fs::path> search_paths;
	if (!user_search_path.has_value()) {
#ifdef GGML_BACKEND_DIR
		search_paths.push_back(GGML_BACKEND_DIR);
#endif
		// default search paths: executable directory, current directory
		search_paths.push_back(get_executable_path());
		search_paths.push_back(fs::current_path());
	}
	else {
		search_paths.push_back(user_search_path.value());
	}

	int best_score = 0;
	fs::path best_path;
	std::error_code ec;

	for (const auto& search_path : search_paths) {
		if (!fs::exists(search_path, ec)) {
			if (ec) {
				GGML_LOG_DEBUG("{}: posix_stat({}) failure, error-message: {}\n", __func__, path_str(search_path), ec.message());
			}
			else {
				GGML_LOG_DEBUG("{}: search path {} does not exist\n", __func__, path_str(search_path));
			}
			continue;
		}
		fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
		for (const auto& entry : dir_it) {
			if (entry.is_regular_file(ec)) {
				auto filename = entry.path().filename();
				auto ext = entry.path().extension();
				if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
					dl_handle_ptr handle{ dl_load_library(entry) };
					if (!handle && !silent) {
						GGML_LOG_ERROR("{}: failed to load {}: {}\n", __func__, path_str(entry.path()), dl_error());
					}
					if (handle) {
						auto score_fn = (ggml_backend_score_t)dl_get_sym(handle.get(), "ggml_backend_score");
						if (score_fn) {
							int s = score_fn();
#ifndef NDEBUG
							GGML_LOG_DEBUG("{}: {} score: {}\n", __func__, path_str(entry.path()), s);
#endif
							if (s > best_score) {
								best_score = s;
								best_path = entry.path();
							}
						}
						else {
							if (!silent) {
								GGML_LOG_INFO("{}: failed to find ggml_backend_score in {}\n", __func__, path_str(entry.path()));
							}
						}
					}
				}
			}
		}
	}

	if (best_score == 0) {
		// try to load the base backend
		for (const auto& search_path : search_paths) {
			fs::path filename = backend_filename_prefix() + name_path + backend_filename_extension();
			fs::path path = search_path / filename;
			if (fs::exists(path)) {
				return get_reg().load_backend(path, silent);
			}
		}
		return nullptr;
	}

	return get_reg().load_backend(best_path, silent);
}

// Dynamic loading
ggml_backend_reg* ggml_backend_load(const char* path) {
	return get_reg().load_backend(path, false);
}

void ggml_backend_load_all_from_path(std::optional<fs::path> dir_path = {}) {
#ifdef NDEBUG
	bool silent = true;
#else
	bool silent = false;
#endif

	ggml_backend_load_best(u8"blas", silent, dir_path);
	ggml_backend_load_best(u8"zendnn", silent, dir_path);
	ggml_backend_load_best(u8"cann", silent, dir_path);
	ggml_backend_load_best(u8"cuda", silent, dir_path);
	ggml_backend_load_best(u8"hip", silent, dir_path);
	ggml_backend_load_best(u8"metal", silent, dir_path);
	ggml_backend_load_best(u8"rpc", silent, dir_path);
	ggml_backend_load_best(u8"sycl", silent, dir_path);
	ggml_backend_load_best(u8"vulkan", silent, dir_path);
	ggml_backend_load_best(u8"opencl", silent, dir_path);
	ggml_backend_load_best(u8"hexagon", silent, dir_path);
	ggml_backend_load_best(u8"musa", silent, dir_path);
	ggml_backend_load_best(u8"cpu", silent, dir_path);
	// check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
	const char* backend_path = std::getenv("GGML_BACKEND_PATH");
	if (backend_path) {
		ggml_backend_load(backend_path);
	}
}

void ggml_backend_load_all()
{
	ggml_backend_load_all_from_path();
}

std::unique_ptr<ggml_backend> ggml_backend_init_best()
{
	ggml_backend_device* dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
	dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
	if (!dev) {
		return nullptr;
	}
	return dev->init_backend(nullptr);
}