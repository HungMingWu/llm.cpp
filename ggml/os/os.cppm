module;
#include <stdlib.h>
#include <filesystem>
#include <memory>
#include <string>

export module ggml:os;

export 
{
	using dl_handle = void*;
	dl_handle dl_load_library(const std::filesystem::path& path);
	void dl_unload_library(dl_handle handle);
	void* dl_get_sym(dl_handle handle, const char* name);
	const char* dl_error();

	struct dl_handle_deleter {
		void operator()(dl_handle handle) {
			dl_unload_library(handle);
		}
	};

	using dl_handle_ptr = std::unique_ptr<std::remove_pointer_t<dl_handle>, dl_handle_deleter>;

	void get_memory(size_t* free, size_t* total);
}