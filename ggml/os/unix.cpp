module;
#include <dlfcn.h>
#include <filesystem>
#include <string>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

module ggml;

void get_memory(size_t* free, size_t* total)
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
}

dl_handle dl_load_library(const std::filesystem::path& path)
{
	return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
}

void dl_unload_library(dl_handle handle)
{
	dlclose(handle);
}

void* dl_get_sym(dl_handle handle, const char* name)
{
	return dlsym(handle, name);
}

const char* dl_error()
{
    const char* rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}