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

std::u8string backend_filename_prefix()
{
    return u8"libggml-";
}

std::u8string backend_filename_extension()
{
    return u8".so";
}

std::filesystem::path get_executable_path()
{
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t)path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
}