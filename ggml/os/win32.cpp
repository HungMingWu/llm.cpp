module;
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

#include <filesystem>
#include <string>

module ggml;

void get_memory(size_t* free, size_t* total)
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	*total = status.ullTotalPhys;
	*free = status.ullAvailPhys;
}

dl_handle dl_load_library(const std::filesystem::path& path)
{
	// suppress error dialogs for missing DLLs
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	HMODULE handle = LoadLibraryW(path.wstring().c_str());

	SetErrorMode(old_mode);

	return handle;
}

void dl_unload_library(dl_handle handle)
{
	FreeLibrary(reinterpret_cast<HMODULE>(handle));
}

void* dl_get_sym(dl_handle handle, const char* name)
{
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	void* p = (void*)GetProcAddress(reinterpret_cast<HMODULE>(handle), name);

	SetErrorMode(old_mode);

	return p;
}

const char* dl_error()
{
	return "";
}

std::u8string backend_filename_prefix()
{
	return u8"ggml-";
}

std::u8string backend_filename_extension()
{
	return u8".dll";
}

std::filesystem::path get_executable_path()
{
	std::vector<wchar_t> path(MAX_PATH);
	DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
	if (len == 0) {
		return {};
	}
	std::wstring base_path(path.data(), len);
	// remove executable name
	auto last_slash = base_path.find_last_of('\\');
	if (last_slash != std::string::npos) {
		base_path = base_path.substr(0, last_slash);
	}
	return base_path + L"\\";
}