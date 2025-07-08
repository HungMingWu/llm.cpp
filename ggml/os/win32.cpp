module;
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

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

dl_handle dl_load_library(const std::wstring& path)
{
	// suppress error dialogs for missing DLLs
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	HMODULE handle = LoadLibraryW(path.c_str());

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