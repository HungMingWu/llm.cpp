module;
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

module os;

void get_memory(size_t* free, size_t* total)
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	*total = status.ullTotalPhys;
	*free = status.ullAvailPhys;
}