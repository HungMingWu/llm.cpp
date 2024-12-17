module;
#include <unistd.h>

module os;

void get_memory(size_t* free, size_t* total)
{
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	*total = pages * page_size;
	*free = *total;
}
