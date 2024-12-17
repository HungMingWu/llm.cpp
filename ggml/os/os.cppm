module;
#include <stdlib.h>

export module os;

export {
	void get_memory(size_t* free, size_t* total);
}
