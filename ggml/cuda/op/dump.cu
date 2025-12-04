#include <stdio.h>
static __global__ void dump(const void* data, size_t length) {
	printf("Dump memory %p, length %llu\n", data, length);
	for (size_t i = 0; i < length; i += 16) {
		for (size_t j = i; j < i + 16 && j < length; j++) {
			printf("%02x ", ((unsigned char*)data)[j]);
		}
		printf("\n");
	}
};

void dump_cuda_memory(const void* data, size_t length)
{
	dump << <1, 1 >> > (data, length);
}