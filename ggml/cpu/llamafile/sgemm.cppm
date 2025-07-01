module;
#include <stdint.h>

module ggml:cpu.llamafile.sgemm;
import :cpu.ds;

bool llamafile_sgemm(const ggml_compute_params* params, int64_t, int64_t, int64_t,
    const void*, int64_t, const void*, int64_t, void*, int64_t,
    int, int, int);