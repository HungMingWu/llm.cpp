module;
#include <barrier>
#include <condition_variable>
#include <mutex>
#include <new>

export module ggml:cpu.ds;
import :ds;

export
{
    struct ggml_compute_params {
        // ith = thread index, nth = number of threads
        int ith, nth;
    };
}
