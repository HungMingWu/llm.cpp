module;
#include <barrier>
#include <condition_variable>
#include <mutex>
#include <new>

export module ggml:cpu.ds;
import :ds;

export
{
    // Threadpool def
    struct ggml_threadpool {
        std::mutex mutex;       // mutex for cond.var
        std::condition_variable  cond;        // cond.var for waiting for new work

        struct ggml_cgraph* cgraph;
        struct ggml_cplan* cplan;

        // synchronization primitives
        std::atomic<int> n_graph;       // incremented when there is work to be done (i.e each graph)
        alignas(std::hardware_destructive_interference_size) std::atomic<int> n_barrier;
        alignas(std::hardware_destructive_interference_size) std::atomic<int> n_barrier_passed;

        // these are atomic as an annotation for thread-sanitizer
        std::atomic<bool> stop;         // Used for stopping the threadpool altogether
        std::atomic<bool> pause;        // Used for pausing the threadpool or individual threads
        std::atomic<bool> abort;        // Used for aborting processing of a graph

        struct ggml_compute_state* workers;   // per thread state
        int          n_threads_max; // number of threads in the pool
        std::atomic<int>   n_threads_cur; // number of threads used in the current graph

        std::barrier<> sync_barrier;

        int32_t      prio;        // Scheduling priority
        uint32_t     poll;        // Polling level (0 - no polling)

        enum ggml_status ec;
    public:
        ggml_threadpool(size_t threads) : sync_barrier(threads) {}
        void barrier() {
            sync_barrier.arrive_and_wait();
        }
    };

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t* work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        ggml_threadpool* threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
    };

    // Per-thread state
    struct ggml_compute_state {
        ggml_cgraph* cgraph;
        ggml_cplan* cplan;
        ggml_threadpool* threadpool;
    };

    struct ggml_compute_params {
        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void* wdata;

        ggml_threadpool* threadpool;
    };
}
