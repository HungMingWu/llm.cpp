module;
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "cuda_pool.h"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

module ggml:cuda.backend;
import :ds;
import :tensor;
import :cuda.buffer;
import :cuda.graph;

bool ggml_backend_buffer_is_cuda(ggml_backend_buffer* buffer) {
    if (auto buft = dynamic_cast<cuda_backend_buffer*>(buffer))
        return true;
    else 
        return false;
}

void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= ggml_cuda_peer_max_batch_size_v;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    }
                    else {
                        // reset the error
                        cudaGetLastError();
                    }
                }
                else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    }
                    else {
                        // reset the error
                        cudaGetLastError();
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

struct ggml_cuda_concurrent_event {
    std::vector<cudaEvent_t> join_events;
    cudaEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor*, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor*> original_order;

    const ggml_tensor* join_node;

    ggml_cuda_concurrent_event() = default;

    ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event&) = delete;
    ggml_cuda_concurrent_event& operator=(const ggml_cuda_concurrent_event&) = delete;

    explicit ggml_cuda_concurrent_event(int n_streams) : n_streams(n_streams) {
        join_events.resize(n_streams);

        for (size_t i = 0; i < join_events.size(); ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    }

    ggml_cuda_concurrent_event(ggml_cuda_concurrent_event&& other) noexcept
        : join_events(std::move(other.join_events))
        , fork_event(other.fork_event)
        , n_streams(other.n_streams)
        , stream_mapping(std::move(other.stream_mapping))
        , original_order(std::move(other.original_order))
        , join_node(other.join_node) {
        other.fork_event = nullptr;
    }

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const {
        std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
        write_ranges.resize(n_streams);

        // get join_node's memory range to exclude from overlap checking.
        // multiple nodes can use join_node's buffer; we synchronize on the join node.
        const ggml_tensor* join_t = join_node->view_src ? join_node->view_src : join_node;
        const int64_t       join_start = (int64_t)join_t->data;
        const int64_t       join_end = join_start + join_t->nbytes();

        for (const auto& [tensor, stream] : stream_mapping) {
            const ggml_tensor* t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t)t->data;
            const int64_t       t_end = t_start + t->nbytes();

            // skip tensors that overlap with join_node's buffer.
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // concurrent streams begin from 1
            write_ranges[stream - 1].emplace_back(t_start, t_end);
        }

        for (int i = 0; i < n_streams; ++i) {
            // sorts first by start then by end of write range
            std::ranges::sort(write_ranges[i]);
        }

        bool writes_overlap = false;
        bool dependent_srcs = false;
        for (const auto& [tensor, stream] : stream_mapping) {
            const ggml_tensor* t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t)t->data;
            const int64_t       t_end = t_start + t->nbytes();

            // skip tensors that overlap with join_node's buffer
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // check if this buffer's write data overlaps with another stream's
            std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
            for (int i = 0; i < n_streams; ++i) {
                if (i == stream - 1) {
                    continue;
                }
                auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

                if (it != write_ranges[i].end()) {
                    const std::pair<int64_t, int64_t>& other = *it;

                    // std::lower_bound returns the first element where other >= data_range (lexicographically).
                    // This guarantees other.first >= data_range.first.
                    // Therefore, overlap occurs iff other.first < data_range.second
                    // (i.e., the other range starts before this range ends).
                    if (other.first < data_range.second) {
                        //GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                        writes_overlap = true;
                        break;
                    }
                }
            }

            //check if all srcs are either in branch or don't have a branch
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (!tensor->src[i]) {
                    continue;
                }

                auto it = stream_mapping.find(tensor->src[i]);

                if (it == stream_mapping.end()) {
                    continue;
                }

                if (it->second != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs || writes_overlap) {
                break;
            }
        }

        return !writes_overlap && !dependent_srcs;
    }

    ~ggml_cuda_concurrent_event() {
        if (fork_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(fork_event));
        }
        for (cudaEvent_t e : join_events) {
            if (e != nullptr) {
                CUDA_CHECK(cudaEventDestroy(e));
            }
        }
    }
};

struct ggml_cuda_stream_context {
    std::unordered_map<const ggml_tensor*, ggml_cuda_concurrent_event> concurrent_events;

    void reset() {
        concurrent_events.clear();
    }
};

class ggml_backend_cuda : public ggml_backend
{
protected:
    enum ggml_status graph_compute_impl(ggml_cgraph* cgraph) override;
    void set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override;
    void get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override;
private:
    bool graph_set_enabled(const void* graph_key);
    void graph_evaluate_and_capture(ggml_cgraph* cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void* graph_key);
    cudaEvent_t copy_event = nullptr;
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };

    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = { nullptr };

    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
    // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
    std::unordered_map<const void*, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;

    ggml_cuda_graph* cuda_graph(const void* first_node_ptr) {
        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            cuda_graphs[first_node_ptr] = std::make_unique<ggml_cuda_graph>();
            return cuda_graphs[first_node_ptr].get();
        }
        return it->second.get();
    }

    int curr_stream_no = 0;
    ggml_cuda_stream_context concurrent_stream_context;

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];

    void mul_mat(ggml_tensor* dst);
    bool compute_forward(ggml_tensor* dst);
public:
    int device;
    std::string name;

    ggml_cuda_stream_context& stream_context() { return concurrent_stream_context; }

    cudaStream_t stream(int device, int stream);
    cudaStream_t stream() { return stream(device, curr_stream_no); }

    ggml_cuda_pool& pool(int device);
    ggml_cuda_pool& pool() { return pool(device); }

    cublasHandle_t cublas_handle(int device);
    cublasHandle_t cublas_handle() { return cublas_handle(device); }

public:
    using ggml_backend::ggml_backend;
    ~ggml_backend_cuda() override;
	const char* get_name() override
	{
		return name.c_str();
	}
    bool cpy_tensor_async(ggml_backend* backend_src, const ggml_tensor* src, ggml_tensor* dst) override;
    void synchronize() override;
    void event_record(ggml_backend_event* event) override;
    void event_wait(ggml_backend_event* event) override;
    void graph_optimize(ggml_cgraph* cgraph) override;
};
