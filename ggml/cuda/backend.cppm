module;
#include <memory>
#include <string>
#include <vector>
#include "common.h"
#include "cuda_pool.h"

#define GGML_ASSERT(...)
#define GGML_LOG_ERROR(...)
#define GGML_ABORT(...)

export module ggml:cuda.backend;
import :ds;
import :tensor;
import :cuda.buffer;

struct ggml_cuda_graph {
    cudaGraph_t graph = nullptr;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
};

bool ggml_backend_is_cuda(ggml_backend_t backend) {
#if 0
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
#else
    return false;
#endif
}

bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    if (auto buft = dynamic_cast<cuda_backend_buffer*>(buffer))
        return true;
    else 
        return false;
}

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

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

export
{
    class ggml_backend_cuda;
    using quantize_cuda_t = void (*)(
        const float* x, const int32_t* ids, void* vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);
    using ggml_cuda_op_mul_mat_t = void(*)(
        ggml_backend_cuda& ctx,
        ggml_tensor* dst, 
        const char* src0_dd_i, 
        const float* src1_ddf_i,
        const char* src1_ddq_i, 
        float* dst_dd_i, 
        const int64_t row_low, 
        const int64_t row_high, 
        const int64_t src1_ncols,
        const int64_t src1_padded_row_size, 
        cudaStream_t stream);
	class ggml_backend_cuda : public ggml_backend
	{
    protected:
        void set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override;
        void get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override;
    private:
        void evaluate_and_capture_cuda_graph(ggml_cgraph* cgraph,
            std::vector<void*>&, bool&, bool&, bool&);
    public:
        int device;
        std::string name;
        cudaEvent_t copy_event = nullptr;
        cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };

        cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = { nullptr };

		ggml_cuda_graph cuda_graph;

        cudaStream_t stream(int device, int stream) {
            if (streams[device][stream] == nullptr) {
                ggml_cuda_set_device(device);
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
            }
            return streams[device][stream];
        }

        cudaStream_t stream() {
            return stream(device, 0);
        }

        // pool
        std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

        static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device);

        ggml_cuda_pool& pool(int device) {
            if (pools[device] == nullptr) {
                pools[device] = new_pool_for_device(device);
            }
            return *pools[device];
        }

        ggml_cuda_pool& pool() {
            return pool(device);
        }

        cublasHandle_t cublas_handle(int device) {
            if (cublas_handles[device] == nullptr) {
                ggml_cuda_set_device(device);
                CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
                CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
            }
            return cublas_handles[device];
        }

        cublasHandle_t cublas_handle() {
            return cublas_handle(device);
        }

        void op_mul_mat(
            ggml_tensor* dst,
            ggml_cuda_op_mul_mat_t op,
            quantize_cuda_t quantize_src1);
        void mul_mat(ggml_tensor* dst);
        bool compute_forward(ggml_tensor* dst);
    public:
        using ggml_backend::ggml_backend;
        ~ggml_backend_cuda() override;
		const char* get_name() override
		{
			return name.c_str();
		}
        bool cpy_tensor_async(ggml_backend_t backend_src, const ggml_tensor* src, ggml_tensor* dst) override;
        void synchronize() override;
        enum ggml_status graph_compute(ggml_cgraph* cgraph) override;
        void event_record(ggml_backend_event_t event) override;
        void event_wait(ggml_backend_event_t event) override;
	};
}
