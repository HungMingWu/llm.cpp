module;
#include <assert.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include "block.h"
#include "common.h"
#include "cuda_pool.h"
#include "op/convert.cuh"
#include "op/cuda_func.h"
#include "cuda_config.h"

#define GGML_ABORT(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :cuda.backend;
import :cuda.fused;
import :cuda.op;

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

namespace 
{
    // pool with virtual memory
    struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
        static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

        int device;
        CUdeviceptr pool_addr = 0;
        size_t pool_used = 0;
        size_t pool_size = 0;
        size_t granularity;

        explicit ggml_cuda_pool_vmm(int device) :
            device(device),
            granularity(ggml_cuda_info().devices[device].vmm_granularity) {
        }

        ~ggml_cuda_pool_vmm() {
            if (pool_addr != 0) {
                CU_CHECK(cuMemUnmap(pool_addr, pool_size));
                CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
            }
        }

        void* alloc(size_t size, size_t* actual_size) override {
            // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
            const size_t alignment = 128;
            size = alignment * ((size + alignment - 1) / alignment);

            size_t avail = pool_size - pool_used;

            if (size > avail) {
                // round up to the next multiple of the granularity
                size_t reserve_size = size - avail;
                reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

                GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

                // allocate more physical memory
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = device;
                CUmemGenericAllocationHandle handle;
                CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

                // reserve virtual address space (if not already reserved)
                if (pool_addr == 0) {
                    CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
                }

                // map at the end of the pool
                CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

                // the memory allocation handle is no longer needed after mapping
                CU_CHECK(cuMemRelease(handle));

                // set access
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

                // add to the pool
                pool_size += reserve_size;

                //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                //       device, (unsigned long long) (pool_size/1024/1024),
                //       (unsigned long long) (reserve_size/1024/1024));
            }

            GGML_ASSERT(pool_addr != 0);

            void* ptr = (void*)(pool_addr + pool_used);
            *actual_size = size;
            pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
            printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

            return ptr;
        }

        void free(void* ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
            printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

            pool_used -= size;

            // all deallocations must be in reverse order of the allocations
            GGML_ASSERT(ptr == (void*)(pool_addr + pool_used));
        }
    };

    // buffer pool for cuda (legacy)
    struct ggml_cuda_pool_leg : public ggml_cuda_pool {
        static const int MAX_BUFFERS = 256;

        int device;
        struct ggml_cuda_buffer {
            void* ptr = nullptr;
            size_t size = 0;
        };

        ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
        size_t pool_size = 0;

        explicit ggml_cuda_pool_leg(int device) :
            device(device) {
        }

        ~ggml_cuda_pool_leg() {
            ggml_cuda_set_device(device);
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr != nullptr) {
                    CUDA_CHECK(cudaFree(b.ptr));
                    pool_size -= b.size;
                }
            }
            GGML_ASSERT(pool_size == 0);
        }

        void* alloc(size_t size, size_t* actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
            int nnz = 0;
            size_t max_size = 0;
#endif
            size_t best_diff = 1ull << 36;
            int ibest = -1;
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                    ++nnz;
                    if (b.size > max_size) max_size = b.size;
#endif
                    if (b.size >= size) {
                        size_t diff = b.size - size;
                        if (diff < best_diff) {
                            best_diff = diff;
                            ibest = i;
                            if (!best_diff) {
                                void* ptr = b.ptr;
                                *actual_size = b.size;
                                b.ptr = nullptr;
                                b.size = 0;
                                return ptr;
                            }
                        }
                    }
                }
            }
            if (ibest >= 0) {
                ggml_cuda_buffer& b = buffer_pool[ibest];
                void* ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                return ptr;
            }
            void* ptr;
            size_t look_ahead_size = (size_t)(1.05 * size);
            look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
            ggml_cuda_set_device(device);
            CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
            *actual_size = look_ahead_size;
            pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
            GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
            return ptr;
        }

        void free(void* ptr, size_t size) override {
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr == nullptr) {
                    b.ptr = ptr;
                    b.size = size;
                    return;
                }
            }
            GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS");
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaFree(ptr));
            pool_size -= size;
        }
    };

    std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device, int /*stream_no*/) {
        if constexpr (ggml_use_vmm_v) {
            if (ggml_cuda_info().devices[device].vmm) {
                return std::make_unique<ggml_cuda_pool_vmm>(device);
            }
        }
        return std::make_unique<ggml_cuda_pool_leg>(device);
    }

    cudaError_t ggml_cuda_Memcpy2DPeerAsync(
        void* dst, int dstDevice, size_t dpitch, void* src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

        if constexpr (!ggml_use_hip_v && !ggml_use_musa_v) {
            // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
            cudaMemcpy3DPeerParms p = {};
            p.dstDevice = dstDevice;
            p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
            p.srcDevice = srcDevice;
            p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
            p.extent = make_cudaExtent(width, height, 1);
            return cudaMemcpy3DPeerAsync(&p, stream);
        }
        else {
            // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
            return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaError_t ggml_cuda_cpy_tensor_2d(
        void* dst, const ggml_tensor* src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

        const char* src_ptr = (const char*)src->data;
        char* dst_ptr = (char*)dst;

        const int64_t ne0 = src->ne[0];
        const int64_t nb0 = src->nb[0];
        const int64_t nb1 = src->nb[1];
        const int64_t nb2 = src->nb[2];
        const int64_t nb3 = src->nb[3];
        const enum ggml_type type = src->type;
        const int64_t ts = ggml_type_size(type);
        const int64_t bs = ggml_blck_size(type);
        const int64_t i1_diff = i1_high - i1_low;

        const char* x = src_ptr + i1_low * nb1 + i2 * nb2 + i3 * nb3;
        if (nb0 == ts && nb1 == ts * ne0 / bs) {
            return cudaMemcpyAsync(dst_ptr, x, i1_diff * nb1, cudaMemcpyDeviceToDevice, stream);
        }
        else if (nb0 == ts) {
            return cudaMemcpy2DAsync(dst_ptr, ts * ne0 / bs, x, nb1, ts * ne0 / bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
        }
        else {
            for (int64_t i1 = 0; i1 < i1_diff; i1++) {
                const void* rx = (const void*)((const char*)x + i1 * nb1);
                void* rd = (void*)(dst_ptr + i1 * ts * ne0 / bs);
                // pretend the row is a matrix with cols=1
                cudaError_t r = cudaMemcpy2DAsync(rd, ts / bs, rx, nb0, ts / bs, ne0, cudaMemcpyDeviceToDevice, stream);
                if (r != cudaSuccess) {
                    return r;
                }
            }
            return cudaSuccess;
        }
    }

    using quantize_cuda_t = void (*)(
        const float* x, const int32_t* ids, void* vy,
        internal::ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
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

    void op_mul_mat(
        ggml_backend_cuda& ctx,
        ggml_tensor* dst,
        ggml_cuda_op_mul_mat_t op,
        quantize_cuda_t quantize_src1)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const int64_t ne00 = src0->ne[0];
        const int64_t ne01 = src0->ne[1];
        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];

        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2];
        const int64_t ne13 = src1->ne[3];
        const int64_t nrows1 = ggml_nrows(src1);

        const int64_t ne0 = dst->ne[0];
        const int64_t ne1 = dst->ne[1];

        const int64_t nb2 = dst->nb[2];
        const int64_t nb3 = dst->nb[3];

        GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
        auto src1_ctx = (cuda_backend_buffer*)src1->buffer;
        auto dst_ctx = (cuda_backend_buffer*)dst->buffer;

        GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

        GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

        const int64_t i02_divisor = ne12 / ne02;
        const int64_t i03_divisor = ne13 / ne03;

        const size_t src0_ts = ggml_type_size(src0->type);
        const size_t src0_bs = ggml_blck_size(src0->type);
        const size_t q8_1_ts = sizeof(block_q8_1);
        const size_t q8_1_bs = QK8_1;

        const bool src0_is_contiguous = ggml_is_contiguous(src0);
        const bool src1_is_contiguous = ggml_is_contiguous(src1);

        const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

        const bool split = to_split_buffer_type(src0->buffer->get_type()) != nullptr;
        GGML_ASSERT(!(split && ne02 > 1));
        GGML_ASSERT(!(split && ne03 > 1));
        GGML_ASSERT(!(split && ne02 < ne12));

        ggml_tensor_extra_gpu* src0_extra = split ? (ggml_tensor_extra_gpu*)src0->extra : nullptr;

        std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
        if (split) {
            auto buft_ctx = (cuda_split_backend_buffer_type*)src0->buffer->get_type();
            tensor_split = buft_ctx->tensor_split;
        }

        struct dev_data {
            int cc;

            ggml_cuda_pool_alloc<char>   src0_dd_alloc;
            ggml_cuda_pool_alloc<float> src1_ddf_alloc;
            ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
            ggml_cuda_pool_alloc<float>   dst_dd_alloc;

            char* src0_dd = nullptr;
            float* src1_ddf = nullptr; // float
            char* src1_ddq = nullptr; // q8_1
            float* dst_dd = nullptr;

            int64_t  row_low;
            int64_t row_high;
        };

        dev_data dev[GGML_CUDA_MAX_DEVICES];

        int used_devices = 0;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            dev[id].cc = ggml_cuda_info().devices[id].cc;

            // by default, use all rows
            dev[id].row_low = 0;
            dev[id].row_high = ne01;

            // for multi GPU, get the row boundaries from tensor split
            // and round to mul_mat_q tile sizes
            if (split) {
                const int64_t rounding = get_row_rounding(tensor_split);

                if (id != 0) {
                    dev[id].row_low = ne01 * tensor_split[id];
                    if (dev[id].row_low < ne01) {
                        dev[id].row_low -= dev[id].row_low % rounding;
                    }
                }

                if (id != ggml_backend_cuda_get_device_count() - 1) {
                    dev[id].row_high = ne01 * tensor_split[id + 1];
                    if (dev[id].row_high < ne01) {
                        dev[id].row_high -= dev[id].row_high % rounding;
                    }
                }
            }
        }

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            used_devices++;

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, 0);

            if (src0_is_contiguous) {
                dev[id].src0_dd = split ? (char*)src0_extra->data_device[id] : (char*)src0->data;
            }
            else {
                // If src0 is not contiguous it will be copied to a temporary buffer.
                // This buffer needs to be cleared entirely because multiple regions will function as padding.
                const size_t nbytes_data = src0->nbytes();
                const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
                dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
                // TODO: remove this for MUSA once the Guilty Lockup issue is resolved
#ifndef GGML_USE_MUSA
                CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
#else // GGML_USE_MUSA
                CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
#endif // !GGML_USE_MUSA
            }

            // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
            if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
                GGML_ASSERT(ggml_is_contiguously_allocated(src0));
                GGML_ASSERT(!src0->view_src);
                const size_t nbytes_data = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low) * ne00);
                const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
                CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
            }

            if (src1_on_device && src1_is_contiguous) {
                dev[id].src1_ddf = (float*)src1->data;
            }
            else {
                dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), src1->nelements());
            }

            if (quantize_src1) {
                size_t src_1_ddq_size = nrows1 * src1_padded_col_size * q8_1_ts / q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src_1_ddq_size += get_mmq_x_max_host(dev[id].cc) * sizeof(block_q8_1_mmq);
                }
                dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);
                if (src1_on_device && src1_is_contiguous) {
                    quantize_src1(
                        dev[id].src1_ddf, nullptr, dev[id].src1_ddq, std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0],
                        src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                        src1_padded_col_size, src1->ne[1], src1->ne[2], src1->ne[3], stream);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            if (dst_on_device) {
                dev[id].dst_dd = (float*)dst->data;
            }
            else {
                const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low) * ne1 : dst->nelements();
                dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
            }

            // if multiple devices are used they need to wait for the main device
            // here an event is recorded that signals that the main device has finished calculating the input data
            if (split && used_devices > 1) {
                ggml_cuda_set_device(ctx.device);
                CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
            }
        }

        const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
        for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
            const int64_t is = split ? (src1_col_0 / src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
            const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

            for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
                if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                    continue;
                }

                const bool src1_on_device = id == src1_ctx->device;
                const bool  dst_on_device = id == dst_ctx->device;
                const int64_t row_diff = dev[id].row_high - dev[id].row_low;

                ggml_cuda_set_device(id);
                cudaStream_t stream = ctx.stream(id, is);

                // wait for main GPU data if necessary
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
                }

                for (int64_t i0 = 0; i0 < ne13 * ne12; ++i0) {
                    const int64_t i03 = i0 / ne12;
                    const int64_t i02 = i0 % ne12;

                    size_t src1_ddq_i_offset = i0 * ne11 * src1_padded_col_size * q8_1_ts / q8_1_bs;

                    if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                        src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                    }
                    else {
                        src1_ddq_i_offset += src1_col_0 * src1_padded_col_size * q8_1_ts / q8_1_bs;
                    }

                    // for split tensors the data begins at i0 == i0_offset_low
                    const size_t nbytes_src0_matrix = ne01 * ne00 * src0_ts / src0_bs;
                    char* src0_dd_i = dev[id].src0_dd + ((i03 / i03_divisor) * ne02 + (i02 / i02_divisor)) * nbytes_src0_matrix;
                    float* src1_ddf_i = dev[id].src1_ddf + (i0 * ne11 + src1_col_0) * ne10;
                    char* src1_ddq_i = dev[id].src1_ddq + src1_ddq_i_offset;
                    float* dst_dd_i = dev[id].dst_dd + (i0 * ne1 + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                    // the main device memory buffer can be on VRAM scratch, with space for all partial results
                    // in that case an offset on dst_ddf_i is needed
                    if (id == ctx.device) {
                        dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                    }

                    // copy src0, src1 to device if necessary
                    if (src1_is_contiguous) {
                        if (id != ctx.device) {
                            if (quantize_src1) {
                                char* src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                    const size_t pitch = ne11 * sizeof(block_q8_1_mmq);
                                    const size_t width = src1_ncols * sizeof(block_q8_1_mmq);
                                    const size_t height = src1_padded_col_size / (4 * QK8_1);
                                    CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                                }
                                else {
                                    CUDA_CHECK(cudaMemcpyPeerAsync(
                                        src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs, stream));
                                }
                            }
                            else {
                                float* src1_ddf_i_source = (float*)src1->data;
                                src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;
                                CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                    src1_ncols * ne10 * sizeof(float), stream));
                            }
                        }
                    }
                    else if (src1_on_device && !src1_is_contiguous) {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                            src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0 + src1_ncols, stream));
                    }
                    else {
                        GGML_ABORT("fatal error");
                    }

                    if (quantize_src1 && !src1_is_contiguous) {
                        quantize_src1(
                            src1_ddf_i, nullptr, src1_ddq_i, std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], src1->ne[0],
                            src1->ne[1] * src1->ne[0], src1->ne[2] * src1->ne[1] * src1->ne[0],
                            src1_padded_col_size, src1_ncols, 1, 1, stream);
                        CUDA_CHECK(cudaGetLastError());
                    }

                    if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02 / i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                    }

                    // do the computation
                    op(ctx, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                        dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());

                    // copy dst to host or other device if necessary
                    if (!dst_on_device) {
                        void* dst_off_device = dst->data;
                        if (split) {
                            // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                            // dst is NOT transposed.
                            // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                            // Instead they need to be copied to the correct slice in ne0 = dst row index.
                            // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                            float* dhf_dst_i = (float*)((char*)dst_off_device + i02 * nb2 + i03 * nb3);
                            GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                            dhf_dst_i += src1_col_0 * ne0 + dev[id].row_low;
                            CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                                dhf_dst_i, ctx.device, ne0 * sizeof(float), dst_dd_i, id, row_diff * sizeof(float), row_diff * sizeof(float), src1_ncols, stream));
                        }
                        else {
                            float* dhf_dst_i = (float*)((char*)dst_off_device + i02 * nb2 + i03 * nb3);
                            GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                            dhf_dst_i += src1_col_0 * ne0;
                            CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols * ne0 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                        }
                    }

                    // add event for the main device to wait on until other device is done
                    if (split && (id != ctx.device || is != 0)) {
                        CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                    }

                }
            }
        }

        // main device waits for all other devices to be finished
        if (split && ggml_backend_cuda_get_device_count() > 1) {
            int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
            is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

            ggml_cuda_set_device(ctx.device);
            for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
                if (dev[id].row_low == dev[id].row_high) {
                    continue;
                }
                for (int64_t is = 0; is < is_max; ++is) {
                    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
                }
            }
        }
    }
}

static void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* /*src1_ddf_i*/,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t /*src1_padded_row_size*/, cudaStream_t stream) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
        || GGML_CUDA_CC_IS_CDNA(cc))
        && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, std::bit_cast<internal::ggml_type>(src0->type), (const int*)src1_ddq_i, nullptr, nullptr, dst_dd_i,
        src0->ne[0], row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols };

    ggml_cuda_mul_mat_q_switch_type(ctx.pool(id), args, stream);
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* /*src1_ddq_i*/, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t /*src1_padded_row_size*/, cudaStream_t stream)
{

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(src0_dd_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            int64_t ne = src1_ncols * ne10;
            size_t type_size = ggml_type_size(src1->type);
            src1_as_bf16.alloc(ne);

            convert_context ctx {
                .src_type = std::bit_cast<internal::ggml_type>(src1->type),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
            };
            convert_to_cuda(ctx, src1_ddf_i, src1_as_bf16.get(), stream);
        }
        const nv_bfloat16* src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16*)src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16* src0_ptr = (const nv_bfloat16*)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff * src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                row_diff, src1_ncols, ne10,
                &alpha_f32, src0_ptr, CUDA_R_16BF, ne00,
                src1_ptr, CUDA_R_16BF, ne10,
                &beta_f32, dst_bf16.get(), CUDA_R_16BF, ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const int64_t ne = row_diff * src1_ncols;
        const size_t type_size = ggml_type_size(GGML_TYPE_BF16);
        convert_context ctx{
            .src_type = std::bit_cast<internal::ggml_type>(GGML_TYPE_BF16),
            .src_ne = { ne, 1, 1, 1 },
            .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
        };
        convert_to_cuda(ctx, dst_bf16.get(), dst_dd_i, stream);
    }
    else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            int64_t ne = row_diff * ne00;
            size_t type_size = ggml_type_size(src0->type);
            src0_as_f16.alloc(ne);
            convert_context ctx{
                .src_type = std::bit_cast<internal::ggml_type>(src0->type),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne },
            };
            convert_to_cuda(ctx, src0_dd_i, src0_as_f16.get(), stream);
        }
        const half* src0_ptr = src0->type == GGML_TYPE_F16 ? (const half*)src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            int64_t ne = src1_ncols * ne10;
            size_t type_size = ggml_type_size(src1->type);
            src1_as_f16.alloc(ne);
            convert_context ctx{
                .src_type = std::bit_cast<internal::ggml_type>(src1->type),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne },
            };
            convert_to_cuda(ctx, src1_ddf_i, src1_as_f16.get(), stream);
        }
        const half* src1_ptr = src1->type == GGML_TYPE_F16 ? (const half*)src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ptr, CUDA_R_16F, ne00,
                    src1_ptr, CUDA_R_16F, ne10,
                    &beta, dst_dd_i, CUDA_R_32F, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff * src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr, CUDA_R_16F, ne00,
                    src1_ptr, CUDA_R_16F, ne10,
                    &beta_f16, dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const int64_t ne = row_diff * src1_ncols;
            const size_t type_size = ggml_type_size(GGML_TYPE_F16);
            convert_context ctx{
                .src_type = std::bit_cast<internal::ggml_type>(GGML_TYPE_F16),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
            };
            convert_to_cuda(ctx, dst_f16.get(), dst_dd_i, stream);
        }
    }
    else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            src0_ddq_as_f32.alloc(row_diff * ne00);
            const int64_t ne = row_diff * ne00;
            const size_t type_size = ggml_type_size(src0->type);
            convert_context ctx{
                .src_type = std::bit_cast<internal::ggml_type>(src0->type),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
            };
            convert_to_cuda(ctx, src0_dd_i, src0_ddq_as_f32.get(), stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            src1_ddq_as_f32.alloc(src1_ncols * ne10);
            const int64_t ne = src1_ncols * ne10;
            const size_t type_size = ggml_type_size(src1->type);
            convert_context ctx{
                .src_type = std::bit_cast<internal::ggml_type>(src1->type),
                .src_ne = { ne, 1, 1, 1 },
                .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
            };
            convert_to_cuda(ctx, src1_ddf_i, src1_ddq_as_f32.get(), stream);
        }

        const float* src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float*)src0_dd_i : src0_ddq_as_f32.get();
        const float* src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float*)src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                row_diff, src1_ncols, ne10,
                &alpha, src0_ddf_i, ne00,
                src1_ddf1_i, ne10,
                &beta, dst_dd_i, ldc));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char*, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t, cudaStream_t stream) {

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0 = dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row = ne00;
    const int64_t stride_col_y = ne10;
    const int64_t stride_col_dst = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x = 1;
    const int64_t nchannels_y = 1;
    const int64_t nchannels_dst = 1;
    const int64_t stride_channel_x = 0;
    const int64_t stride_channel_y = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x = 1;
    const int64_t nsamples_dst = 1;
    const int64_t stride_sample_x = 0;
    const int64_t stride_sample_y = 0;
    const int64_t stride_sample_dst = 0;

    mul_mat_vec_f_context ctx1{
        .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
        .src0_d = src0_dd_i,
        .src1_d = src1_ddf_i,
        .ids_d = nullptr,
        .fusion_local = {},
        .dst_d = dst_dd_i,
        .ne00 = src0->ne[0],
        .ne01 = row_diff,
        .ne02 = nchannels_x,
        .ne03 = nsamples_x,
        .ne3 = nsamples_dst,

        .ncols_dst = src1_ncols,
        .nchannels_y = nchannels_y,
        .nchannels_dst = nchannels_dst,
        .stride_channel_dst = stride_channel_dst,
        .stride_channel_y = stride_channel_y,
        
        .s01 = stride_row,
        .s02 = stride_channel_x,
        .s03 = stride_sample_x,
        .s11 = stride_col_y,
        .s13 = stride_sample_y,
        .s1 = stride_col_dst,
        .s3 = stride_sample_dst,
        .prec = std::bit_cast<internal::ggml_prec>(fast_fp16_available(cc) ? dst->op_params[0] : GGML_PREC_F32)
    };

    mul_mat_vec_f_cuda(&ctx1, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream)
{
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    mat_vec_q_switch_context ctx1 {
        .type_x = std::bit_cast<internal::ggml_type>(src0->type),
        .vx = src0_dd_i,
        .vy = src1_ddq_i,
        .ids = nullptr,
        .fusion = {},
		.dst = dst_dd_i,
        .ncols_x = ne00,
        .nrows_x = row_diff,
		.ncols_dst = src1_ncols,
        .stride_row_x = stride_row_x,
        .stride_col_y = stride_col_y,
        .stride_col_dst = nrows_dst,
        .nchannels_x = 1,
        .nchannels_y = 1,
        .nchannels_dst = 1,
        .stride_channel_x = 1,
        .stride_channel_y = 1,
        .stride_channel_dst = 1,
        .nsamples_x = 1,
        .nsamples_dst = 1,
        .stride_sample_x = 1,
        .stride_sample_y = 1,
        .stride_sample_dst = 1
    };

    mul_mat_vec_q_switch_type(ctx1, stream);
}

template <ggml_type type>
const float alphaVal = 1.0f;

template <ggml_type type>
const float betaVal = 0.0f;

template <>
const half alphaVal<GGML_TYPE_F16> = 1.0;

template <>
const half betaVal<GGML_TYPE_F16> = 0.0;

template<ggml_type src0_type, typename src0_t, cublasComputeType_t compute_type, cudaDataType_t data_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(dynamic_cast<cuda_split_backend_buffer_type*>(src0->buffer->get_type()) == nullptr);
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    const int64_t ne_dst = dst->nelements();
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float* dst_ddf = (float*)dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(src1->nb[0] == ts_src1);
    int64_t s11 = src1->nb[1] / ts_src1;
    int64_t s12 = src1->nb[2] / ts_src1;
    int64_t s13 = src1->nb[3] / ts_src1;

    const src0_t* src0_ptr = nullptr;
    const src0_t* src1_ptr = nullptr;

    ggml_cuda_pool_alloc<src0_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<src0_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const src0_t*)src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const src0_t*)src1->data;
    }
    else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = src1->nelements();
        src1_alloc.alloc(ne_src1);

        convert_context ctx {
            .src_type = std::bit_cast<internal::ggml_type>(src1->type),
            .src_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
            .src_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] },
        };
        convert_to_cuda(ctx, src1->data, src1_alloc.get(), main_stream);
        src1_ptr = src1_alloc.get();
        s11 = src1->ne[0];
        s12 = src1->ne[1] * s11;
        s13 = src1->ne[2] * s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<src0_t> dst_temp(ctx.pool());
    char* dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = compute_type;
    cudaDataType_t cu_data_type = data_type;
    const void* alpha = &alphaVal<src0_type>;
    const void* beta = &betaVal<src0_type>;
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char*)dst_ddf;  // Direct F32 output
        }
        else {
            dst_t = (char*)dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(src0_t);
            nbd3 /= sizeof(float) / sizeof(src0_t);
        }
    }
    else {
        dst_t = (char*)dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    GGML_ASSERT(src1->ne[2] % src0->ne[2] == 0);
    GGML_ASSERT(src1->ne[3] % src0->ne[3] == 0);

    // broadcast factors
    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and src0->ne[2]==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = src0->ne[2] == 1 ? src0->nb[3] / src0->nb[0] : src0->nb[2] / src0->nb[0];
        const int64_t smb = src1->ne[2] == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
            cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                alpha, src0_ptr, /*type_a*/data_type, src0->nb[1] / src0->nb[0], sma,     // strideA
                src1_ptr, /*type_b*/data_type, s11, smb,     // strideB
                beta, dst_t, cu_data_type, dst->ne[0], dst->ne[1] * dst->ne[0], // strideC
                src1->ne[2] * src1->ne[3],
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = src1->ne[2] * src1->ne[3];

        ggml_cuda_pool_alloc<const void*> ptrs_src(ctx.pool(), 2 * ne23);
        ggml_cuda_pool_alloc<      void*> ptrs_dst(ctx.pool(), 1 * ne23);

        size_t src1_stride_size = sizeof(src0_t);

        k_compute_batched_ptrs_cuda(
            src0_ptr, src1_ptr, dst_t,
            ptrs_src.get(), ptrs_dst.get(),
            src1->ne[2], src1->ne[3],
            ne23,
            src0->nb[2], src0->nb[3],
            (src1->type == src0_type) ? src1->nb[2] : s12 * src1_stride_size,
            (src1->type == src0_type) ? src1->nb[3] : s13 * src1_stride_size,
            nbd2, nbd3,
            r2, r3, main_stream);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                alpha, (const void**)(ptrs_src.get() + 0 * ne23), /*type_a*/data_type, src0->nb[1] / src0->nb[0],
                (const void**)(ptrs_src.get() + 1 * ne23), /*type_b*/data_type, s11,
                beta, (void**)(ptrs_dst.get() + 0 * ne23), cu_data_type, dst->ne[0],
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const int64_t ne = ne_dst;
        const size_t type_size = ggml_type_size(src0_type);
        convert_context ctx{
            .src_type = std::bit_cast<internal::ggml_type>(src0_type),
            .src_ne = { ne, 1, 1, 1 },
            .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
        };
        convert_to_cuda(ctx, dst_temp.get(), dst_ddf, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
    case GGML_TYPE_F32:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32, float, CUBLAS_COMPUTE_32F, CUDA_R_32F>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_BF16:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16, nv_bfloat16, CUBLAS_COMPUTE_32F, CUDA_R_16BF>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_F16:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16, half, CUBLAS_COMPUTE_16F, CUDA_R_16F>(ctx, src0, src1, dst);
        break;
    default:
        GGML_ABORT("Unsupported type");
    }
}

void ggml_backend_cuda::mul_mat(ggml_tensor* dst)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const bool split = to_split_buffer_type(src0->buffer->get_type()) != nullptr;

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && src0->nbytes() != src0->buffer->get_alloc_size(src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        auto buft_ctx = (cuda_split_backend_buffer_type*)src0->buffer->get_type();
        auto& tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc = ggml_cuda_info().devices[id].cc;
            const int warp_size = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q = use_mul_mat_q && utils::should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
            use_mul_mat_f = use_mul_mat_f && utils::should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f = use_mul_mat_vec_f && utils::should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_hardware_available(cc);
        }
    }
    else {
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        use_mul_mat_q = use_mul_mat_q && utils::should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
        use_mul_mat_f = use_mul_mat_f && utils::should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f = use_mul_mat_vec_f && utils::should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_hardware_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    //TODO update for generic tensor parallelism
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16 = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32 = src0->type == GGML_TYPE_F32;

    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        op::mul_mat_vec_f(stream(), src0, src1, nullptr, dst);
    }
    else if (!split && use_mul_mat_f) {
        op::mul_mat_f(pool(), stream(), nullptr, dst);
    }
    else if (!split && use_mul_mat_vec_q) {
        op::mul_mat_vec_q(pool(), stream(), src0, src1, nullptr, dst);
    }
    else if (!split && use_mul_mat_q) {
        op::mul_mat_q(pool(), stream(), nullptr, dst);
    }
    else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(*this, src0, src1, dst);
    }
    else if (use_mul_mat_vec_f) {
        op_mul_mat(*this, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    }
    else if (use_mul_mat_vec_q) {
        op_mul_mat(*this, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    }
    else if (use_mul_mat_q) {
        op_mul_mat(*this, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    }
    else {
        op_mul_mat(*this, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

void ggml_backend_cuda::set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, stream()));
}

void ggml_backend_cuda::get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, stream()));
}

bool ggml_backend_cuda::cpy_tensor_async(ggml_backend* backend_src, const ggml_tensor* src, ggml_tensor* dst)
{
    ggml_backend_buffer* buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer* buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cuda* cuda_backend_src = dynamic_cast<ggml_backend_cuda*>(backend_src);
    if (!cuda_backend_src) return false;

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    cuda_backend_buffer* cuda_buf_src = dynamic_cast<cuda_backend_buffer*>(buf_src);
    cuda_backend_buffer* cuda_buf_dst = dynamic_cast<cuda_backend_buffer*>(buf_dst);

    if (cuda_backend_src->device != cuda_buf_src->device || this->device != cuda_buf_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: backend and buffer devices do not match", __func__);
#endif
        return false;
    }

    if (cuda_backend_src != this) {
        // copy on src stream
        if (cuda_backend_src->device == device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, cuda_backend_src->stream()));
        }
        else {
            if constexpr (ggml_cuda_no_peer_copy_v) {
                return false;
            }
            else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, device, src->data, cuda_backend_src->device, dst->nbytes(), cuda_backend_src->stream()));
            }
        }

        // record event on src stream after the copy
        if (!cuda_backend_src->copy_event) {
            ggml_cuda_set_device(cuda_backend_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_backend_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_backend_src->copy_event, cuda_backend_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(stream(), cuda_backend_src->copy_event, 0));
    }
    else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, cuda_backend_src->stream()));
    }

    return true;
}

void ggml_backend_cuda::synchronize()
{
    CUDA_CHECK(cudaStreamSynchronize(stream()));
}

static void ggml_cuda_graph_update_executable(ggml_cuda_graph& cuda_graph) {
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(cuda_graph.instance, cuda_graph.graph, &result_info);

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: CUDA graph update failed", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph.instance));
        cuda_graph.instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph.instance, cuda_graph.graph, NULL, NULL, 0));
    }
    else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

void ggml_backend_cuda::graph_evaluate_and_capture(ggml_cgraph* cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required)
{
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated = ggml_cuda_info().devices[device].integrated;

    ggml_cuda_stream_context& stream_ctx = stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event* concurrent_event = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor* node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching {} streams at {}\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = stream();  // this should be stream 0
            GGML_ASSERT(curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = this->stream(this->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto& [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }
            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor*, int> node_to_idx;
                node_to_idx.reserve(cgraph->nodes.size());
                for (int i = 0; i < cgraph->nodes.size(); ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto& [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor* orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        }
                        else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::ranges::sort(sorted_positions);

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i - 1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor*>(event.original_order[i]);
                    }
                }
            }
            else {
                stream_ctx.concurrent_events.clear();
            }

            for (size_t i = 0; i < cgraph->nodes.size(); i++) {
                auto node = cgraph->nodes[i];

                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                stream(this->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event = nullptr;
                    }
                    else {
                        GGML_ASSERT(concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to {} for node {}\n", curr_stream_no, node->name);
                    }
                }
                else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor* prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to {} for node {}\n", curr_stream_no, node->name);
                    }
                }

                if constexpr (ggml_cuda_debug_v) {
                    const int nodes_fused = i - prev_i - 1;
                    if (nodes_fused > 0) {
                        GGML_LOG_INFO("nodes_fused: {}", nodes_fused);
                    }
                }

                prev_i = i;

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }


                // start of fusion operations
                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion) {

                    if (fused::ggml_cuda_can_fuse(cgraph, i, fused::ggml_cuda_topk_moe_ops(/*with norm*/ true), {})) {
                        ggml_tensor* weights = cgraph->nodes[i + 9];
                        ggml_tensor* selected_experts = cgraph->nodes[i + 3];
                        ggml_tensor* clamp = cgraph->nodes[i + 7];
                        fused::topk_moe(stream(), node->src[0], weights, selected_experts, /*with norm*/ true,
                            /*delayed softmax*/ false, clamp);
                        i += 9;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i, fused::ggml_cuda_topk_moe_ops(/*with norm*/ false), {})) {
                        ggml_tensor* weights = cgraph->nodes[i + 4];
                        ggml_tensor* selected_experts = cgraph->nodes[i + 3];
                        fused::topk_moe(stream(), node->src[0], weights, selected_experts, /*with norm*/ false,
                            /*delayed softmax*/ false);
                        i += 4;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i,
                        fused::ggml_cuda_topk_moe_ops(/*with norm*/ false, /*delayed softmax*/ true), {})) {
                        ggml_tensor* weights = cgraph->nodes[i + 5];
                        ggml_tensor* ids = cgraph->nodes[i + 1];

                        fused::topk_moe(stream(), node->src[0], weights, ids, /*with norm*/ false,
                            /*delayed_softmax*/ true);
                        i += 5;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                        ggml_tensor* rope = cgraph->nodes[i];
                        ggml_tensor* set_rows = cgraph->nodes[i + 2];

                        op::rope(stream(), rope, true, set_rows);
                        i += 2;
                        continue;
                    }

                    if (node->op == GGML_OP_ADD) {
                        int n_fuse = 0;
                        ggml_op ops[8];
                        std::fill(ops, ops + 8, GGML_OP_ADD);

                        for (; n_fuse <= 6; ++n_fuse) {
                            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                                break;
                            }
                            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                                break;
                            }
                            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                                break;
                            }
                        }

                        n_fuse++;

                        if (n_fuse > 1) {
                            for (int j = 0; j < n_fuse - 1; ++j) {
                                node->src.push_back(cgraph->nodes[i + j + 1]->src[1]);
                            }
                            cgraph->nodes[i + n_fuse - 1]->data = node->data;
                            fused::add(stream(), node, n_fuse);
                            i += n_fuse - 1;

                            continue;
                        }
                    }

                    bool fused_mul_mat_vec = false;
                    int fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (fused::ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
                            ggml_tensor* glu = cgraph->nodes[i + 4];
                            ggml_tensor* gate_bias_n = glu->src[0];
                            ggml_tensor* up_bias_n = glu->src[1];

                            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
                            ggml_tensor* gate_n = nullptr;
                            ggml_tensor* up_n = nullptr;

                            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                                gate_n = cgraph->nodes[i];
                                up_n = cgraph->nodes[i + 2];
                            }
                            else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                                gate_n = cgraph->nodes[i + 2];
                                up_n = cgraph->nodes[i];
                            }
                            else {
                                continue;
                            }

                            auto get_bias_tensor = [](const ggml_tensor* bias_node, const ggml_tensor* mul_node, ggml_op op_bias) {
                                if (op_bias == GGML_OP_ADD) {
                                    if (bias_node->src[0] == mul_node) {
                                        return bias_node->src[1];
                                    }
                                    if (bias_node->src[1] == mul_node) {
                                        return bias_node->src[0];
                                    }
                                    return (ggml_tensor*) nullptr;
                                }
                                GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
                                GGML_ASSERT(bias_node->src[0] == mul_node);
                                return bias_node->src[1];
                            };

                            ggml_tensor* up_bias_tensor = get_bias_tensor(up_bias_n, up_n, bias_op);
                            ggml_tensor* gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

                            if (!up_bias_tensor || !gate_bias_tensor) {
                                continue;
                            }

                            // we don't support repeating adds
                            if (bias_op == GGML_OP_ADD &&
                                (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                    !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                                continue;
                            }

                            const ggml_tensor* src0 = up_n->src[0];
                            const ggml_tensor* src1 = up_n->src[1];
                            const ggml_tensor* ids = up_n->src[2];

                            if (fused::should_mul_mat_vec_f(up_n)) {
                                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate = gate_n->src[0];
                                fusion_data.x_bias = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                op::mul_mat_vec_f(stream(), src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }

                            if (fused::should_mul_mat_vec_q(up_n)) {
                                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate = gate_n->src[0];
                                fusion_data.x_bias = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                op::mul_mat_vec_q(pool(), stream(), src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }

                        }
                        else if (fused::ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
                            ggml_tensor* glu = cgraph->nodes[i + 2];
                            ggml_tensor* gate = glu->src[0];
                            ggml_tensor* up = glu->src[1];

                            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1])
                                || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

                            if (!ok) continue;

                            const ggml_tensor* src0 = up->src[0];
                            const ggml_tensor* src1 = up->src[1];
                            const ggml_tensor* ids = up->src[2];

                            if (fused::should_mul_mat_vec_f(up)) {
                                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                op::mul_mat_vec_f(stream(), src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }

                            if (fused::should_mul_mat_vec_q(up)) {
                                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                op::mul_mat_vec_q(pool(), stream(), src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    fused_mul_mat_vec = false;
                    fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (!fused::ggml_can_fuse(cgraph, i, { op, bias_op })) {
                            continue;
                        }

                        ggml_tensor* mm_node = cgraph->nodes[i];
                        ggml_tensor* bias_node = cgraph->nodes[i + 1];

                        ggml_tensor* bias_tensor = nullptr;
                        if (bias_op == GGML_OP_ADD) {
                            if (bias_node->src[0] == mm_node) {
                                bias_tensor = bias_node->src[1];
                            }
                            else if (bias_node->src[1] == mm_node) {
                                bias_tensor = bias_node->src[0];
                            }
                            else {
                                continue;
                            }
                        }
                        else {
                            if (bias_node->src[0] != mm_node) {
                                continue;
                            }
                            bias_tensor = bias_node->src[1];
                        }

                        const ggml_tensor* src0 = mm_node->src[0];
                        const ggml_tensor* src1 = mm_node->src[1];
                        const ggml_tensor* ids = mm_node->src[2];

                        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
                            continue;
                        }

                        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
                            continue;
                        }

                        op::ggml_cuda_mm_fusion_args_host fusion_data{};
                        fusion_data.x_bias = bias_tensor;

                        if (fused::should_mul_mat_vec_f(mm_node)) {
                            op::mul_mat_vec_f(stream(), src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }

                        if (fused::should_mul_mat_vec_q(mm_node)) {
                            op::mul_mat_vec_q(pool(), stream(), src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
                        fused::rms_norm_add(stream(), node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
                        i += 2;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
                        fused::rms_norm(stream(), node, cgraph->nodes[i + 1]);
                        i++;
                        continue;
                    }

                    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                        i += 2;
                        fused::softcap(stream(), cgraph->nodes[i], node);
                        continue;
                    }
                }
#ifndef NDEBUG
                assert(node->buffer->get_type() == ggml_backend_cuda_buffer_type(device));
                for (auto& src : node->src) {
                    if (!src) continue;
                    assert(src->buffer);
                    assert(buffer_type_from_device(src->buffer->get_type(), device));
                }
#endif // NDEBUG

                bool ok = compute_forward(node);
                if (!ok) {
                    GGML_LOG_ERROR("{}: op not supported {} ({})", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
                }
            }
        }

        if constexpr (use_cuda_graph_v) {
            if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
                if (cuda_graph->graph != nullptr) {
                    CUDA_CHECK(cudaGraphDestroy(cuda_graph->graph));
                    cuda_graph->graph = nullptr;
                }

                CUDA_CHECK(cudaStreamEndCapture(stream(), &cuda_graph->graph));
                graph_evaluated_or_captured = true; // CUDA graph has been captured

                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }
            else {
                graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
            }
        } else {
            graph_evaluated_or_captured = true;
        }
    }

    if constexpr (use_cuda_graph_v) {
        if (use_cuda_graph) {
            if (cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
                CUDA_CHECK(cudaGraphInstantiate(&cuda_graph->instance, cuda_graph->graph, NULL, NULL, 0));
            }
            if (cuda_graph_update_required) { // Update graph executable
                ggml_cuda_graph_update_executable(*cuda_graph);
            }
            // Launch graph
            CUDA_CHECK(cudaGraphLaunch(cuda_graph->instance, stream()));
        }
    }
}

static bool ggml_cuda_graph_set_enabled(int device, std::unique_ptr<ggml_cuda_graph>& cuda_graph) {
    if constexpr (use_cuda_graph_v) {
        if (cuda_graph == nullptr) {
            cuda_graph.reset(new ggml_cuda_graph());
        }

        if (cuda_graph->graph == nullptr) {
            if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) {
                cuda_graph->disable_due_to_gpu_arch = true;
                GGML_LOG_DEBUG("{}: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
        }
        return cuda_graph->is_enabled();
    }
    else {
        return false;
    }
}

static bool ggml_cuda_graph_node_properties_match(ggml_tensor* node, ggml_cuda_graph_node_properties* props) {
    if (node->data != props->node_address &&
        node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != props->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != props->ne[i]) {
            return false;
        }
        if (node->nb[i] != props->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != props->src_address[i] &&
            node->op != GGML_OP_VIEW
            ) {
            return false;
        }
    }

    if ((node->op == GGML_OP_SCALE || node->op == GGML_OP_GLU) &&
        memcmp(props->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static void ggml_cuda_graph_node_set_properties(ggml_cuda_graph_node_properties* props, ggml_tensor* node) {
    props->node_address = node->data;
    props->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        props->ne[i] = node->ne[i];
        props->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        props->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(props->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_cuda_graph_update_required(ggml_cuda_graph* cuda_graph, ggml_cgraph* cgraph) {
    bool res = false;

    if (cuda_graph->instance == nullptr) {
        res = true;
    }

    // Check if the graph size has changed
    if (cuda_graph->props.size() != (size_t)cgraph->nodes.size() + cgraph->leafs.size()) {
        res = true;
        cuda_graph->props.resize(cgraph->nodes.size() + cgraph->leafs.size());
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->nodes.size(); i++) {
        bool props_match = true;
        if (!res) {
            props_match = ggml_cuda_graph_node_properties_match(cgraph->nodes[i], &cuda_graph->props[i]);
        }
        if (!props_match) {
            res = true;
        }
        ggml_cuda_graph_node_set_properties(&cuda_graph->props[i], cgraph->nodes[i]);
    }

    for (int i = 0; i < cgraph->leafs.size(); i++) {
        bool props_match = true;
        if (!res) {
            props_match = ggml_cuda_graph_node_properties_match(cgraph->leafs[i], &cuda_graph->props[cgraph->nodes.size() + i]);
        }
        if (!props_match) {
            res = true;
        }
        ggml_cuda_graph_node_set_properties(&cuda_graph->props[cgraph->nodes.size() + i], cgraph->leafs[i]);
    }

    return res;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph* cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";
    const std::string nemotron_h_block_out_prefix = "nemotron_h_block_out";
    const std::string mamba2_y_add_d_prefix = "mamba2_y_add_d";

    for (int i = 0; i < cgraph->nodes.size(); i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        const bool is_cuda_split_buffer_type = [=]() {
            if (!node->src[0]) return false;
            if (!node->src[0]->buffer) return false;
            auto type = node->src[0]->buffer->get_type();
            return dynamic_cast<cuda_split_backend_buffer_type*>(type) != nullptr;

        }();
        if (is_cuda_split_buffer_type) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
        }

        if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] != 1) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
        }

        if (node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            !node->name.starts_with(ffn_moe_gate_bias_prefix) &&
            !node->name.starts_with(ffn_moe_up_bias_prefix) &&
            !node->name.starts_with(ffn_moe_down_bias_prefix) &&
            !node->name.starts_with(nemotron_h_block_out_prefix) &&
            !node->name.starts_with(mamba2_y_add_d_prefix)) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

enum ggml_status ggml_backend_cuda::graph_compute_impl(ggml_cgraph* cgraph)
{
    ggml_cuda_set_device(device);

    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;

    if constexpr (use_cuda_graph_v) {
        use_cuda_graph = ggml_cuda_graph_set_enabled(device, cuda_graph);

        if (cuda_graph->is_enabled()) {
            cuda_graph_update_required = ggml_cuda_graph_update_required(cuda_graph.get(), cgraph);
            use_cuda_graph = ggml_cuda_graph_check_compability(cgraph);

            cuda_graph->record_update(use_cuda_graph, cuda_graph_update_required);
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(stream(), cudaStreamCaptureModeRelaxed));
    }

    graph_evaluate_and_capture(cgraph, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_cuda::event_record(ggml_backend_event* event)
{
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, stream()));
}

void ggml_backend_cuda::event_wait(ggml_backend_event* event)
{
    if (true) {
        CUDA_CHECK(cudaStreamWaitEvent(stream(), (cudaEvent_t)event->context, 0));
    }
    else {
#if 0
        // untested
        auto wait_fn = [](void* user_data) {
            ggml_backend_event* event = (ggml_backend_event*)user_data;
            ggml_backend_event_synchronize(event);
            };

        CUDA_CHECK(cudaLaunchHostFunc(stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

bool ggml_backend_cuda::compute_forward(ggml_tensor* dst) {
    // why is this here instead of mul_mat?
    if (dst->src.size() > 1 && dst->src[0] != nullptr && to_cuda_buffer_type(dst->src[0]->buffer->get_type())) {
        const ggml_tensor* src1 = dst->src[1];
        if (src1) ggml_cuda_set_peer_access(src1->ne[1], device);
    }

    switch (dst->op) {
    case GGML_OP_ARGMAX:
        op::argmax(stream(), dst);
        break;
    case GGML_OP_COUNT_EQUAL:
        op::count_equal(stream(), dst);
        break;
    case GGML_OP_REPEAT:
        op::repeat(stream(), dst);
        break;
    case GGML_OP_REPEAT_BACK:
        op::repeat_back(stream(), dst);
        break;
    case GGML_OP_GET_ROWS:
        op::get_rows(stream(), dst);
        break;
    case GGML_OP_GET_ROWS_BACK:
        op::get_rows_back(stream(), dst);
        break;
    case GGML_OP_SET_ROWS:
        op::set_rows(stream(), dst);
        break;
    case GGML_OP_SET:
        op::set(stream(), dst);
        break;
    case GGML_OP_DUP:
    case GGML_OP_CONT:
        op::dup(stream(), dst);
        break;
    case GGML_OP_CPY:
        op::cpy(stream(), dst);
        break;
    case GGML_OP_ADD:
    case GGML_OP_ADD1: // TODO: more efficient implementation
        op::add(stream(), dst);
        break;
    case GGML_OP_ADD_ID:
        op::add_id(stream(), dst);
        break;
    case GGML_OP_SUB:
        op::sub(stream(), dst);
        break;
    case GGML_OP_ACC:
        op::acc(stream(), dst);
        break;
    case GGML_OP_MUL:
        op::mul(stream(), dst);
        break;
    case GGML_OP_DIV:
        op::div(stream(), dst);
        break;
    case GGML_OP_UNARY:
        op::unary(stream(), dst);
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            op::glu(stream(), dst);
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            op::swiglu_oai(stream(), dst);
			break;
        default:
            return false;
        }
        break;
    case GGML_OP_NORM:
        op::norm(stream(), dst);
        break;
    case GGML_OP_GROUP_NORM:
        op::group_norm(stream(), dst);
        break;
    case GGML_OP_L2_NORM:
        op::l2_norm(stream(), dst);
        break;
    case GGML_OP_CONCAT:
        op::concat(stream(), dst);
        break;
    case GGML_OP_UPSCALE:
        op::upscale(stream(), dst);
        break;
    case GGML_OP_PAD:
        op::pad(stream(), dst);
        break;
    case GGML_OP_PAD_REFLECT_1D:
        op::pad_reflect_1d(stream(), dst);
        break;
    case GGML_OP_ARANGE:
        op::arange(stream(), dst);
        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        op::timestep_embedding(stream(), dst);
        break;
    case GGML_OP_LEAKY_RELU:
        op::leaky_relu(stream(), dst);
        break;
    case GGML_OP_SILU_BACK:
        op::silu_back(stream(), dst);
        break;
    case GGML_OP_RMS_NORM:
        op::rms_norm(stream(), dst);
        break;
    case GGML_OP_RMS_NORM_BACK:
        op::rms_norm_back(stream(), dst);
        break;
    case GGML_OP_MUL_MAT:
        mul_mat(dst);
        break;
    case GGML_OP_MUL_MAT_ID:
        op::mul_mat_id(pool(), stream(), dst, [this](ggml_tensor* dst) {
            mul_mat(dst);
        });
        break;
    case GGML_OP_OUT_PROD:
        op::out_prod(stream(), cublas_handle(), dst);
        break;
    case GGML_OP_SCALE:
        op::scale(stream(), dst);
        break;
    case GGML_OP_SQR:
        op::sqr(stream(), dst);
        break;
    case GGML_OP_SQRT:
        op::sqrt(stream(), dst);
        break;
    case GGML_OP_SIN:
        op::sin(stream(), dst);
        break;
    case GGML_OP_COS:
        op::cos(stream(), dst);
        break;
    case GGML_OP_CLAMP:
        op::clamp(stream(), dst);
        break;
    case GGML_OP_LOG:
        op::log(stream(), dst);
        break;
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        break;
    case GGML_OP_DIAG:
        op::diag(stream(), dst);
        break;
    case GGML_OP_DIAG_MASK_INF:
        op::diag_mask_inf(stream(), dst);
        break;
    case GGML_OP_SOFT_MAX:
        op::soft_max(pool(), stream(), dst);
        break;
    case GGML_OP_SOFT_MAX_BACK:
        op::soft_max_back(stream(), dst);
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        op::rope(stream(), dst, (dst->op == GGML_OP_ROPE) ? true : false);
        break;
    case GGML_OP_ROLL:
        op::roll(stream(), dst);
        break;
    case GGML_OP_IM2COL:
        op::im2col(stream(), dst);
        break;
    case GGML_OP_IM2COL_3D:
        op::im2col_3d(stream(), dst);
        break;
    case GGML_OP_CONV_2D:
        op::conv2d(stream(), dst);
        break;
    case GGML_OP_CONV_2D_DW:
        op::conv2d_dw(stream(), dst);
        break;
    case GGML_OP_CONV_TRANSPOSE_2D:
        op::conv_2d_transpose_p0(stream(), dst);
        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        op::conv_transpose_1d(stream(), dst);
        break;
    case GGML_OP_POOL_2D:
        op::pool2d(stream(), dst);
        break;
    case GGML_OP_SUM:
        op::sum(pool(), stream(), dst);
        break;
    case GGML_OP_CUMSUM:
        op::cumsum(pool(), stream(), dst);
        break;
    case GGML_OP_SUM_ROWS:
        op::sum_rows(stream(), dst);
        break;
    case GGML_OP_MEAN: {
        const bool cuda_graph_exists = [=, this]() {
            if constexpr (use_cuda_graph_v) {
                return cuda_graph->instance != nullptr;
            }
            else {
                return false;
            }
        }();
        const bool cuda_graph_enable = [=, this]() {
            if constexpr (use_cuda_graph_v) {
                return cuda_graph->is_enabled();
            }
            else {
                return false;
            }
        }();
        op::mean(pool(), stream(), cuda_graph_exists, cuda_graph_enable, dst);
        break;
    }
    case GGML_OP_SSM_CONV:
        op::ssm_conv(stream(), dst);
        break;
    case GGML_OP_TOP_K:
        op::top_k(pool(), stream(), dst);
        break;
    case GGML_OP_SSM_SCAN:
        op::ssm_scan(stream(), dst);
        break;
    case GGML_OP_ARGSORT:
        op::argsort(pool(), stream(), dst);
        break;
    case GGML_OP_FLASH_ATTN_EXT:
        op::flash_attn_ext(device, pool(), stream(), dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS:
        op::cross_entropy_loss(pool(), stream(), dst);
        break;
    case GGML_OP_TRI:
        op::tri(stream(), dst);
        break;
    case GGML_OP_RWKV_WKV6:
        op::rwkv_wkv6(stream(), dst);
        break;
    case GGML_OP_GATED_LINEAR_ATTN:
        op::gated_linear_attn(stream(), dst);
        break;
    case GGML_OP_RWKV_WKV7:
        op::rwkv_wkv7(stream(), dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        op::cross_entropy_loss_back(pool(), stream(), dst);
        break;
    case GGML_OP_OPT_STEP_ADAMW:
        op::opt_step_adamw(stream(), dst);
        break;
    case GGML_OP_OPT_STEP_SGD:
        op::opt_step_sgd(stream(), dst);
        break;
    case GGML_OP_SOLVE_TRI: {
        const int id = ggml_cuda_get_device();
        op::solve_tri(pool(id), cublas_handle(id), stream(), dst);
        break;
    }
    case GGML_OP_FILL:
        op::fill(stream(), dst);
        break;
    default:
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("{}: {} failed", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

ggml_backend_cuda::~ggml_backend_cuda()
{
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, [] { return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }
}

bool ggml_backend_cuda::graph_set_enabled() {
    if constexpr (use_cuda_graph_v) {
        if (!cuda_graph) {
            cuda_graph = std::make_unique<ggml_cuda_graph>();
        }

        if (!cuda_graph->graph) {
            if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) {
                cuda_graph->disable_due_to_gpu_arch = true;
                GGML_LOG_DEBUG("{}: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
        }

        return cuda_graph->is_enabled();
    }
    else {
        return false;
    }
}

cudaStream_t ggml_backend_cuda::stream(int device, int stream) {
    if (streams[device][stream] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
    }
    return streams[device][stream];
}

ggml_cuda_pool& ggml_backend_cuda::pool(int device) {
    if (pools[device][curr_stream_no] == nullptr) {
        pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no);
    }
    return *pools[device][curr_stream_no];
}

cublasHandle_t ggml_backend_cuda::cublas_handle(int device) {
    if (cublas_handles[device] == nullptr) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return cublas_handles[device];
}

void ggml_backend_cuda::graph_optimize(ggml_cgraph* cgraph) {
    const bool use_cuda_graph = graph_set_enabled();

    static bool enable_graph_optimization = [] {
        const char* env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context& stream_context = this->stream_context();
    stream_context.reset();

    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
        return;
    }

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor*, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor*, int> node_indices;

    const auto& is_noop = [](const ggml_tensor* node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto& depends_on = [](const ggml_tensor* dst, const ggml_tensor* src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor* dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor* src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->nodes.size(); node_idx++) {
        const ggml_tensor* node = cgraph->nodes[node_idx];
        node_indices[node] = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor* src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto& [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!root_node->name.starts_with("attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto& [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor*>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->nodes.size(); ++i) {
                const ggml_tensor* node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t)count);

            //find the join point
            const ggml_tensor* join_node = nullptr;

            const auto& belongs_to_branch = [&](const ggml_tensor* node,
                                                const std::vector<const ggml_tensor*>& branch) -> bool {
                for (const ggml_tensor* n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->nodes.size(); ++i) {
                const ggml_tensor* curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor*>& branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor* n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx = fork_node_idx + 1;
                const int n_branches = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor*> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor*, ggml_cuda_concurrent_event>& concurrent_events = this->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                //GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor*>& branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor*> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor*>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor*>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}
