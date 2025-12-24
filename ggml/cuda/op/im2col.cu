#include "common.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"
#include "cuda_func.h"

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC, KH, KW]
template <typename dst_t>
void im2col_cuda(const float* x, dst_t* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream) {
    std::mdspan src_data(x, N, IC, IH, IW);
    std::mdspan dst_data(dst, N, OH, OW, IC, KH, KW);
    launch_functor(stream, std::make_tuple(N, OH, OW, IC, KH, KW),
        [=] __device__ (int64_t in, int64_t ioh, int64_t iow, int64_t iic, int64_t ikh, int64_t ikw) {
            using dst_t = decltype(dst_data);
            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                dst_data(in, ioh, iow, iic, ikh, ikw) = ggml_cuda_cast<typename dst_t::value_type>(0.0f);
            }
            else {
                dst_data(in, ioh, iow, iic, ikh, ikw) = ggml_cuda_cast<typename dst_t::value_type>(src_data(in, iic, iih, iiw));
            }
        }
    );
}

void im2col_cuda(internal::ggml_type dst_type, const float* x, void* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream)
{
    if (dst_type == internal::GGML_TYPE_F16) {
        im2col_cuda(x, (half*)dst, IW, IH, OW, OH, KW, KH, IC, N, s0, s1, p0, p1, d0, d1, stream);
    }
    else {
        im2col_cuda(x, (float*)dst, IW, IH, OW, OH, KW, KH, IC, N, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N, IC, ID, IH, IW] => [N, OD, OH, OW, IC, KD, KH, KW]
template <typename T>
void im2col_3d_cuda(const float* src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    size_t stride_q, size_t stride_z, size_t stride_y, size_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    std::array<int64_t, 5> src_ne = { IW, IH, ID, IC, N }; // reverse order
    std::array<size_t, 5> src_nb = { stride_x, stride_y, stride_z, stride_q, stride_q * IC };
    auto src_data = make_strided_mdspan<5>(src, src_ne, src_nb);
    std::mdspan dst_data(dst, N, OD, OH, OW, IC, KD, KH, KW);
    launch_functor(stream, std::make_tuple(N, OD, OH, OW, IC, KD, KH, KW),
        [=] __device__ (int64_t in, int64_t iod, int64_t ioh, int64_t iow, int64_t iic, int64_t ikd, int64_t ikh, int64_t ikw) {
            using dst_t = decltype(dst_data);
            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst_data(in, iod, ioh, iow, iic, ikd, ikh, ikw) = ggml_cuda_cast<typename dst_t::value_type>(0.0f);
            }
            else {
                dst_data(in, iod, ioh, iow, iic, ikd, ikh, ikw) = ggml_cuda_cast<typename dst_t::value_type>(src_data(in, iic, iid, iih, iiw));
            }
        }
    );
}

void im2col_3d_cuda(internal::ggml_type dst_type, const float* src1_d, void* dst_d,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    size_t stride_q, size_t stride_z, size_t stride_y, size_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream)
{
    if (dst_type == internal::GGML_TYPE_F16) {
        im2col_3d_cuda(src1_d, (half*)dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
			stride_q, stride_z, stride_y, stride_x,
            s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
    else {
        im2col_3d_cuda(src1_d, (float*)dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
            stride_q, stride_z, stride_y, stride_x,
            s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}