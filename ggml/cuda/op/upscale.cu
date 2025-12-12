#include "helper.h"
#include "cuda_func.h"
#include "launch.cuh"

void upscale_f32_cuda(const upscale_context& ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
            int64_t i00 = i10 / ctx.sf0;
            int64_t i01 = i11 / ctx.sf1;
            int64_t i02 = i12 / ctx.sf2;
            int64_t i03 = i13 / ctx.sf3;

            dst_data(i13, i12, i11, i10) = src0_data(i03, i02, i01, i00);
        }
    );
}

void upscale_f32_bilinear_cuda(const upscale_context& ctx, const float pixel_offset, bool antialias, cudaStream_t stream) {
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    if (antialias) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
            [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
                const int i02_src = (int)(i12 / ctx.sf2);
                const int i03_src = (int)(i13 / ctx.sf3);

                const float y = ((float)i11 + pixel_offset) / ctx.sf1;
                const float x = ((float)i10 + pixel_offset) / ctx.sf0;

                // support and invscale, minimum 1 pixel for bilinear
                const float support1 = max(1.0f / ctx.sf1, 1.0f);
                const float invscale1 = 1.0f / support1;
                const float support0 = max(1.0f / ctx.sf0, 1.0f);
                const float invscale0 = 1.0f / support0;

                // the range of source pixels that contribute
                const int64_t x_min = max(int64_t(0), int64_t(x - support0 + pixel_offset));
                const int64_t x_max = min(int64_t(ctx.src0_ne[0]), int64_t(x + support0 + pixel_offset));
                const int64_t y_min = max(int64_t(0), int64_t(y - support1 + pixel_offset));
                const int64_t y_max = min(int64_t(ctx.src0_ne[1]), int64_t(y + support1 + pixel_offset));

                // bilinear filter with antialiasing
                float val = 0.0f;
                float total_weight = 0.0f;

                auto triangle_filter = [](float x) -> float {
                    return max(1.0f - fabsf(x), 0.0f);
                };

                for (int64_t sy = y_min; sy < y_max; sy++) {
                    const float weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                    for (int64_t sx = x_min; sx < x_max; sx++) {
                        const float weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                        const float weight = weight_x * weight_y;

                        if (weight <= 0.0f) {
                            continue;
                        }

                        val += src0_data(i03_src, i02_src, sy, sx) * weight;
                        total_weight += weight;
                    }
                }

                if (total_weight > 0.0f) {
                    val /= total_weight;
                }

                dst_data(i13, i12, i11, i10) = val;
            }
        );
    }
    else {
        launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
            [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
                const int i02_src = (int)(i12 / ctx.sf2);
                const int i03_src = (int)(i13 / ctx.sf3);

                const float y_src_f = ((float)i11 + pixel_offset) / ctx.sf1 - pixel_offset;
                int64_t y0_src = (int64_t)floorf(y_src_f);
                int64_t y1_src = y0_src + 1;

                y0_src = max(int64_t{ 0 }, min(y0_src, ctx.src0_ne[1] - 1));
                y1_src = max(int64_t{ 0 }, min(y1_src, ctx.src0_ne[1] - 1));

                float dy = y_src_f - (float)y0_src;
                dy = max(0.0f, min(dy, 1.0f));

                float x_src_f = ((float)i10 + pixel_offset) / ctx.sf0 - pixel_offset;
                int64_t x0_src = (int64_t)floorf(x_src_f);
                int64_t x1_src = x0_src + 1;

                x0_src = max(int64_t{ 0 }, min(x0_src, ctx.src0_ne[0] - 1));
                x1_src = max(int64_t{ 0 }, min(x1_src, ctx.src0_ne[0] - 1));

                float dx = x_src_f - (float)x0_src;
                dx = max(0.0f, min(dx, 1.0f));

                const float val_a = src0_data(i03_src, i02_src, y0_src, x0_src);
                const float val_b = src0_data(i03_src, i02_src, y0_src, x1_src);
                const float val_c = src0_data(i03_src, i02_src, y1_src, x0_src);
                const float val_d = src0_data(i03_src, i02_src, y1_src, x1_src);

                float result = val_a * (1.0f - dx) * (1.0f - dy) +
                    val_b * dx * (1.0f - dy) +
                    val_c * (1.0f - dx) * dy +
                    val_d * dx * dy;

                dst_data(i13, i12, i11, i10) = result;
            }
        );
    }
}

namespace bicubic_interpolation {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    __device__ const float a = -0.75f; // use alpha = -0.75 (same as PyTorch)

    static __device__ float weight1(float x) { return ((a + 2) * x - (a + 3)) * x * x + 1; };
    static __device__ float weight2(float x) { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };

    static __device__ float bicubic(float p0, float p1, float p2, float p3, float x) {
        const float w0 = weight2(x + 1);
        const float w1 = weight1(x + 0);
        const float w2 = weight1(1 - x);
        const float w3 = weight2(2 - x);
        return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
    };
} // namespace bicubic_interpolation

void upscale_f32_bicubic_cuda(const upscale_context& ctx, const float pixel_offset, cudaStream_t stream)
{
    using bicubic_interpolation::bicubic;
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
            const int i02_src = (int)(i12 / ctx.sf2);
            const int i03_src = (int)(i13 / ctx.sf3);

            const float y_src_f = ((float)i11 + pixel_offset) / ctx.sf1 - pixel_offset;
            const int64_t y0_src = (int64_t)floorf(y_src_f);
            const float dy = y_src_f - (float)y0_src;

            const float x_src_f = ((float)i10 + pixel_offset) / ctx.sf0 - pixel_offset;
            const int64_t x0_src = (int64_t)floorf(x_src_f);
            const float dx = x_src_f - (float)x0_src;

            auto load = [=](int64_t x_off, int64_t y_off) -> float {
                int64_t i00_src = max(int64_t{ 0 }, min(x0_src + x_off, ctx.src0_ne[0] - 1));
                int64_t i01_src = max(int64_t{ 0 }, min(y0_src + y_off, ctx.src0_ne[1] - 1));
                return src0_data(i03_src, i02_src, i01_src, i00_src);
            };

            const float result = bicubic(
                bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx),
                bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx),
                bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx),
                bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx), dy);

            dst_data(i13, i12, i11, i10) = result;
        }
    );
}
