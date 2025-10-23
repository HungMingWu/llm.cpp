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

void upscale_f32_bilinear_cuda(const upscale_context& ctx, const float pixel_offset, cudaStream_t stream) {
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
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