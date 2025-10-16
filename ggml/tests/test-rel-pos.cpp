#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdspan.hpp"

import ggml;

void check_tensor(
    const std::experimental::mdspan<float, std::experimental::dims<3>>& actual_mdspan,
    const std::experimental::mdspan<float, std::experimental::dims<3>>& expected_mdspan)
{
    assert(actual_mdspan.extents() == expected_mdspan.extents());
    for (int i2 = 0; i2 < actual_mdspan.extent(0); ++i2) {
        for (int i1 = 0; i1 < actual_mdspan.extent(1); ++i1) {
            for (int i0 = 0; i0 < actual_mdspan.extent(2); ++i0) {
                float expected = expected_mdspan[i2, i1, i0];
                float actual = actual_mdspan[i2, i1, i0];
                assert(expected == actual);
            }
        }
    }
}

int main(int argc, const char** argv) {
    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f16[i] = fromFloat32<ggml_fp16_t>((float)i);
    }

    float expected_out[4 * 9] = {
        8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
        2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0,
        14.0, 15.0, 16.0, 15.0, 16.0, 17.0, 16.0, 17.0, 18.0,
        8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0
    };

    {
        ggml_context ctx;

        ggml_tensor* t = ctx.create(GGML_TYPE_F16, { 3, 3 });

        ggml_tensor* t_2 = ctx.create(GGML_TYPE_F16, { 3, 3 });
        ggml_tensor* rw = ggml_get_rel_pos(&ctx, t, 2, 2);
        ggml_tensor* rh = ggml_get_rel_pos(&ctx, t_2, 2, 2);

        ggml_tensor* rw_f32 = ggml_cpy(&ctx, rw, ctx.create(GGML_TYPE_F32, { 3, 2, 2 }));
        ggml_tensor* rh_f32 = ggml_cpy(&ctx, rh, ctx.create(GGML_TYPE_F32, { 3, 2, 2 }));

        ggml_tensor* in = ctx.create(GGML_TYPE_F32, { 9, 4 });
        ggml_tensor* out_inplace = ctx.create(GGML_TYPE_F32, { 9, 4 });

        ggml_tensor* out = ggml_add_rel_pos(&ctx, in, rw_f32, rh_f32, false);
        ggml_cgraph gf;
        gf.build_forward_expand(out);

        auto backend = ggml_backend_cpu_init();
        auto buffer = backend->alloc_tensors(&ctx);
        ggml_backend_tensor_set(t, buf_f16, 0, t->nbytes());
        ggml_backend_tensor_set(t_2, buf_f16 + 1, 0, t_2->nbytes());
        std::vector<float> fill_data(9 * 4, 1.f);
        ggml_backend_tensor_set(in, fill_data.data(), 0, in->nbytes());
        ggml_backend_tensor_set(out_inplace, fill_data.data(), 0, out_inplace->nbytes());
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.alloc_graph(&gf);
        backend->compute(&gf);

        out_inplace = ggml_add_rel_pos(&ctx, out_inplace, rw_f32, rh_f32, true);
        ggml_cgraph gf_2;
        gf_2.build_forward_expand(out_inplace);
        allocr.alloc_graph(&gf_2);
        backend->compute(&gf_2);

        assert(out->type == GGML_TYPE_F32 && out_inplace->type == GGML_TYPE_F32);
        std::vector<float> out_result(out->nelements());
        ggml_backend_tensor_get(out, out_result.data(), 0, out->nbytes());
        std::experimental::mdspan out_result_mdspan(out_result.data(), out->ne[2], out->ne[1], out->ne[0]);
        std::experimental::mdspan expected_out_mdspan(expected_out, 1, 4, 9);
        check_tensor(out_result_mdspan, expected_out_mdspan);
        ggml_backend_tensor_get(out_inplace, out_result.data(), 0, out_inplace->nbytes());
        check_tensor(out_result_mdspan, expected_out_mdspan);
    }

    return 0;
}
