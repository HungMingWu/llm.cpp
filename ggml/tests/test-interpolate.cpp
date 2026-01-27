#include <cassert>
#include <cmath>
#include <cstdio>
#include <array>
#include <memory>
#include <print>
#include <vector>

import ggml;

bool check_equal(const float* result, const float* expected, int64_t n) {
    for (int i = 0; i < n; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-4) {
            std::println("result[{}] {} != {} expected[{}]", i, result[i], expected[i], i);
            return false;
        }
    }
    return true;
}

bool test_interpolate(char const* name,
    std::array<int64_t, 4> src_ne, const float* src_data,
    std::array<int32_t, 4> dst_ne, const float* expected,
    uint32_t mode) {

    ggml_context ctx;
    ggml_cgraph gf;

    // Build graph
    ggml_tensor* src = ctx.create(GGML_TYPE_F32, src_ne);
    ggml_tensor* res = ggml_interpolate(&ctx, src, dst_ne[0], dst_ne[1], dst_ne[2], dst_ne[3], mode);
    gf.build_forward_expand(res);

    // Create backend & allocate buffers
    std::unique_ptr<ggml_cpu_backend> backend = ggml_backend_cpu_init();
    backend->set_n_threads(2);
    std::unique_ptr<ggml_backend_buffer> buf = backend->alloc_tensors(&ctx);

    // Execute and compare results
    ggml_backend_tensor_set(src, src_data, 0, src->nbytes());
    backend->graph_compute(&gf);

    std::vector<float> res_values(res->nelements());
    ggml_backend_tensor_get(res, res_values.data(), 0, res->nbytes());

    bool passed = check_equal(res_values.data(), expected, res->nelements());

    std::println("ggml_interpolate({}): {}", name, passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed;
}

const float input_upscale[] = {
    0.0f, 1.0f,
    2.0f, 4.0f
};

const float expected_upscale_x2_nearest[] = {
    0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 0.0f, 1.0f, 1.0f,
    2.0f, 2.0f, 4.0f, 4.0f,
    2.0f, 2.0f, 4.0f, 4.0f
};

const float expected_upscale_x2_bilinear[] = {
    0.0f, 0.2500f, 0.7500f, 1.00f,
    0.5f, 0.8125f, 1.4375f, 1.75f,
    1.5f, 1.9375f, 2.8125f, 3.25f,
    2.0f, 2.5000f, 3.5000f, 4.00f
};

const float expected_upscale_x2_bilinear_align_corners[] = {
    0.0000f, 0.3333f, 0.6667f, 1.0000f,
    0.6667f, 1.1111f, 1.5556f, 2.0000f,
    1.3333f, 1.8889f, 2.4444f, 3.0000f,
    2.0000f, 2.6667f, 3.3333f, 4.0000f
};

const float expected_upscale_x1_5_bilinear_align_corners[] = {
    0.0f, 1.0f,
    1.0f, 2.5f,
    2.0f, 4.0f
};

const float input_downscale[] = {
    0.0f, -1.0f, -2.0f, 0.0f,
    1.0f, 2.0f , 4.0f , 4.0f,
    2.0f, 2.0f , 1.0f , 1.0f,

    1.0f, 2.0f , 3.0f , 4.0f,
    2.0f, 2.0f , 2.0f , 2.0f,
    -2.0f, 2.0f, -4.0f, 4.0f
};

const float expected_downscale_nearest[] = {
    0.0f, -2.0f,

    1.0f, 3.0f
};

const float expected_downscale_bilinear[] = {
    0.1667f, -0.3750f,  0.7500f,
    1.7917f,  1.8750f,  1.7500f,

    1.3750f,  2.3750f,  3.3750f,
   -0.5000f, -0.2500f,  2.5000f
};

const float expected_downscale_bilinear_align_corners[] = {
    0.0f , -1.5f, 0.0f,
    2.0f ,  1.5f, 1.0f,

    1.0f ,  2.5f, 4.0f,
    -2.0f, -1.0f, 4.0f
};

int main() {
    bool passed = true;

    passed &= test_interpolate("upscale_x2_nearest",
        { 2, 2, 1, 1 }, input_upscale,
        { 4, 4, 1, 1 }, expected_upscale_x2_nearest,
        GGML_SCALE_MODE_NEAREST);

    passed &= test_interpolate("upscale_x2_bilinear",
        { 2, 2, 1, 1 }, input_upscale,
        { 4, 4, 1, 1 }, expected_upscale_x2_bilinear,
        GGML_SCALE_MODE_BILINEAR);

    passed &= test_interpolate("upscale_x2_bilinear_align_corners",
        { 2, 2, 1, 1 }, input_upscale,
        { 4, 4, 1, 1 }, expected_upscale_x2_bilinear_align_corners,
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);

    passed &= test_interpolate("upscale_x1_5_bilinear_align_corners",
        { 2, 2, 1, 1 }, input_upscale,
        { 2, 3, 1, 1 }, expected_upscale_x1_5_bilinear_align_corners,
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);

    passed &= test_interpolate("downscale_nearest",
        { 4, 3, 2, 1 }, input_downscale,
        { 2, 1, 2, 1 }, expected_downscale_nearest,
        GGML_SCALE_MODE_NEAREST);

    passed &= test_interpolate("downscale_bilinear",
        { 4, 3, 2, 1 }, input_downscale,
        { 3, 2, 2, 1 }, expected_downscale_bilinear,
        GGML_SCALE_MODE_BILINEAR);

    passed &= test_interpolate("downscale_bilinear_align_corners",
        { 4, 3, 2, 1 }, input_downscale,
        { 3, 2, 2, 1 }, expected_downscale_bilinear_align_corners,
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);

    return passed ? 0 : 1;
}