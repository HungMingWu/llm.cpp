#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <print>
#include <vector>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

import ggml;
import test;

template <typename T>
std::vector<T> arange(int64_t num_elements) {
    std::vector<T> result(num_elements);
    for (int32_t i = 0; auto & ele : result) {
        if constexpr (isIntegerType_v<T>) {
            ele = i++;
        }
        else {
            ele = fromFloat32<T>(i++);
        }
    }
    return result;
}

void dup_to(ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(dst->op == GGML_OP_VIEW);
    GGML_ASSERT(src->nelements() == dst->nelements());
    dst->op = GGML_OP_DUP;
    dst->src.clear();
    dst->src.push_back(src);
}

bool can_dup(enum ggml_type src_type, enum ggml_type dst_type) {
    if (src_type == dst_type) return true;
    if (src_type == GGML_TYPE_F32) {
        return dst_type == GGML_TYPE_F16 || dst_type == GGML_TYPE_F32;
    }
    if (dst_type == GGML_TYPE_F32) {
        return src_type == GGML_TYPE_F16 || src_type == GGML_TYPE_F32;
    }

    return false;
}

template <typename T>
void check(const std::vector<T>& dst)
{
    // src_cont -> dst_cont_1
    GGML_ASSERT(dst[49] == 0);
    GGML_ASSERT(dst[50] == 20);
    GGML_ASSERT(dst[51] == 21);
    GGML_ASSERT(dst[52] == 22);
    GGML_ASSERT(dst[59] == 29);

    // src_stride -> dst_cont_2
    GGML_ASSERT(dst[60] == 3);
    GGML_ASSERT(dst[61] == 13);
    GGML_ASSERT(dst[62] == 23);
    GGML_ASSERT(dst[69] == 93);
    GGML_ASSERT(dst[70] == 0);

    // src_cont -> dst_stride_1
    GGML_ASSERT(dst[6] == 0);
    GGML_ASSERT(dst[7] == 20);
    GGML_ASSERT(dst[17] == 21);
    GGML_ASSERT(dst[27] == 22);
    GGML_ASSERT(dst[97] == 29);
    GGML_ASSERT(dst[107] == 0);

    // src_stride -> dst_stride_2
    GGML_ASSERT(dst[8] == 03);
    GGML_ASSERT(dst[18] == 13);
    GGML_ASSERT(dst[28] == 23);
    GGML_ASSERT(dst[98] == 93);
    GGML_ASSERT(dst[108] == 0);
}

template <typename src_t, typename dst_t>
void test_dup(ggml_type src_type, ggml_type dst_type)
{
    ggml_context ctx;

    ggml_tensor* src = ctx.create(src_type, 10, 11);
    ggml_tensor* dst = ctx.create(dst_type, 10, 11);

    // 2nd-row: [20, 21, ..., 29]
    ggml_tensor* src_cont = ggml_view(&ctx, src, { 10 }, {}, src->nb[1] * 2);

    // 3rd-col: [03, 13, ..., 93]
    ggml_tensor* src_stride = ggml_view(&ctx, src, { 1, 10 }, { src->nb[1] }, src->nb[0] * 3);

    ggml_tensor* dst_cont_1 = ggml_view(&ctx, dst, { 10 }, {}, dst->nb[1] * 5); // 5nd-row
    ggml_tensor* dst_cont_2 = ggml_view(&ctx, dst, { 10 }, {}, dst->nb[1] * 6); // 6rd-row

    ggml_tensor* dst_stride_1 = ggml_view(&ctx, dst, { 1, 10 }, { dst->nb[1] }, dst->nb[0] * 7); // 7th-col
    ggml_tensor* dst_stride_2 = ggml_view(&ctx, dst, { 1, 10 }, { dst->nb[1] }, dst->nb[0] * 8); // 8th-col

    ggml_cgraph gf;

    dup_to(src_cont, dst_cont_1);
    dup_to(src_stride, dst_cont_2);
    dup_to(src_cont, dst_stride_1);
    dup_to(src_stride, dst_stride_2);

    gf.build_forward_expand(dst_cont_1);
    gf.build_forward_expand(dst_cont_2);
    gf.build_forward_expand(dst_stride_1);
    gf.build_forward_expand(dst_stride_2);

    auto src_data = arange<src_t>(src->nelements());
    auto dst_data = std::vector<dst_t>(dst->nelements(), 0);
    run_graph_in_cpu1(&ctx, gf, [&](ggml_tensor* tensor) {
        if (tensor == src) {
            ggml_backend_tensor_set(tensor, src_data.data(), 0, tensor->nbytes());
        }
        else if (tensor == dst) {
            ggml_backend_tensor_set(tensor, dst_data.data(), 0, tensor->nbytes());
        }
        }, [&] {
            ggml_backend_tensor_get(dst, dst_data.data(), 0, dst->nbytes());
        });

    if constexpr (std::is_same_v<dst_t, ggml_fp16_t>) {
        std::vector<ggml_fp32_t> transform_dst;
        for (auto ele : dst_data) transform_dst.push_back(toFloat32(ele));
        check(transform_dst);
    }
    else {
        check(dst_data);
    }
}

template <typename src_t>
void test_dup(ggml_type src_type, ggml_type dst_type)
{
    switch (dst_type) {
    case GGML_TYPE_I16:
        return test_dup<src_t, ggml_i16_t>(src_type, dst_type);
    case GGML_TYPE_I32:
        return test_dup<src_t, ggml_i32_t>(src_type, dst_type);
    case GGML_TYPE_F16:
        return test_dup<src_t, ggml_fp16_t>(src_type, dst_type);
    case GGML_TYPE_F32:
        return test_dup<src_t, ggml_fp32_t>(src_type, dst_type);
    default:
        assert(false);
    }
}
void test_dup(ggml_type src_type, ggml_type dst_type)
{
    switch (src_type) {
    case GGML_TYPE_I16:
        return test_dup<ggml_i16_t>(src_type, dst_type);
    case GGML_TYPE_I32:
        return test_dup<ggml_i32_t>(src_type, dst_type);
    case GGML_TYPE_F16:
        return test_dup<ggml_fp16_t>(src_type, dst_type);
    case GGML_TYPE_F32:
        return test_dup<ggml_fp32_t>(src_type, dst_type);
    default:
        assert(false);
    }
}

int main(int argc, const char** argv) {
    enum ggml_type type[] = { GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_F16, GGML_TYPE_F32};

    for (auto src_type : type) {
        for (auto dst_type : type) {
            if (!can_dup(src_type, dst_type)) continue;
            std::println("Testing dup on {} -> {} copy", ggml_type_name(src_type), ggml_type_name(dst_type));
            test_dup(src_type, dst_type);
        }
    }

    return 0;
}
