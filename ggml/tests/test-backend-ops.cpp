#include <float.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <future>
#include <memory>
#include <print>
#include <string>
#include <random>
#include <vector>

#define GGML_ABORT(...)
#define GGML_ASSERT(...)

import ggml;

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
};

static void usage(char** argv) {
    printf("Usage: %s [mode] [-o op] [-b backend]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - grad (compare gradients from backpropagation with method of finite differences)\n");
    printf("      - perf (performance evaluation)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc)\n");
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t*)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float* a, const float* b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static std::vector<float> tensor_to_float(const ggml_tensor* t) {
    std::vector<float> tv;
    tv.reserve(t->nbytes());

    std::vector<uint8_t> buf(t->nbytes());
    ggml_backend_tensor_get(t, buf.data(), 0, t->nbytes());

    const auto* tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 / bs * t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(toFloat32(*(ggml_fp16_t*)&buf[i]));
                    }
                    else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(toFloat32(*(ggml_bf16_t*)&buf[i]));
                    }
                    else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float*)&buf[i]);
                    }
                    else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t*)&buf[i]);
                    }
                    else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t*)&buf[i]);
                    }
                    else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t*)&buf[i]);
                    }
                    else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t*)&buf[i]);
                    }
                    else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    }
                    else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float* a, const float* b, const size_t n, const std::vector<float>& expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float& ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum / nvalid;
}

static void init_tensor_uniform(ggml_tensor* tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = tensor->nelements();
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto& gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start = i * nels / n_threads;
            size_t end = (i + 1) * nels / n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto& t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    }
    else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

        // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float* im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f * (min + max)) {
                im = nullptr;
            }
        }
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency() / 2,
                std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start = i * n_blocks / n_threads;
                size_t end = (i + 1) * n_blocks / n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto& t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    }
    else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
    }
    else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = tensor->nbytes() / 2;
        ggml_backend_tensor_set(tensor, data.data(), 0 * nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1 * nbytes_half, nbytes_half);
    }
    else {
        GGML_ABORT("fatal error");
    }
}

static void init_tensor_one(ggml_tensor* tensor) {
    size_t nels = tensor->nelements();
    std::vector<float> data(nels, 1.0);
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    }
    else {
        GGML_ABORT("fatal error");
    }
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor* t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor* build_graph(ggml_context* ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual double max_maa_err() {
        return 1e-4;
    }

    virtual float grad_eps() {
        return 1e-1f;
    }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() {
        return false;
    }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() {
        return 10000;
    }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() {
        return {};
    }

    virtual void initialize_tensors(ggml_context* ctx) {
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor* t) {
        size_t size = t->nbytes();
        // add source tensors
        for (auto& src : t->src)
            size += src->nbytes();
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor*) {
        return 0;
    }

    ggml_cgraph gf;
    ggml_cgraph gb;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor*> sentinels;

    void add_sentinel(ggml_context* ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD) {
            return;
        }
        ggml_tensor* sentinel = ctx->create(GGML_TYPE_F32, { sentinel_size });
        sentinel->set_name(std::format("sent_{}", sentinels.size()));
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor* ggml_new_tensor(ggml_context* ctx, ggml_type type, std::initializer_list<int64_t> ne) {
        ggml_tensor* t = ctx->create(type, ne);
        add_sentinel(ctx);
        return t;
    }

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char* op_name) {
        mode = MODE_TEST;

        ggml_context ctx;

        // pre-graph sentinel
        add_sentinel(&ctx);

        ggml_tensor* out = build_graph(&ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        std::print("  {}({}): ", op_desc(out), vars());

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (auto& t : ctx.getTensors()) {
                if (!backend->get_device()->supports_op(t)) {
                    std::print("not supported [{}] ", backend->get_name());
                    supported = false;
                    break;
                }
            }
        }
        if (!supported) {
            std::println();
            return true;
        }

        // post-graph sentinel
        add_sentinel(&ctx);

        // allocate
        auto buf = ggml_backend_alloc_ctx_tensors(&ctx, backend1);
        if (buf == NULL) {
            std::print("failed to allocate tensors [{}] ", backend1->get_name());
            return false;
        }

        // build graph
        gf.build_forward_expand(out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor* sentinel : sentinels) {
            gf.add_node(sentinel);
        }

        // randomize tensors
        initialize_tensors(&ctx);

        // compare
        double max_err = max_nmse_err();

        auto callback = [&](ggml_tensor* t1, ggml_tensor* t2) -> bool {
            const char* bn1 = backend1->get_name();
            const char* bn2 = backend2->get_name();

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(t1->nbytes());
                std::vector<uint8_t> t2_data(t2->nbytes());
                ggml_backend_tensor_get(t1, t1_data.data(), 0, t1->nbytes());
                ggml_backend_tensor_get(t2, t2_data.data(), 0, t2->nbytes());

                if (memcmp(t1_data.data(), t2_data.data(), t1->nbytes()) != 0) {
                    std::print("sentinel mismatch: {} ", t1->name);
                    return false;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);
#if 0
            if (t1->get_name() == "out") {
                printf("Dump t1:\n");
                for (auto v : f1) printf("%2lf ", v);
                printf("\n");
                printf("Dump t2:\n");
                for (auto v : f2) printf("%2lf ", v);
                printf("\n");
            }
#endif
            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    std::print("[{}] NaN at index {} ({}={} {}={}) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    return false;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            std::print("[{}] inf sign mismatch: {}={} {}={} ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            return false;
                        }
                    }
                    else {
                        std::print("[{}] inf mismatch: {}={} {}={} ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        return false;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > max_err) {
                std::print("[{}] NMSE = {:.9f} > {:.9f} ", ggml_op_desc(t1), err, max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                return false;
            }
            return true;
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, &gf, callback);

        if (cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }

    bool eval_perf(ggml_backend_t backend, const char* op_name) {
#if 0
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        ggml_tensor* out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            ggml_free(ctx);
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            ggml_free(ctx);
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // build graph
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_backend_graph_compute(backend, gf);

        // determine number of runs
        int n_runs;
        bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu = 8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t target_flops = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_flops / op_flops(out)) + 1;
        }
        else {
            // based on memory size
            const size_t GB = 1ULL << 30;
            const size_t target_size_cpu = 8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t target_size = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_size / op_size(out)) + 1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor* t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
            };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_backend_graph_compute(backend, gf);
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000 * 1000); // run for at least 1 second

        printf("    %8d runs - %8.2f us/run - ",
            total_runs,
            (double)total_time_us / total_runs);

        if (op_flops(out) > 0) {
            double flops_per_sec = (op_flops(out) * total_runs) / (total_time_us / 1e6);
            auto format_flops = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                }
                else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                }
                else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                }
                else {
                    snprintf(buf, sizeof(buf), "%6.2f KFLOP", flops / 1e3);
                }
                return buf;
                };
            printf("%s/run - \033[1;34m%sS\033[0m",
                format_flops(op_flops(out)).c_str(),
                format_flops(flops_per_sec).c_str());

        }
        else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m",
                op_size(out) / 1024,
                total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0);
        }
        printf("\n");

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);
#endif
        return true;
    }

    bool eval_grad(ggml_backend_t backend, const char* op_name) {
        mode = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_context ctx;

        ggml_tensor* out = build_graph(&ctx);

        if ((op_name != nullptr && op_desc(out) != op_name) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            //std::print("  {}: skipping\n", op_desc(out));
            return true;
        }

        std::print("  {}({}): ", op_desc(out), vars());
        fflush(stdout);

        if (out->type != GGML_TYPE_F32) {
            std::print("not supported [{}->type != FP32]\n", out->name);
            return true;
        }

        // check if the backend supports the ops
        bool supported = true;
        bool any_params = false;
        for (auto t : ctx.getTensors()) {
            if (!backend->get_device()->supports_op(t)) {
                std::print("not supported [{}] ", backend->get_name());
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    std::print("not supported [{}->type != FP32] ", t->name);
                    supported = false;
                    break;
                }
            }
        }
        if (!any_params) {
            std::print("not supported [{}] \n", op_name);
            supported = false;
        }
        if (!supported) {
            std::println();
            return true;
        }

        int64_t ngrads = 0;
        for (auto t : ctx.getTensors()) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += t->nelements();
            }
        }
        if (ngrads > grad_nmax()) {
            printf("skipping large tensors for speed \n");
            return true;
        }

        if (!ggml_is_scalar(out)) {
            out = ggml_sum(&ctx, out);
            out->set_name("sum_of_out");
        }
        ggml_set_loss(out);

        gf.build_forward_expand(out);
        gb = gf;
        gb.build_backward_expand(&ctx, &ctx, false);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (auto t : ctx.getTensors()) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) || ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (auto t : ctx.getTensors()) {
            if (!backend->get_device()->supports_op(t)) {
                std::print("not supported [{}] ", backend->get_name());
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                std::print("not supported [{}->type != FP32] ", t->name);
                supported = false;
                break;
            }
        }
        if (!supported) {
            std::println();
            return true;
        }

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(&ctx, backend);
        if (buf == NULL) {
            std::print("failed to allocate tensors [{}] ", backend->get_name());
            return false;
        }


        initialize_tensors(&ctx); // Randomizes all tensors (including gradients).
        gb.reset();    // Sets gradients to 1 if loss, 0 otherwise.

        backend->graph_compute(&gf);
        backend->graph_compute(&gb);

        bool ok = true;
        for (auto t : ctx.getTensors()) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char* bn = backend->get_name();
            const int64_t ne = t->nelements();

            std::vector<float> ga;
            struct ggml_tensor* grad = ggml_graph_get_grad(&gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            }
            else {
                ga.resize(ne); // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) { // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    std::print("[{}] nonfinite gradient at index {} ({}={}) ", ggml_op_desc(t), i, bn, ga[i]);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne); // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t); // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu = x0[i] + 1.0f * eps; // x, index i, up
                const float xiuh = x0[i] + 0.5f * eps; // x, index i, up half
                const float xidh = x0[i] - 0.5f * eps; // x, index i, down half
                const float xid = x0[i] - 1.0f * eps; // x, index i, down

                float fu, fuh, fdh, fd; // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i * sizeof(float), sizeof(float));
                backend->graph_compute(&gf);
                ggml_backend_tensor_get(out, &fu, 0, out->nbytes());

                ggml_backend_tensor_set(t, &xid, i * sizeof(float), sizeof(float));
                backend->graph_compute(&gf);
                ggml_backend_tensor_get(out, &fd, 0, out->nbytes());

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i * sizeof(float), sizeof(float));
                    backend->graph_compute(&gf);
                    ggml_backend_tensor_get(out, &fuh, 0, out->nbytes());

                    ggml_backend_tensor_set(t, &xidh, i * sizeof(float), sizeof(float));
                    backend->graph_compute(&gf);
                    ggml_backend_tensor_get(out, &fdh, 0, out->nbytes());

                    gn[i] = (8.0 * (double)fuh + (double)fd - (8.0 * (double)fdh + (double)fu)) / (6.0 * (double)eps);
                }
                else {
                    gn[i] = (fu - fd) / (2.0f * eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, t->nbytes());
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                printf("[%s] MAA = %.9f > %.9f ", ggml_op_desc(t), err, max_maa_err());
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        if (!ok) {
            printf("compare failed ");
        }

        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");

        return false;
    }
};

template <>
struct std::formatter<ggml_type> : std::formatter<std::string> {
    auto format(const ggml_type& type, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", ggml_type_name(type));
    }
};

template <>
struct std::formatter<ggml_prec> : std::formatter<std::string> {
    auto format(const ggml_prec& prec, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", prec == GGML_PREC_F32 ? "f32" : "def");
    }
};

template <>
struct std::formatter<ggml_scale_mode> : std::formatter<std::string> {
    auto format(const ggml_scale_mode& scale, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", scale == GGML_SCALE_MODE_NEAREST ? "nearest" : "bilinear");
    }
};

template <>
struct std::formatter<ggml_op_pool> : std::formatter<std::string> {
    auto format(const ggml_op_pool& pool, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", pool == GGML_OP_POOL_AVG ? "avg" : "max");
    }
};

template <typename T, size_t N>
struct std::formatter<std::array<T, N>> : std::formatter<std::string> {
	auto format(const std::array<T, N>& x, std::format_context& ctx) const {
        auto outIt = std::format_to(ctx.out(), "[");
        for (size_t i = 0; i < N; i++) {
            if (i > 0) {
                outIt = std::format_to(outIt, ",");
            }
            outIt = std::format_to(outIt, "{}", x[i]);
        }
        return std::format_to(outIt, "]");
	}
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return std::format("type={},ne_a={},v={}", type, ne_a, v);
    }

    test_unary(ggml_unary_op op,
        ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 128, 2, 2, 2 },
        int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        const bool grad_supported = op == GGML_UNARY_OP_ABS || op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_NEG ||
            op == GGML_UNARY_OP_STEP || op == GGML_UNARY_OP_RELU || op == GGML_UNARY_OP_SILU;

        ggml_tensor* a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, { ne[0], ne[1], ne[2], ne[3]});
            if (grad_supported) {
                ggml_set_param(a);
            }
            a->set_name("a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);

            a->set_name("view_of_a");
        }
        else {
            a = ggml_new_tensor(ctx, type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
            if (grad_supported) {
                ggml_set_param(a);
            }
            a->set_name("a");
        }

        ggml_tensor* out = ggml_unary(ctx, a, op);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
            //init_tensor_one(t);
        }
    }

    float grad_eps() override {
        return 15.0f;
    }

    std::vector<float> grad_expect() override {
        if (op == GGML_UNARY_OP_ABS) {
            return { -1.0f, 1.0f };
        }
        if (op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_STEP) {
            return { 0.0f };
        }
        if (op == GGML_UNARY_OP_RELU) {
            return { 0.0f, 1.0f };
        }
        return {};
    }

};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return std::format("type={},n={},m={},r={},b={},v={}", 
            type, n, m, r, b, static_cast<int>(v));
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* in = ctx->create(type, { n, m, b });
        in->set_name("in");

        ggml_tensor* rows = ctx->create(GGML_TYPE_I32, { r, b });
        rows->set_name("rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r / 2, b, rows->nb[1], 0);
            rows->set_name("view_of_rows");
        }

        const bool grad_supported = ggml_is_matrix(in) && ggml_is_vector(*rows);
        if (grad_supported) {
            ggml_set_param(in);
            // rows is a constant input -> no gradients
        }

        ggml_tensor* out = ggml_get_rows(ctx, in, rows);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r * b);
                for (int i = 0; i < r * b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }
};

struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return std::format("pool_type={},type_input={},ne_input={},k0={},k1={},s0={},s1={},p0={},p1={}",
            pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
        ggml_type type_input = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_input = { 10, 10, 3, 1 }, // [input_width, input_height, input_channels, 1]
        int k0 = 3, int k1 = 3,
        int s0 = 1, int s1 = 1,
        int p0 = 1, int p1 = 1)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), k1(k1), s0(s0), s1(s1), p0(p0), p1(p1) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* input = ctx->create(type_input, { ne_input[0], ne_input[1], ne_input[2], ne_input[3] });
        ggml_set_param(input);
        input->set_name("input");

        ggml_tensor* out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        out->set_name("out");

        return out;
    }
};

struct test_get_rows_back : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return std::format("type={},n={},m={},r={},b={},v={}",
            type, n, m, r, b, static_cast<int>(v));
    }

    test_get_rows_back(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* in_forward = ctx->create(type, { n, m, b });
        in_forward->set_name("in_forward");

        ggml_tensor* rows = ctx->create(GGML_TYPE_I32, { r, b });
        rows->set_name("rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r / 2, b, rows->nb[1], 0);
            rows->set_name("view_of_rows");
        }

        ggml_tensor* grad = ctx->create(type, { n, r, b });
        grad->set_name("grad");

        ggml_tensor* out = ggml_get_rows_back(ctx, grad, rows, in_forward);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r * b);
                for (int i = 0; i < r * b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }
};

struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return std::format("type_input={},type_kernel={},dst_type={},ne_input={},ne_kernel={},s0={}"
            ",s1={},p0={},p1={},d0={},d1={},is_2D={}",
            type_input, type_kernel, dst_type,
            ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, static_cast<int>(is_2D));
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_input = { 10, 10, 3, 1 }, // [input_width, input_height, input_channels, 1]
        std::array<int64_t, 4> ne_kernel = { 3, 3, 3, 1 }, // [kernel_width, kernel_height, input_channels, 1]
        int s0 = 1, int s1 = 1,
        int p0 = 1, int p1 = 1,
        int d0 = 1, int d1 = 1,
        bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* input = ctx->create(type_input, { ne_input[0], ne_input[1], ne_input[2], ne_input[3] });
        ggml_set_param(input);
        input->set_name("input");

        ggml_tensor* kernel = ggml_new_tensor(ctx, type_kernel, { ne_kernel[0], ne_kernel[1], ne_kernel[2], ne_kernel[3] });
        kernel->set_name("kernel");

        ggml_tensor* out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        out->set_name("out");

        return out;
    }
};

struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0; // stride
    const int p0; // padding
    const int d0; // dilation

    std::string vars() override {
        return std::format("ne_input={},ne_kernel={},s0={},p0={},d0={}",
            ne_input, ne_kernel, s0, p0, d0);
    }

    test_conv_transpose_1d(std::array<int64_t, 4> ne_input = { 197, 32, 1, 1 }, // [input_width, input_height, input_channels, 1]
        std::array<int64_t, 4> ne_kernel = { 16, 32, 32, 1 }, // [kernel_width, kernel_height, input_channels, 1]
        int s0 = 1, int p0 = 0, int d0 = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), p0(p0), d0(d0) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* input = ctx->create(GGML_TYPE_F32, { ne_input[0], ne_input[1], ne_input[2], ne_input[3] });
        input->set_name("input");

        ggml_tensor* kernel = ctx->create(GGML_TYPE_F32, { ne_kernel[0], ne_kernel[1], ne_kernel[2], ne_kernel[3] });;
        kernel->set_name("kernel");

        ggml_tensor* out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        out->set_name("out");

        return out;
    }
};

struct test_count_equal : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_count_equal(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 4, 500, 1, 1 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* a_argmax = ggml_argmax(ctx, a);
        a_argmax->set_name("a_argmax");

        ggml_tensor* b = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        b->set_name( "b");

        ggml_tensor* b_argmax = ggml_argmax(ctx, a);
        b_argmax->set_name("b_argmax");

        ggml_tensor* out = ggml_count_equal(ctx, a_argmax, b_argmax);
        out->set_name("out");

        return out;
    }

    double max_nmse_err() override {
        return 0.0;
    }
};

struct test_argmax : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_argmax(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 100, 1, 1 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_argmax(ctx, a);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        return 0.0;
    }
};

struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return std::format("type={},ne={},nr={}", type, ne, nr);
    }

    size_t op_size(ggml_tensor* t) override {
        return t->nbytes() * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 },
        std::array<int, 4> nr = { 2, 2, 2, 2 })
        : type(type), ne(ne), nr(nr) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* target = ctx->create(type, { ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3] });
        target->set_name("target");

        ggml_tensor* src = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(src);
        src->set_name("src");

        ggml_tensor* out = ggml_repeat(ctx, src, target);
        out->set_name("out");

        return out;
    }
};

struct test_repeat_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    const bool v; // whether src is a noncontiguous view

    std::string vars() override {
        return std::format("type={},ne={},nr={},v={}", type, ne, nr, static_cast<int>(v));
    }

    size_t op_size(ggml_tensor* t) override {
        return t->nbytes() * 2;
    }

    test_repeat_back(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 8, 6, 4, 2 },
        std::array<int, 4> nr = { 2, 2, 2, 2 },
        bool v = false)
        : type(type), ne(ne), nr(nr), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* src = ctx->create(type, { ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3] });
        src->set_name("src");

        if (v) {
            GGML_ASSERT(ne[0] % 2 == 0);
            GGML_ASSERT(ne[1] % 2 == 0);
            GGML_ASSERT(ne[2] % 2 == 0);
            GGML_ASSERT(ne[3] % 2 == 0);
            GGML_ASSERT(nr[0] % 2 == 0 || nr[0] == 1);
            GGML_ASSERT(nr[1] % 2 == 0 || nr[1] == 1);
            GGML_ASSERT(nr[2] % 2 == 0 || nr[2] == 1);
            GGML_ASSERT(nr[3] % 2 == 0 || nr[3] == 1);

            const int64_t ne00 = nr[0] == 1 ? src->ne[0] : src->ne[0] / 2;
            const int64_t ne01 = nr[1] == 1 ? src->ne[1] : src->ne[1] / 2;
            const int64_t ne02 = nr[2] == 1 ? src->ne[2] : src->ne[2] / 2;
            const int64_t ne03 = nr[3] == 1 ? src->ne[3] : src->ne[3] / 2;

            src = ggml_view_4d(ctx, src, ne00, ne01, ne02, ne03, src->nb[1], src->nb[2], src->nb[3], 0);
        }

        ggml_tensor* target = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        target->set_name("target");

        ggml_tensor* out = ggml_repeat_back(ctx, src, target);
        out->set_name("out");

        return out;
    }
};

struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = std::format("type={},ne={}", type, ne);
        if (_use_permute) v += std::format(",permute={}", permute);
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 20, 1 },
        std::array<int64_t, 4> permute = { 0, 0, 0, 0 })
        : type(type), ne(ne), permute(permute),
        _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* src = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(src);
        src->set_name("src");

        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            src->set_name("src_permuted");
        }

        ggml_tensor* out = ggml_dup(ctx, src);
        out->set_name("out");

        return out;
    }
};

struct test_set : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const int dim;

    std::string vars() override {
        return std::format("type_src={},type_dst={},ne={},dim={}",
            type_src, type_dst, ne, dim);
    }

    size_t op_size(ggml_tensor* t) override {
        return t->nbytes() + t->src[0]->nbytes();
    }

    test_set(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 6, 5, 4, 3 }, int dim = 1)
        : type_src(type_src), type_dst(type_dst), ne(ne), dim(dim) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* src = ctx->create(type_src, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(src);
        src->set_name("src");

        auto ne_dst = ne;
        for (int i = 0; i < dim; ++i) {
            ne_dst[i] *= 2;
        }
        ggml_tensor* dst = ctx->create(type_dst, { ne_dst[0], ne_dst[1], ne_dst[2], ne_dst[3] });
        ggml_set_param(dst);
        dst->set_name("dst");

        size_t offset = 0;
        for (int i = 0; i < dim; ++i) {
            offset += ((ne_dst[i] - ne[i]) / 2) * dst->nb[i];
        }
        ggml_tensor* out = ggml_set(ctx, dst, src,
            // The backward pass requires setting a contiguous region:
            src->nb[1], src->nb[2], src->nb[3], offset);
        out->set_name("out");

        return out;
    }
};

struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute_src;
    const std::array<int64_t, 4> permute_dst;
    bool _src_use_permute;
    bool _dst_use_permute;

    std::string vars() override {
        return std::format("type_src={},type_dst={},ne={},permute_src={},permute_dst={}",
            type_src, type_dst, ne, permute_src, permute_dst);
    }

    double max_nmse_err() override {
        return 1e-6;
    }

    size_t op_size(ggml_tensor* t) override {
        return t->nbytes() + t->src[0]->nbytes();
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 10, 1 },
        std::array<int64_t, 4> permute_src = { 0, 0, 0, 0 },
        std::array<int64_t, 4> permute_dst = { 0, 0, 0, 0 })
        : type_src(type_src), type_dst(type_dst), ne(ne), permute_src(permute_src), permute_dst(permute_dst),
        _src_use_permute(permute_src[0] + permute_src[1] + permute_src[2] + permute_src[3] > 0),
        _dst_use_permute(permute_dst[0] + permute_dst[1] + permute_dst[2] + permute_dst[3] > 0) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* src = ggml_new_tensor(ctx, type_src, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(src);
        src->set_name("src");

        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute_src[0], permute_src[1], permute_src[2], permute_src[3]);
            src->set_name("src_permuted");
        }

        ggml_tensor* dst = ggml_new_tensor(ctx, type_dst, { src->ne[0], src->ne[1], src->ne[2], src->ne[3] });
        dst->set_name("dst");

        if (_dst_use_permute) {
            dst = ggml_permute(ctx, dst, permute_dst[0], permute_dst[1], permute_dst[2], permute_dst[3]);
            dst->set_name("dst_permuted");
        }

        ggml_tensor* out = ggml_cpy(ctx, src, dst);
        out->set_name("out");

        return out;
    }
};

struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 10, 1 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* src = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(src);
        src->set_name("src");

        src = ggml_transpose(ctx, src);
        src->set_name("src_transposed");

        ggml_tensor* out = ggml_cont(ctx, src);
        out->set_name("out");

        return out;
    }
};

struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context*, ggml_tensor*, ggml_tensor*);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return std::format("type={},ne={},nr={}", type, ne, nr);
    }

    size_t op_size(ggml_tensor* t) override {
        return t->nbytes() * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 1, 1 },
        std::array<int, 4> nr = { 1, 2, 1, 1 })
        : op(op), type(type), ne(ne), nr(nr) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3] });
        a->set_name("a");

        ggml_tensor* b = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        b->set_name("b");

        // The backward pass supports broadcasting only for GGML_ADD:
        const bool grad_supported = op == ggml_add || ggml_are_same_shape(a, b);
        if (grad_supported) {
            ggml_set_param(a);
            ggml_set_param(b);
        }

        ggml_tensor* out = op(ctx, a, b);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (ggml_tensor* t : ctx->getTensors()) {
            if (op == ggml_mul || op == ggml_div) {
                // MUL and DIV have numerical issues around zero:
                init_tensor_uniform(t, 0.9f, 1.1f);
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }

    float grad_eps() override {
        return 0.1f * (op == ggml_mul ? ne[0] * ne[1] * ne[2] * ne[3] : 1);
    }

    bool grad_precise() override {
        return op == ggml_div;
    }

    double max_maa_err() override {
        return op == ggml_add ? 1e-4 : 1e-3;
    }
};

struct test_add1 : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_add1(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param( a);
        a->set_name("a");

        ggml_tensor* b = ctx->create(type, { 1 });
        // ggml_set_param(b); // TODO: implement
        b->set_name("b");

        ggml_tensor* out = ggml_add1(ctx, a, b);
        out->set_name("out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * ne[0] * ne[1] * ne[2] * ne[3];
    }
};

struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float scale;

    std::string vars() override {
        return std::format("type={},ne={},scale={}", type, ne, scale);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 10, 10 },
        float scale = 2.0f)
        : type(type), ne(ne), scale(scale) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_scale(ctx, a, scale);
        out->set_name("out");

        return out;
    }
};

struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;

    std::string vars() override {
        return std::format("type={},ne={},v={},eps={:6f}", type, ne, static_cast<int>(v), eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 5, 4, 3 },
        bool v = false,
        float eps = 1e-6f)
        : type(type), ne(ne), v(v), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0] / 2, a->ne[1] / 2, a->ne[2] / 2, a->ne[3] / 2, a->nb[1], a->nb[2], a->nb[3], 0);
            a->set_name("view of a");
        }

        ggml_tensor* out = ggml_norm(ctx, a, eps);
        out->set_name("out");

        return out;
    }
};

struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;

    std::string vars() override {
        return std::format("type={},ne={},v={},eps={:6f}", type, ne, static_cast<int>(v), eps);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 5, 4, 3 },
        bool v = false,
        float eps = 1e-6f)
        : type(type), ne(ne), v(v), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0] / 2, a->ne[1] / 2, a->ne[2] / 2, a->ne[3] / 2, a->nb[1], a->nb[2], a->nb[3], 0);
            a->set_name("view of a");
        }

        ggml_tensor* out = ggml_rms_norm(ctx, a, eps);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (ggml_tensor* t : ctx->getTensors()) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_rms_norm_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;

    std::string vars() override {
        return std::format("type={},ne={},eps={:6f}", type, ne, eps);
    }

    test_rms_norm_back(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 5, 4, 3 },
        float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* b = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        b->set_name("b");

        ggml_tensor* out = ggml_rms_norm_back(ctx, a, b, eps);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (ggml_tensor* t : ctx->getTensors()) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }
};

struct test_l2_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_l2_norm(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 64, 320, 1 },
        float eps = 1e-12f)
        : type(type), ne(ne), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_l2_norm(ctx, a, eps);
        out->set_name("out");

        return out;
    }
};

struct test_silu_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return std::format("type={},ne={},eps={}", type, ne, eps);
    }

    test_silu_back(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 5, 4, 3 },
        float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* grad = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        grad->set_name("grad");

        ggml_tensor* out = ggml_silu_back(ctx, a, grad);
        out->set_name("out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_ssm_conv : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return std::format("type={},ne_a={},ne_b={}", type, ne_a, ne_b);
    }

    test_ssm_conv(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 10, 10, 10, 1 },
        std::array<int64_t, 4> ne_b = { 3, 3, 1, 1 })
        : type(type), ne_a(ne_a), ne_b(ne_b) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
        ggml_tensor* b = ctx->create(type, { ne_b[0], ne_b[1], ne_b[2], ne_b[3] });
        ggml_tensor* out = ggml_ssm_conv(ctx, a, b);
        return out;
    }
};

struct test_ssm_scan : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t d_inner;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return std::format("type={},d_state={},d_inner={},n_seq_tokens={},n_seqs={}",
            type, d_state, d_inner, n_seq_tokens, n_seqs);
    }

    test_ssm_scan(ggml_type type = GGML_TYPE_F32,
        int64_t d_state = 32, int64_t d_inner = 32, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), d_state(d_state), d_inner(d_inner), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* s = ctx->create(type, { d_state, d_inner, n_seqs, 1 });
        ggml_tensor* x = ctx->create(type, { d_inner, n_seq_tokens, n_seqs, 1 });
        ggml_tensor* dt = ctx->create(type, { d_inner, n_seq_tokens, n_seqs, 1 });
        ggml_tensor* A = ctx->create(type, { d_state, d_inner, 1, 1 });
        ggml_tensor* B = ctx->create(type, { d_state, n_seq_tokens, n_seqs, 1 });
        ggml_tensor* C = ctx->create(type, { d_state, n_seq_tokens, n_seqs, 1 });
        ggml_tensor* out = ggml_ssm_scan(ctx, s, x, dt, A, B, C);
        return out;
    }
};

struct test_rwkv_wkv6 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return std::format("type={},head_count={},head_size={},n_seq_tokens={},n_seqs={}",
            type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv6(ggml_type type = GGML_TYPE_F32,
        int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor* r = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* k = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* v = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* tf = ctx->create(type, { head_size, head_count });
        ggml_tensor* td = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* s = ctx->create(type, { head_size* head_size* head_count, n_seqs });
        ggml_tensor* out = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, s);
        return out;
    }
};

struct test_rwkv_wkv7 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return std::format("type={},head_count={},head_size={},n_seq_tokens={},n_seqs={}",
            type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv7(ggml_type type = GGML_TYPE_F32,
        int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor* r = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* w = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* k = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* v = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* a = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* b = ctx->create(type, { head_size, head_count, n_tokens });
        // Outputs may become NaN with long seqlen without these normalization
        a = ggml_l2_norm(ctx, a, 1e-7F);
        b = ggml_l2_norm(ctx, b, 1e-7F);
        ggml_tensor* s = ctx->create(type, { head_size* head_size* head_count, n_seqs });
        ggml_tensor* out = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, s);
        return out;
    }
};

struct test_gla : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return std::format("type={},head_count={},head_size={},n_seq_tokens={},n_seqs={}",
            type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_gla(ggml_type type = GGML_TYPE_F32,
        int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor* q = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* k = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* v = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* g = ctx->create(type, { head_size, head_count, n_tokens });
        ggml_tensor* s = ctx->create(type, { head_size * head_size * head_count, n_seqs });
        ggml_tensor* out = ggml_gated_linear_attn(ctx, k, v, q, g, s, pow(head_size, -0.5));
        return out;
    }
};

struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions
    const bool v; // whether a is a non-contiguous view

    std::string vars() override {
        return std::format("type_a={},type_b={},m={},n={},k={},bs={},nr={},per={},v={}",
            type_a, type_b, m, n, k, bs, nr, per, static_cast<int>(v));
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    int64_t grad_nmax() override {
        return 20000;
    }

    uint64_t op_flops(ggml_tensor*) override {
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
        int64_t m = 32, int64_t n = 32, int64_t k = 32,
        std::array<int64_t, 2> bs = { 10, 10 },
        std::array<int64_t, 2> nr = { 2, 2 },
        std::array<int64_t, 4> per = { 0, 1, 2, 3 },
        bool v = false)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor* a;
        ggml_tensor* b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(!v); // not handled
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = { k, m, bs[0],       bs[1] };
            const int64_t ne_b[4] = { k, n, bs[0] * nr[0], bs[1] * nr[1] };

            a = ctx->create(type_a, { ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]] });
            b = ctx->create(type_b, { ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]] });
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }
            a->set_name("a");
            b->set_name("b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            a->set_name("a_permuted");
            b->set_name("b_permuted");
        }
        else {

            if (v) {
                a = ctx->create(type_a, { k * 2, m, bs[0], bs[1] });
                a = ggml_view_4d(ctx, a, k, m, bs[0], bs[1], a->nb[1], a->nb[2], a->nb[3], 0);
            }
            else {
                a = ctx->create(type_a, { k, m, bs[0], bs[1] });
            }
            b = ctx->create(type_b, { k, n, bs[0] * nr[0], bs[1] * nr[1] });
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }
            a->set_name("a");
            b->set_name("b");
        }

        ggml_tensor* out = ggml_mul_mat(ctx, a, b);
        out->set_name("out");

        return out;
    }
};

struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // brodcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;

    std::string vars() override {
        return std::format("type_a={},type_b={},n_mats={},n_used={},b={},m={},n={},k={}",
            type_a, type_b, n_mats, n_used, static_cast<int>(b), m, n, k);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor*) override {
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
        int n_mats = 8, int n_used = 2, bool b = false,
        int64_t m = 32, int64_t n = 32, int64_t k = 32)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
        m(m), n(n), k(k) {
        GGML_ASSERT(n_used <= n_mats);
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor* as = ctx->create(type_a, { k, m, n_mats });
        as->set_name("as");

        ggml_tensor* ids = ctx->create(GGML_TYPE_I32, { n_mats, n });
        ids->set_name("ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
            ids->set_name("view_of_ids");
        }

        ggml_tensor* b = ctx->create(type_b, { k, this->b ? 1 : n_used, n });
        b->set_name("b");

        ggml_tensor* out = ggml_mul_mat_id(ctx, as, b, ids);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (auto  t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }
};

struct test_out_prod : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4
    const bool trans_b;

    std::string vars() override {
        return std::format("type_a={},type_b={},m={},n={},k={},bs={},nr={},trans_b={}",
            type_a, type_b, m, n, k, bs, nr, static_cast<int>(trans_b));
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_out_prod(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
        int64_t m = 32, int64_t n = 32, int64_t k = 32,
        std::array<int64_t, 2> bs = { 10, 10 },
        std::array<int64_t, 2> nr = { 2, 2 },
        bool trans_b = false)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), trans_b(trans_b) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type_a, { m, k, bs[0], bs[1] });
        a->set_name("a");

        ggml_tensor* b;
        if (trans_b) {
            b = ctx->create(type_b, { k, n, bs[0] * nr[0], bs[1] * nr[1] });
            b = ggml_transpose(ctx, b);
        }
        else {
            b = ctx->create(type_b, { n, k, bs[0] * nr[0], bs[1] * nr[1] });
        }
        b->set_name("b");

        ggml_tensor* out = ggml_out_prod(ctx, a, b);
        out->set_name("out");

        return out;
    }
};

struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_sqr(ctx, a);
        out->set_name("out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * 0.25f * ne[0] * ne[1] * ne[2] * ne[3]; // 10% of expected value of sum.
    }
};

struct test_sqrt : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_sqrt(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 3, 3, 2 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_sqrt(ctx, a);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        // fill with positive values
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t, 50.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 20.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_log : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_log(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_log(ctx, a);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            // log(1) == 0, cluster values there to keep the sum low for better precision in the backward pass:
            init_tensor_uniform(t, 0.9f, 1.1f);
        }
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_sin : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_sin(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 2, 2, 2 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_sin(ctx, a);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_cos : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_cos(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 2, 2, 2 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_cos(ctx, a);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return std::format("type={},ne={},min={:6f},max={:6f}", type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 },
        float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_clamp(ctx, a, min, max);
        out->set_name("out");

        return out;
    }

    float grad_eps() override {
        return 1e-2f;
    }

    std::vector<float> grad_expect() override {
        return { 0.0f, 1.0f };
    }
};

struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return std::format("type={},ne={},n_past={}", type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 10, 3, 2 },
        int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_diag_mask_inf(ctx, a, n_past);
        out->set_name("out");

        return out;
    }
};

struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const ggml_type m_prec;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return std::format("type={},ne={},mask={},m_prec={},scale={:6f},max_bias={:6f}",
            type, ne, static_cast<int>(mask), m_prec, scale, max_bias);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override {
        return 1e-6;
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 },
        bool mask = false,
        ggml_type m_prec = GGML_TYPE_F32,
        float scale = 1.0f,
        float max_bias = 0.0f)
        : type(type), ne(ne), mask(mask), m_prec(m_prec), scale(scale), max_bias(max_bias) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* mask = nullptr;
        if (this->mask) {
            mask = ctx->create(m_prec, { ne[0], ne[1] });
            mask->set_name("mask");
        }

        ggml_tensor* out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        out->set_name("out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_soft_max_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return std::format("type={},ne={},scale={:6f},max_bias={:6f}", type, ne, scale, max_bias);
    }

    test_soft_max_back(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 },
        float scale = 1.0f,
        float max_bias = 0.0f)
        : type(type), ne(ne), scale(scale), max_bias(max_bias) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* b = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        b->set_name("b");

        ggml_tensor* out = ggml_soft_max_ext_back(ctx, a, b, scale, max_bias);
        out->set_name("out");

        return out;
    }
};

struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)
    bool forward;

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return std::format("type={},ne_a={},n_dims={},mode={},n_ctx={},fs={:6f},ef={:6f},af={:6f},ff={},v={}",
            type, ne_a, n_dims, mode, n_ctx, fs, ef, af, static_cast<int>(ff), static_cast<int>(v));
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 10, 5, 3, 1 },
        int n_dims = 10, int mode = 0, int n_ctx = 512, float fs = 1.0f,
        float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0, bool forward = true)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v), forward(forward) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
            if (forward) {
                ggml_set_param(a);
            }
            a->set_name("a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            a->set_name("view_of_a");
        }
        else {
            a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
            if (forward) {
                ggml_set_param(a);
            }
            a->set_name("a");
        }

        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        ggml_tensor* pos;
        if (is_mrope || is_vision) {
            pos = ctx->create(GGML_TYPE_I32, { ne_a[2] * 4 });
        }
        else {
            pos = ctx->create(GGML_TYPE_I32, { ne_a[2] });
        }
        pos->set_name("pos");

        ggml_tensor* freq = nullptr;
        if (ff) {
            freq = ctx->create(GGML_TYPE_F32, { n_dims / 2 });
            freq->set_name("freq");
        }

        ggml_tensor* out;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims / 4 > 0);
                int rope_sections[4] = { n_dims / 4, n_dims / 4, 0, 0 }; // Vision-RoPE only use first two dimension for image (x, y) coordinate
                if (forward) {
                    out = ggml_rope_multi(ctx, a, pos, freq, n_dims / 2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
                else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims / 2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            }
            else {
                GGML_ASSERT(n_dims / 3 > 0);
                int rope_sections[4] = { n_dims / 3, n_dims / 3, n_dims / 3, 0 };
                if (forward) {
                    out = ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
                else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            }
        }
        else {
            if (forward) {
                out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
            else {
                out = ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
        }
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                const int num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, num_pos_ids * sizeof(int));
            }
            else {
                if (t->ne[0] == n_dims / 2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                }
                else {
                    init_tensor_uniform(t);
                }
            }
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int64_t ne_b_d;
    const int dim;
    const int v; // view (1 << 0: non-cont a, 1 << 1: non-cont b)

    std::string vars() override {
        return std::format("type={},ne_a={},ne_b_d={},dim={},v={}",
            type, ne_a, ne_b_d, dim, v);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 10, 5, 5, 5 },
        int64_t ne_b_d = 5,
        int dim = 2, int v = 0)
        : type(type), ne_a(ne_a), ne_b_d(ne_b_d), dim(dim), v(v) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor* a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
            a->set_name("a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            a->set_name("view_of_a");
        }
        else {
            a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
            a->set_name("a");
        }
        ggml_tensor* b;
        if (v & 2) {
            auto ne = ne_b; ne[0] *= 3; ne[1] *= 2; ne[2] *= 4;
            b = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
            b->set_name("b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            b->set_name("view_of_b");
        }
        else {
            b = ctx->create(type, { ne_b[0], ne_b[1], ne_b[2], ne_b[3] });
            b->set_name("b");
        }

        ggml_tensor* out = ggml_concat(ctx, a, b, dim);
        out->set_name("out");

        return out;
    }
};

struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return std::format("type={},ne={},order={}",
            type, ne, static_cast<int>(order));
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 16, 10, 10, 10 },
        ggml_sort_order order = GGML_SORT_ORDER_ASC)
        : type(type), ne(ne), order(order) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_argsort(ctx, a, order);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(t->nelements());
                for (int i = 0; i < t->nelements(); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0] * ne[1] * ne[2] * ne[3] * sizeof(int));
            }
            else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            }
            else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

struct test_sum : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_sum(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_sum(ctx, a);
        out->set_name("out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * sqrtf(ne[0] * ne[1] * ne[2] * ne[3]);
    }
};

struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_sum_rows(ctx, a);
        out->set_name("out");

        return out;
    }
};

struct test_mean : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_mean(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* out = ggml_mean(ctx, a);
        out->set_name("out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * ne[0] * ne[1] * ne[2] * ne[3];
    }
};

struct test_upscale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t scale_factor;
    const bool transpose;
    const ggml_scale_mode mode;

    std::string vars() override {
        return std::format("type={},ne={},scale_factor={},mode={},transpose={}",
            type, ne, scale_factor, mode, static_cast<int>(transpose));
    }

    test_upscale(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 512, 512, 3, 1 },
        int32_t scale_factor = 2, ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST, bool transpose = false)
        : type(type), ne(ne), scale_factor(scale_factor), mode(mode), transpose(transpose) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        if (transpose) {
            a = ggml_transpose(ctx, a);
            a->set_name("a_transposed");
        }

        ggml_tensor* out = ggml_upscale(ctx, a, scale_factor, mode);
        out->set_name("out");

        return out;
    }
};

struct test_upscale_ext : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;
    const ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST;

    std::string vars() override {
        return std::format("type={},ne={},ne_tgt={},mode={}", type, ne, ne_tgt, mode);
    }

    test_upscale_ext(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 2, 5,  7, 11 },
        std::array<int64_t, 4> ne_tgt = { 5, 7, 11, 13 },
        ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST)
        : type(type), ne(ne), ne_tgt(ne_tgt), mode(mode) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_upscale_ext(ctx, a, ne_tgt[0], ne_tgt[1], ne_tgt[2], ne_tgt[3], mode);
        out->set_name("out");

        return out;
    }
};

struct test_group_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t num_groups;
    const float eps;

    std::string vars() override {
        return std::format("type={},ne={},num_groups={},eps={:6f}", type, ne, num_groups, eps);
    }

    test_group_norm(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 64, 64, 320, 1 },
        int32_t num_groups = 32,
        float eps = 1e-6f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_group_norm(ctx, a, num_groups, eps);
        out->set_name("out");

        return out;
    }
};

struct test_acc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return std::format("type={},ne_a={},ne_b={}",
            type, ne_a, ne_b);
    }

    test_acc(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 256, 17, 1, 1 },
        std::array<int64_t, 4> ne_b = { 256, 16, 1, 1 })
        : type(type), ne_a(ne_a), ne_b(ne_b) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
        ggml_set_param(a);
        a->set_name("a");

        ggml_tensor* b = ctx->create(type, { ne_b[0], ne_b[1], ne_b[2], ne_b[3] });
        ggml_set_param(b);
        b->set_name("b");

        ggml_tensor* out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], b->nb[1]);
        out->set_name("out");

        return out;
    }
};

struct test_pad : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return std::format("type={},ne_a={},pad_0={},pad_1={}",
            type, ne_a, pad_0, pad_1);
    }

    test_pad(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 512, 512, 1, 1 },
        int pad_0 = 1, int pad_1 = 1)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        out->set_name("out");

        return out;
    }
};

struct test_pad_reflect_1d : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return std::format("type={},ne_a={},pad_0={},pad_1={}",
            type, ne_a, pad_0, pad_1);
    }

    test_pad_reflect_1d(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 512, 34, 2, 1 },
        int pad_0 = 10, int pad_1 = 9)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, {ne_a[0], ne_a[1]});
        a->set_name("a");

        ggml_tensor* out = ggml_pad_reflect_1d(ctx, a, pad_0, pad_1);
        out->set_name("out");

        return out;
    }
};

struct test_arange : public test_case {
    const ggml_type type;
    const float start;
    const float stop;
    const float step;

    std::string vars() override {
        return std::format("type={},start={:6f},stop={:6f},step={:6f}", type, start, stop, step);
    }

    test_arange(ggml_type type = GGML_TYPE_F32,
        float start = 0.f, float stop = 10.f, float step = 1.f)
        : type(type), start(start), stop(stop), step(step) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* out = ggml_arange(ctx, start, stop, step);
        out->set_name("out");

        return out;
    }
};

struct test_timestep_embedding : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int dim;
    const int max_period;

    std::string vars() override {
        return std::format("type={},ne_a={},dim={},max_period={}", type, 
            ne_a, dim, max_period);
    }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 2, 1, 1, 1 },
        int dim = 320, int max_period = 10000)
        : type(type), ne_a(ne_a), dim(dim), max_period(max_period) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_timestep_embedding(ctx, a, dim, max_period);
        out->set_name("out");

        return out;
    }
};

struct test_leaky_relu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const float negative_slope;

    std::string vars() override {
        return std::format("type={},ne_a={},negative_slope={:6f}", type, ne_a, negative_slope);
    }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne_a = { 10, 5, 4, 3 },
        float negative_slope = 0.1f)
        : type(type), ne_a(ne_a), negative_slope(negative_slope) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne_a[0], ne_a[1], ne_a[2], ne_a[3] });
        a->set_name("a");

        ggml_tensor* out = ggml_leaky_relu(ctx, a, negative_slope, true);
        out->set_name("out");

        return out;
    }
};

struct test_flash_attn_ext : public test_case {
    const int64_t hsk; // K head size
    const int64_t hsv; // V head size
    const int64_t nh; // num heads
    const int64_t nr; // repeat in Q, tests for grouped-query attention
    const int64_t kv; // kv size
    const int64_t nb; // batch size

    const bool mask; // use mask

    const float max_bias; // ALiBi
    const float logit_softcap; // Gemma 2

    const ggml_prec prec;
    const ggml_type type_KV;
    std::array<int32_t, 4> permute;

    std::string vars() override {
        return std::format("hsk={},hsv={},nh={},nr={},kv={},nb={},mask={},max_bias={:6f},"
            "logit_softcap={:6f},prec={},type_KV={},permute={}", hsk, hsv,
            nh, nr, kv, nb, static_cast<int>(mask), max_bias, logit_softcap, prec, type_KV, permute);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor*) override {
        // Just counting matmul costs:
        // Q*K^T is nb x hsk x kv, P*V is nb x kv x hsv, per head
        return 2 * nh * nr * nb * (hsk + hsv) * kv;
    }

    test_flash_attn_ext(int64_t hsk = 128, int64_t hsv = 128, int64_t nh = 32, int64_t nr = 1, int64_t kv = 96, int64_t nb = 8,
        bool mask = true, float max_bias = 0.0f, float logit_softcap = 0.0f, ggml_prec prec = GGML_PREC_F32,
        ggml_type type_KV = GGML_TYPE_F16, std::array<int32_t, 4> permute = { 0, 1, 2, 3 })
        : hsk(hsk), hsv(hsv), nh(nh), nr(nr), kv(kv), nb(nb), mask(mask), max_bias(max_bias), logit_softcap(logit_softcap), prec(prec), type_KV(type_KV), permute(permute) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_KV));
        const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_KV));

        auto const& create_permuted = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) -> ggml_tensor* {
            int64_t ne[4] = { ne0, ne1, ne2, ne3 };
            int64_t ne_perm[4];
            for (int i = 0; i < 4; ++i) {
                ne_perm[permute[i]] = ne[i];
            }
            ggml_tensor* t = ctx->create(type, { ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3] });
            if (permute != std::array<int32_t, 4>{0, 1, 2, 3}) {
                t = ggml_permute(ctx, t, permute[0], permute[1], permute[2], permute[3]);
            }
            return t;
        };

        ggml_tensor* q = create_permuted(GGML_TYPE_F32, hsk_padded, nb, nh * nr, 1);
        q->set_name("q");

        ggml_tensor* k = create_permuted(type_KV, hsk_padded, kv, nh, 1);
        k->set_name("k");

        ggml_tensor* v = create_permuted(type_KV, hsv_padded, kv, nh, 1);
        v->set_name("v");

        ggml_tensor* m = nullptr;
        if (mask) {
            m = ctx->create(GGML_TYPE_F16, { kv, (int64_t)GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1 });
            m->set_name("m");
        }

        ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf(hsk), max_bias, logit_softcap);
        ggml_flash_attn_ext_set_prec(out, prec);
        out->set_name("out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_cross_entropy_loss : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_cross_entropy_loss(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* logits = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(logits);
        logits->set_name("logits");

        ggml_tensor* labels = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        // The labels are assumed to be constant -> no gradients.
        labels->set_name("labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        labels->set_name("labels_normalized");

        ggml_tensor* out = ggml_cross_entropy_loss(ctx, logits, labels);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        // For larger abs. diffs between logits softmax is more linear, therefore more precise num. gradients.
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_cross_entropy_loss_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_cross_entropy_loss_back(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* grad = ctx->create(GGML_TYPE_F32, { 1 });
        grad->set_name("grad");

        ggml_tensor* logits = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        logits->set_name("logits");

        ggml_tensor* labels = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        labels->set_name("labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        labels->set_name("labels_normalized");

        ggml_tensor* out = ggml_cross_entropy_loss_back(ctx, grad, logits, labels);
        out->set_name("out");

        return out;
    }
};

struct test_opt_step_adamw : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return std::format("type={},ne={}", type, ne);
    }

    test_opt_step_adamw(ggml_type type = GGML_TYPE_F32,
        std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        ggml_tensor* a = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        ggml_set_param(a); // Despite tensor a having gradients the output tensor will not.
        a->set_name("a");

        ggml_tensor* grad = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        grad->set_name("grad");

        ggml_tensor* grad_m = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        grad_m->set_name("grad_m");

        ggml_tensor* grad_v = ctx->create(type, { ne[0], ne[1], ne[2], ne[3] });
        grad_v->set_name("grad_v");

        ggml_tensor* adamw_params = ctx->create(GGML_TYPE_F32, { 7 });
        adamw_params->set_name("adamw_params");

        ggml_tensor* out = ggml_opt_step_adamw(ctx, a, grad, grad_m, grad_v, adamw_params);
        out->set_name("out");

        return out;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            init_tensor_uniform(t, 0.0f, 1.0f); // grad_v and adamw_params need non-negative values.
        }
    }

    bool grad_precise() override {
        return true;
    }
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t n_rot;
    uint32_t n_embd_head; // dimension of values (d_v)
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx = 512; // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv = 32; // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;  // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

protected:
    test_llm(llama_hparams hp)
        : hp(std::move(hp)) {
    }

public:
    ggml_tensor* llm_build_norm(
        ggml_context* ctx,
        ggml_tensor* cur,
        ggml_tensor* mw,
        ggml_tensor* mb,
        llm_norm_type   type) {
        switch (type) {
        case LLM_NORM:     cur = ggml_norm(ctx, cur, hp.f_norm_eps); break;
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps); break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(
        ggml_context* ctx,
        ggml_tensor* k_l,
        ggml_tensor* v_l,
        ggml_tensor* k_cur,
        ggml_tensor* v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        ggml_tensor* v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        ggml_tensor* k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens * hp.n_embd_gqa(),
            (ggml_row_size(k_l->type, hp.n_embd_gqa())) * hp.kv_head);

        ggml_tensor* v_cache_view = ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(),
            (hp.n_ctx) * ggml_element_size(v_l),
            (hp.kv_head) * ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur, k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    ggml_tensor* llm_build_kqv(
        ggml_context* ctx,
        ggml_tensor* k_l,
        ggml_tensor* v_l,
        ggml_tensor* q_cur,
        ggml_tensor* kq_mask,
        float     kq_scale) {
        ggml_tensor* q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        ggml_tensor* k =
            ggml_view_3d(ctx, k_l,
                hp.n_embd_head, hp.n_kv, hp.n_head_kv,
                ggml_row_size(k_l->type, hp.n_embd_gqa()),
                ggml_row_size(k_l->type, hp.n_embd_head),
                0);

        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        ggml_tensor* v =
            ggml_view_3d(ctx, v_l,
                hp.n_kv, hp.n_embd_head, hp.n_head_kv,
                ggml_element_size(v_l) * hp.n_ctx,
                ggml_element_size(v_l) * hp.n_ctx * hp.n_embd_head,
                0);

        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);

        ggml_tensor* kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        ggml_tensor* cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head * hp.n_head, hp.n_tokens);

        ggml_tensor* wo = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_embd });
        cur = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context* ctx) override {
        for (auto t : ctx->getTensors()) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            }
            else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor*) override {
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return std::format("n_tokens={}", n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_llama(int n_tokens = 1)
        : test_llm({
        /*n_vocab        =*/ 32000,
        /*n_embd         =*/ 3200,
        /*n_head         =*/ 32,
        /*n_head_kv      =*/ 32,
        /*n_rot          =*/ 100,
        /*n_embd_head    =*/ 100,
        /*n_ff           =*/ 8640,
        /*f_norm_eps     =*/ 0.f,
        /*f_norm_rms_eps =*/ 1e-5f,
        /*n_tokens       =*/ n_tokens,
            }) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = ctx->create(GGML_TYPE_F32, { hp.n_embd, hp.n_tokens });

        // inp_pos - contains the positions
        ggml_tensor* inp_pos = ctx->create(GGML_TYPE_I32, { hp.n_tokens });

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        ggml_tensor* KQ_mask = ctx->create(GGML_TYPE_F16, { hp.n_kv, hp.n_tokens, 1 });

        ggml_tensor* k_l = ctx->create(GGML_TYPE_F16, { 1638400 });
        ggml_tensor* v_l = ctx->create(GGML_TYPE_F16, { 1638400 });

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            ggml_tensor* attn_norm = ctx->create(GGML_TYPE_F32, { hp.n_embd });
            cur = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor* wq = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_embd });
                ggml_tensor* wk = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_embd_gqa() });
                ggml_tensor* wv = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_embd_gqa() });

                // compute Q and K and RoPE them
                ggml_tensor* Qcur = ggml_mul_mat(ctx, wq, cur);
                ggml_tensor* Kcur = ggml_mul_mat(ctx, wk, cur);
                ggml_tensor* Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f / sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor* ffn_norm = ctx->create(GGML_TYPE_F32, { hp.n_embd });
            cur = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor* ffn_gate = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_ff });
            ggml_tensor* ffn_down = ctx->create(GGML_TYPE_Q4_0, { hp.n_ff, hp.n_embd });
            ggml_tensor* ffn_up = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_ff });
            struct ggml_tensor* tmp = ggml_mul_mat(ctx, ffn_up, cur);
            cur = ggml_mul_mat(ctx, ffn_gate, cur);
            cur = ggml_silu(ctx, cur);
            cur = ggml_mul(ctx, cur, tmp);
            cur = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor* output_norm = ctx->create(GGML_TYPE_F32, { hp.n_embd });
        cur = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor* output = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_vocab });
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

struct test_falcon : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor*) override {
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return std::format("n_tokens={}", n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_falcon(int n_tokens = 1)
        : test_llm({
        /*n_vocab        =*/ 32000,
        /*n_embd         =*/ 3200,
        /*n_head         =*/ 50,
        /*n_head_kv      =*/ 1,
        /*n_rot          =*/ 64,
        /*n_embd_head    =*/ 64,
        /*n_ff           =*/ 8640,
        /*f_norm_eps     =*/ 1e-5f,
        /*f_norm_rms_eps =*/ 0.f,
        /*n_tokens       =*/ n_tokens,
            }) {
    }

    ggml_tensor* build_graph(ggml_context* ctx) override {
        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = ctx->create(GGML_TYPE_F32, { hp.n_embd, hp.n_tokens });

        // inp_pos - contains the positions
        ggml_tensor* inp_pos = ctx->create(GGML_TYPE_I32, { hp.n_tokens });

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = ctx->create(GGML_TYPE_F16, { hp.n_kv, hp.n_tokens, 1 });

        ggml_tensor* k_l = ctx->create(GGML_TYPE_F16, { 1638400 });
        ggml_tensor* v_l = ctx->create(GGML_TYPE_F16, { 1638400 });

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor* attn_norm_w = ctx->create(GGML_TYPE_F32, { hp.n_embd });
            ggml_tensor* attn_norm_b = ctx->create(GGML_TYPE_F32, { hp.n_embd });
            ggml_tensor* attn_norm = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor* wqkv = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_embd + 2 * hp.n_embd_gqa() });

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor* Qcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd, hp.n_tokens, cur->nb[1], 0 * sizeof(float) * (hp.n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1 * sizeof(float) * (hp.n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1 * sizeof(float) * (hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head, hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f / sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor* ffn_inp = cur;

            // feed forward
            {
                ggml_tensor* ffn_up = ctx->create(GGML_TYPE_Q4_0, { hp.n_embd, hp.n_ff });
                ggml_tensor* ffn_down = ctx->create(GGML_TYPE_Q4_0, { hp.n_ff, hp.n_embd });
                cur = attn_norm;
                cur = ggml_mul_mat(ctx, ffn_up, cur);
                cur = ggml_gelu(ctx, cur);
                cur = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor* output_norm = ctx->create(GGML_TYPE_F32, { hp.n_embd });
        ggml_tensor* output_norm_b = ctx->create(GGML_TYPE_F32, { hp.n_embd });
        cur = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor* output = ctx->create(GGML_TYPE_Q8_0, { hp.n_embd, hp.n_vocab });
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################
static const ggml_type all_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16,
    GGML_TYPE_Q8_0, // for I8MM tests
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, // for I8MM tests
    GGML_TYPE_Q4_K,
    GGML_TYPE_IQ2_XXS
};

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // unary ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
                test_cases.emplace_back(new test_unary((ggml_unary_op)op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_unary((ggml_unary_op)op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, v));
            }
        }
    }
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, v));
        }
    }

    test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows_back(type, 256, 5, 4, 1, v));
        }
    }
    for (bool v : {false, true}) {
        test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_I32, 256, 5, 4, 1, v));
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 3}) {
                for (int k1 : {1, 3}) {
                    for (int s0 : {1, 2}) {
                        for (int s1 : {1, 2}) {
                            for (int p0 : {0, 1}) {
                                for (int p1 : {0, 1}) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, { 10, 10, 3, 1 }, k0, k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // im2col 1D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, { 3000, 128, 1, 1 }, { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, { 3000, 128, 1, 1 }, { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 3000, 128, 1, 1 }, { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    for (int s0 : {1, 3}) {
        for (int p0 : {0, 3}) {
            for (int d0 : {1, 3}) {
                test_cases.emplace_back(new test_im2col(
                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, { 20, 2, 2, 1 }, { 3, 2, 2, 1 },
                    s0, 0, p0, 0, d0, 0, false));
            }
        }
    }

    // im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : {1, 3}) {
        for (int s1 : {1, 3}) {
            for (int p0 : {0, 3}) {
                for (int p1 : {0, 3}) {
                    for (int d0 : {1, 3}) {
                        for (int d1 : {1, 3}) {
                            test_cases.emplace_back(new test_im2col(
                                GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, { 20, 20, 2, 2 }, { 3, 3, 2, 2 },
                                s0, s1, p0, p1, d0, d1, true));
                        }
                    }
                }
            }
        }
    }

    // extra tests for im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 32 }, { 3, 3, 1, 32 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 32 }, { 3, 3, 2, 32 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 1024 }, { 3, 3, 1, 1024 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 1024 }, { 3, 3, 2, 1024 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 2048 }, { 3, 3, 1, 2048 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 2048 }, { 3, 3, 2, 2048 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 2560 }, { 3, 3, 1, 2560 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 2560 }, { 3, 3, 2, 2560 }, 1, 1, 1, 1, 1, 1, true));

    // sycl backend will limit task global_range < MAX_INT
    // test cases for 2D im2col with large input W and H (occurs in stable-diffusion)
    // however these cases need to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // these cases are verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 2,3,2,1 }, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 2,3,2,1 }, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 2,3,2,1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 3,2,2,1 }, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 3,2,2,1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3,2,1,1 }, { 3,1,2,1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 2,1,1,1 }, { 3,1,1,1 }, 1, 0, 1));

    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, { 4,  500, 1, 1 }));
    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, { 4, 5000, 1, 1 }));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 32,    1, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 100,  10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 1024, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 1024, 12, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 2000, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 5438,  3, 1, 1 }));

    for (int ne3 : {1, 3}) { // CUDA backward pass only supports ne3 == 1
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 2, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 2, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 1, 2 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, { 10, 5, 4, ne3 }, { 1, 1, 1, 2 }));
    }

    for (bool view : {false, true}) {
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 2, 1, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 2, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 2, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 1, 2 }, view));
    }

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, { 10, 10, 5, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, { 10, 10, 5, 1 }, { 0, 2, 1, 3 })); // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, { 10, 10, 5, 1 }, { 1, 0, 2, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, { 10, 10, 5, 1 }, { 1, 0, 2, 3 })); // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, { 10,  8, 3, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, { 10,  8, 3, 1 }, { 1, 2, 0, 3 }));

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_F32, GGML_TYPE_F32, { 6, 5, 4, 3 }, dim));
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_I32, GGML_TYPE_I32, { 6, 5, 4, 3 }, dim));
    }

    // same-type copy
    for (ggml_type type : all_types) {
        const int64_t nk = ggml_blck_size(type);

        for (int k = 1; k < 4; ++k) {
            test_cases.emplace_back(new test_cpy(type, type, { k * nk, 2, 3, 4 }));
            test_cases.emplace_back(new test_cpy(type, type, { k * nk, 2, 3, 4 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_cpy(type, type, { k * nk, 2, 3, 4 }, { 0, 3, 1, 2 }, { 0, 2, 1, 3 }));
        }
    }

    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : all_types) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 4, 4, 4 }));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 0, 2, 1, 3 })); // cpy by rows
        }
    }
    for (ggml_type type_src : all_types) {
        for (ggml_type type_dst : {GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 4, 4, 4 }));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 0, 2, 1, 3 })); // cpy by rows
        }
    }
    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 1, 0, 2, 3 })); // cpy not-contiguous
        }
    }

    test_cases.emplace_back(new test_cont());
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 1, 1 ,1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 1, 3 ,5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 3, 5 ,7 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 1, 1 ,1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 1, 3 ,5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 3, 5 ,7 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 1, 1 ,1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 1, 3 ,5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 3, 5 ,7 }));

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : { ggml_add, ggml_sub, ggml_mul, ggml_div }) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        add_test_bin_bcast(type, { 1, 1, 8, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1, 1 }, { 32, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 320, 320 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 1, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 2, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 2, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 2, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 1, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 2, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 2, 2, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 2, 2, 2, 2 });

        // stable diffusion
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 16, 16, 1 });
        add_test_bin_bcast(type, { 1280, 16, 16, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 256, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1280, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 16, 16, 1280, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1920, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 2560, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1280, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1920, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 640, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 5120, 1, 1, 1 }, { 1, 256, 1, 1 });
        add_test_bin_bcast(type, { 640, 1, 1, 1 }, { 1, 1, 1, 1 });
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {2, 1, 1, 1});
    }

    test_cases.emplace_back(new test_add1());
    test_cases.emplace_back(new test_scale());
    test_cases.emplace_back(new test_silu_back());

    for (float eps : {0.0f, 1e-6f, 1e-4f, 1e-1f}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, v, eps));
            test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, v, eps));
        }
        test_cases.emplace_back(new test_rms_norm_back(GGML_TYPE_F32, { 64, 5, 4, 3 }, eps));
        test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, eps));
    }

    test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, 1e-12f));

    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 4, 1536, 1, 1 }, { 4, 1536, 1, 1 }));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 8, 1536, 1, 1 }, { 4, 1536, 1, 1 }));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 4, 1536, 4, 1 }, { 4, 1536, 1, 1 }));

    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 16, 1024, 32, 4));

    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 128, 4));

    for (ggml_type type_a : all_types) {
        for (int i = 1; i < 10; ++i) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 16, i, 256, { 1,  1 }, { 1, 1 }));
        }
    }

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            // test cases without permutation
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 2, 2 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 2, 2 }));

            // test cases with permutation
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));
        }
    }
    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), { 1,  1 }, { 1, 1 }));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1,  1 }, { 1, 1 }));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1 }, { 1, 1 }));
            }
        }
    }
#endif

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 2, 128, { 8,  1 }, { 1, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 83, 2, 128, { 8,  1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 2, 64, { 8,  1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 83, 2, 64, { 8,  1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 45, 128, { 8,  1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45, 64, { 8,  1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 193, { 1,  1 }, { 4, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 67, { 1,  1 }, { 4, 1 }, { 0, 2, 1, 3 }));

    for (auto bs : { 1,2,4,8 }) {
        for (auto nr : { 1,4 }) {
            for (uint32_t m = 0; m < 2; ++m) {
                for (uint32_t k = 0; k < 2; ++k) {
                    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056 + m, 1, 128 + k, { bs,  1 }, { nr, 1 }, { 0, 2, 1, 3 }));
                    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128 + m, 1, 1056 + k, { bs,  1 }, { nr, 1 }, { 0, 1, 2, 3 }, true));
                }
            }
        }
    }

    // sycl backend will limit task global_range < MAX_INT
    // test case for f16-type-convert-to-fp32 kernel with large k under fp32 compute dtype (occurs in stable-diffusion)
    // however this case needs to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // this case is verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 512, 262144, 9216, {1, 1}, {1, 1}));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (bool b : {false, true}) {
                        for (int n : {1, 32, 129}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4}) {
                for (int n_used : {2}) {
                    for (bool b : {false}) {
                        for (int n : {1, 32}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (int n : {1, 16}) {
                for (int k : {1, 16}) {
                    for (int bs2 : {1, 3}) {
                        for (int bs3 : {1, 3}) {
                            for (int nr2 : {1, 2}) {
                                for (int nr3 : {1, 2}) {
                                    test_cases.emplace_back(new test_out_prod(type_a, type_b, 256, n, k, { bs2, bs3 }, { nr2, nr3 }));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_sqr(type));
        test_cases.emplace_back(new test_sqrt(type));
        test_cases.emplace_back(new test_log(type));
        test_cases.emplace_back(new test_sin(type));
        test_cases.emplace_back(new test_cos(type));
        test_cases.emplace_back(new test_clamp(type));
    }

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 1, 1 }, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 3, 1 }, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 3, 2 }, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2 * exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, { ne0, ne1, 1, 1 }, n / 2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : {false, true}) {
        for (float max_bias : {0.0f, 8.0f}) {
            if (!mask && max_bias > 0.0f) continue;
            for (float scale : {1.0f, 0.1f}) {
                for (int64_t ne0 : {16, 1024}) {
                    for (int64_t ne1 : {16, 1024}) {
                        if (mask) {
                            for (ggml_type m_prec : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0,   ne1,   1, 1 }, mask, m_prec, scale, max_bias));
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 }, mask, m_prec, scale, max_bias));
                            }
                        }
                        else {
                            /* The precision of mask here doesn't matter as boolean mask is false */
                            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0,   ne1,   1, 1 }, mask, GGML_TYPE_F32, scale, max_bias));
                            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 }, mask, GGML_TYPE_F32, scale, max_bias));
                        }
                    }
                }
            }
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, false, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 8.0f));

    for (float max_bias : {0.0f, 8.0f}) {
        for (float scale : {1.0f, 0.1f}) {
            for (int64_t ne0 : {16, 1024}) {
                for (int64_t ne1 : {16, 1024}) {
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, { ne0,   ne1,   1, 1 }, scale, max_bias));
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 }, scale, max_bias));
                }
            }
        }
    }

    for (bool fw : {true, false}) { // fw == forward
        bool all = true;

        for (float v : { 0, 1 }) {
            for (float fs : { 1.0f, 1.4245f }) {
                for (float ef : { 0.0f, 0.7465f }) {
                    for (float af : { 1.0f, 1.4245f }) {
                        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                            for (bool ff : {false, true}) { // freq_factors
                                test_cases.emplace_back(new test_rope(type, { 128,  32, 2, 1 }, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 7B

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 128,  40, 2, 1 }, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 13B
                                    test_cases.emplace_back(new test_rope(type, { 128,  52, 2, 1 }, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 30B
                                    test_cases.emplace_back(new test_rope(type, { 128,  64, 2, 1 }, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 65B
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 64,   1, 2, 1 }, 64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,  71, 2, 1 }, 64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,   8, 2, 1 }, 64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1 }, 20, 2, 512, fs, ef, af, ff, v, fw)); // neox (stablelm)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1 }, 32, 2, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 128,  12, 2, 1 }, 128, GGML_ROPE_TYPE_MROPE, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                                    test_cases.emplace_back(new test_rope(type, { 128,  28, 2, 1 }, 128, GGML_ROPE_TYPE_MROPE, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 7B)
                                    test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1 }, 80, GGML_ROPE_TYPE_VISION, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                                }

                                test_cases.emplace_back(new test_rope(type, { 64, 128, 2, 1 }, 64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, { 11, 12, 13, 14 }, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, { 11, 12, 13, 14 }, 7, dim, v));
        }
    }

    for (ggml_sort_order order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 8, 1, 1, 1 }, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 16, 10, 10, 10 }, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 60, 10, 10, 10 }, order)); // qwen
    }

    for (ggml_scale_mode mode : {GGML_SCALE_MODE_NEAREST, GGML_SCALE_MODE_BILINEAR}) {
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, { 512, 512, 3, 2 }, 2, mode));
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, { 512, 512, 3, 2 }, 2, mode, true));
        test_cases.emplace_back(new test_upscale_ext(GGML_TYPE_F32, { 2, 5,  7, 11 }, { 5, 7, 11, 13 }, mode));
    }

    test_cases.emplace_back(new test_sum());
    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_mean());
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, { 64, 64, 320, 1 }));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, { 9, 9, 1280, 1 }));
    test_cases.emplace_back(new test_acc());
    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_pad_reflect_1d());
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    for (int hsk : { 64, 80, 128, 192, 256, }) {
        for (int hsv : { 64, 80, 128, 192, 256, }) {
            if (hsk != 192 && hsk != hsv) continue;
            if (hsk == 192 && (hsv != 128 && hsv != 192)) continue;

            for (bool mask : { true, false }) {
                for (float max_bias : { 0.0f, 8.0f }) {
                    if (!mask && max_bias > 0.0f) continue;
                    for (float logit_softcap : {0.0f, 10.0f}) {
                        if (hsk != 128 && logit_softcap != 0.0f) continue;
                        for (int nh : { 4, }) {
                            for (int nr : { 1, 4, 16 }) {
                                if (nr == 16 && hsk != 128) continue;
                                for (int kv : { 512, 1024, }) {
                                    if (nr != 1 && kv != 512) continue;
                                    for (int nb : { 1, 3, 32, 35, }) {
                                        for (ggml_prec prec : {GGML_PREC_F32, GGML_PREC_DEFAULT}) {
                                            if (hsk != 128 && prec == GGML_PREC_DEFAULT) continue;
                                            for (ggml_type type_KV : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
                                                test_cases.emplace_back(new test_flash_attn_ext(
                                                    hsk, hsv, nh, nr, kv, nb, mask, max_bias, logit_softcap, prec, type_KV));
                                                // run fewer test cases permuted
                                                if (mask == true && max_bias == 0.0f && logit_softcap == 0 && kv == 512) {
                                                    test_cases.emplace_back(new test_flash_attn_ext(
                                                        hsk, hsv, nh, nr, kv, nb, mask, max_bias, logit_softcap, prec, type_KV, { 0, 2, 1, 3 }));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_cross_entropy_loss(GGML_TYPE_F32, { 10, 5, 4, 3 }));
    test_cases.emplace_back(new test_cross_entropy_loss(GGML_TYPE_F32, { 30000, 1, 1, 1 }));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, { 10, 5, 4, 3 }));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, { 30000, 1, 1, 1 }));

    test_cases.emplace_back(new test_opt_step_adamw(GGML_TYPE_F32, { 10, 5, 4, 3 }));

    // these tests are disabled to save execution time, but they can be handy for debugging
#if 0
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    return test_cases;
}

static bool test_backend(ggml_backend_t backend, test_mode mode, const char* op_name) {
    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        //ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        auto backend_cpu = ggml_backend_cpu_init();
        if (backend_cpu == NULL) {
            printf("  Failed to initialize CPU backend\n");
            return false;
        }

        size_t n_ok = 0;
        for (auto& test : test_cases) {
            if (test->eval(backend, backend_cpu.get(), op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        return n_ok == test_cases.size();
    }
}

int main(int argc, char** argv) {
    test_mode mode = MODE_TEST;
    const char* op_name_filter = nullptr;
    const char* backend_filter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        }
        else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        }
        else if (strcmp(argv[i], "grad") == 0) {
            mode = MODE_GRAD;
        }
        else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            }
            else {
                usage(argv);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            }
            else {
                usage(argv);
                return 1;
            }
        }
        else {
            usage(argv);
            return 1;
        }
    }

    auto backend = ggml_backend_cuda_init(0); // init device 0
    bool ok = test_backend(backend.get(), mode, op_name_filter);

    printf("  Backend %s: ", backend->get_name());
    if (ok) {
        printf("\033[1;32mOK\033[0m\n");
        //n_ok++;
    }
    else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
}
