#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <array>
#include <memory>
#include <print>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static constexpr size_t MAX_NARGS = 2;

import ggml;

template <>
struct std::formatter<std::array<int64_t, 4>> : std::formatter<std::string> {
    auto format(const std::array<int64_t, 4>& x, std::format_context& ctx) const {
        auto outIt = std::format_to(ctx.out(), "[");
        for (size_t i = 0; i < 4; i++) {
            if (i > 0) {
                outIt = std::format_to(outIt, ", ");
            }
            outIt = std::format_to(outIt, "{}", x[i]);
        }
        return std::format_to(outIt, "]");
    }
};

void get_random_dims(int64_t* dims, int ndims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 4);
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = distrib(gen);
    }
}

float get_element(const ggml_tensor* t, int idx) {
    return ((float*)t->data)[idx];
}

void set_element(ggml_tensor* t, int idx, float value) {
    ((float*)t->data)[idx] = value;
}

bool check_gradient(
    ggml_backend_t backend,
    const char* op_name,
    ggml_context* ctx0,
    const std::array<ggml_tensor *, MAX_NARGS> &x,
    ggml_tensor* f,
    int ndims,
    int nargs,
    float eps,
    float max_error_abs,
    float max_error_rel) {
    const int n_threads = 1;
    f->set_flag(GGML_TENSOR_FLAG_LOSS);

    ggml_cgraph gf;
    gf.build_forward_expand(f);
    ggml_cgraph gb = gf;
    gb.build_backward_expand(ctx0, nullptr);

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx0, backend);

    for (auto& tensor : ctx0->getTensors()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(-1.0, 1.0);
        if (tensor == x[0]) {
            std::vector<float> data(tensor->nelements());
            for (auto& v : data) {
                v = distrib(gen);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
        }
        else {
            std::vector<float> data(tensor->nelements());
            for (auto& v : data) {
                v = distrib(gen);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
        }
    }

    backend->compute(&gf);
    gb.reset();
    backend->compute(&gb);
    ggml_graph_dump_dot(&gf, NULL, "test-grad0-forward.dot");
    ggml_graph_dump_dot(&gb, &gf, "test-grad0-backward.dot");


    for (size_t i = 0; i < nargs; ++i) {
        const int64_t nelements = x[i]->nelements();
        for (int64_t k = 0; k < nelements; ++k) {
            // compute gradient using finite differences
            const float x0 = get_element(x[i], k);

            set_element(x[i], k, x0 + eps);
            backend->compute(&gf);

            const float f0 = ggml_get_f32_1d(f, 0);

            set_element(x[i], k, x0 - eps);
            backend->compute(&gf);

            const float f1 = ggml_get_f32_1d(f, 0);

            const float g0 = (f0 - f1) / (2.0f * eps);

            set_element(x[i], k, x0);

            // compute gradient using backward graph
            gb.reset();
            backend->compute(&gb);

            const float g1 = get_element(ggml_graph_get_grad(&gb, x[i]), k);

            const float error_abs = fabsf(g0 - g1);
            const float error_rel = g0 != 0 ? fabsf(g0 - g1) / fabs(g0) : 0;

            if (error_abs > max_error_abs || error_rel > max_error_rel) {
                std::println("{}: ndims={}, i={}, k={}, g0={}, g1={}, error_abs={}, error_rel={}", op_name, ndims, i, k, g0, g1, error_abs, error_rel);
                assert(false);
            }
        }
    }

    return true;
}


float mat_get(const ggml_tensor* t, int i0, int i1, int i2, int i3) {
    const size_t nb0 = t->nb[0];
    const size_t nb1 = t->nb[1];
    const size_t nb2 = t->nb[2];
    const size_t nb3 = t->nb[3];

    return
        *((float*)((char*)t->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3));
}

bool check_mat_mul(
    const ggml_tensor* y,
    const ggml_tensor* x0,
    const ggml_tensor* x1) {

    std::println("x0: {}", x0->ne);
    for (int j = 0; j < x0->ne[1]; ++j) {
        for (int i = 0; i < x0->ne[0]; ++i) {
            printf("%6.3f ", mat_get(x0, i, j, 0, 0));
        }
        printf("\n");
    }
    std::println();

    std::println("x1: {}", x1->ne);
    for (int j = 0; j < x1->ne[1]; ++j) {
        for (int i = 0; i < x1->ne[0]; ++i) {
            printf("%6.3f ", mat_get(x1, i, j, 0, 0));
        }
        printf("\n");
    }
    std::println();

    std::println("y: {}", y->ne);
    for (int j = 0; j < y->ne[1]; ++j) {
        for (int i = 0; i < y->ne[0]; ++i) {
            printf("%6.3f ", mat_get(y, i, j, 0, 0));
        }
        printf("\n");
    }

    for (int i3 = 0; i3 < y->ne[3]; ++i3) {
        for (int i2 = 0; i2 < y->ne[2]; ++i2) {
            for (int i1 = 0; i1 < y->ne[1]; ++i1) {
                for (int i0 = 0; i0 < y->ne[0]; ++i0) {
                    float sum = 0.0f;
                    for (int k = 0; k < x0->ne[0]; ++k) {
                        sum += mat_get(x0, k, i0, i2, i3) * mat_get(x1, k, i1, i2, i3);
                    }
                    if (fabsf(sum - mat_get(y, i0, i1, i2, i3)) > 1e-5) {
                        std::println("error: i0={}, i1={}, i2={}, i3={}, sum={}, y={}",
                            i0, i1, i2, i3, sum, mat_get(y, i0, i1, i2, i3));
                        assert(false);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

int main(int argc, const char** argv) {
    int64_t ne[4];

    // original loop: 500
    int niter = 500;
    const char* env = getenv("GGML_NLOOP");
    if (env != NULL) {
        niter = atoi(env);
    }
    if (argc > 1) {
        niter = atoi(argv[1]);
    }

    int n_threads = 1;

	std::unique_ptr<ggml_backend> backend = ggml_backend_cpu_init();

    auto create_random_tenser = [](ggml_context* ctx0, int ndims, int64_t* ne) {
        if (ndims == 2) {
            return ctx0->create(GGML_TYPE_F32, {ne[0], ne[1]});
        }
        else if (ndims == 3) {
            return ctx0->create(GGML_TYPE_F32, {ne[0], ne[1], ne[2]});
        }
        else {
            return ctx0->create(GGML_TYPE_F32, {ne[0], ne[1], ne[2], ne[3]});
        }
    };

    for (int iter = 0; iter < niter; ++iter) {
        std::println("test-mul-mat0: iter:{}/{}", iter, niter);
        ggml_context ctx0;

        get_random_dims(ne, 4);

        std::array<ggml_tensor*, MAX_NARGS> x;

        // mul_mat
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = create_random_tenser(&ctx0, ndims, ne);
                ne[1] = rand() % 4 + 1;
                x[1] = create_random_tenser(&ctx0, ndims, ne);

                x[0]->set_flag(GGML_TENSOR_FLAG_PARAM);

                ggml_tensor* m = ggml_mul_mat(&ctx0, x[1], x[0]);
                ggml_tensor* f = ggml_sum(&ctx0, m);

                //auto buffer = backend->get_default_buffer_type()->alloc_buffer(16384);
                //ggml_tallocr alloc(buffer.get());
#if 0
                for (auto& tensor : ctx0.getTensors()) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> distrib(-1.0, 1.0);
                    if (tensor == x[0]) {
                        std::vector<float> data(tensor->nelements());
                        for (auto& v : data) {
                            v = distrib(gen);
                        }
                        alloc.alloc(tensor);
                        ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                    }
                    else {
                        std::vector<float> data(tensor->nelements());
                        for (auto& v : data) {
                            v = distrib(gen);
                        }
                        //alloc.alloc(tensor);
                        ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                    }
                }
#endif
                std::println("testing: mul_mat, {} = {} * {}", m->ne, x[1]->ne, x[0]->ne);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient(backend.get(), "mul_mat", &ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                }
                else {
                    ggml_cgraph gf;
                    gf.build_forward_expand(m);
                    auto buffer = ggml_backend_alloc_ctx_tensors(&ctx0, backend.get());

                    for (auto& tensor : ctx0.getTensors()) {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<float> distrib(-1.0, 1.0);
                        if (tensor == x[0]) {
                            std::vector<float> data(tensor->nelements());
                            for (auto& v : data) {
                                v = distrib(gen);
                            }
                            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                        }
                        else {
                            std::vector<float> data(tensor->nelements());
                            for (auto& v : data) {
                                v = distrib(gen);
                            }
                            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                        }
                    }
                    backend->compute(&gf);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }

        // mul_mat (transposed)
        {
            const int nargs = 1;
            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = create_random_tenser(&ctx0, ndims, ne);
                ne[1] = ne[0];
                ne[0] = rand() % 4 + 1;
                x[1] = ggml_cont(&ctx0, ggml_transpose(&ctx0, create_random_tenser(&ctx0, ndims, ne)));

                x[0]->set_flag(GGML_TENSOR_FLAG_PARAM);

                ggml_tensor* m = ggml_mul_mat(&ctx0, x[1], x[0]);
                ggml_tensor* f = ggml_sum(&ctx0, m);

                std::println("testing: mul_mat, {} = {} * {}", m->ne, x[1]->ne, x[0]->ne);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient(backend.get(), "mul_mat", &ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                }
                else {
                    ggml_cgraph gf;
                    gf.build_forward_expand(m);

                    auto buffer = ggml_backend_alloc_ctx_tensors(&ctx0, backend.get());

                    for (auto& tensor : ctx0.getTensors()) {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<float> distrib(-1.0, 1.0);
                        if (tensor == x[0]) {
                            std::vector<float> data(tensor->nelements());
                            for (auto& v : data) {
                                v = distrib(gen);
                            }
                            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                        }
                        else {
                            std::vector<float> data(tensor->nelements());
                            for (auto& v : data) {
                                v = distrib(gen);
                            }
                            ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                        }
                    }

                    backend->compute(&gf);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }

    }

    return 0;
}
