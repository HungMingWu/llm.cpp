#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <memory>
#include <print>
#include <random>
#include <span>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_NARGS 2

import ggml;

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
    std::span<ggml_tensor *> x,
    ggml_tensor* f,
    int ndims,
    float eps,
    float max_error_abs,
    float max_error_rel) {
    const int n_threads = 1;
    ggml_set_loss(f);

    ggml_cgraph gf;
    gf.build_forward_expand(f);
    ggml_cgraph gb = gf;
    gb.build_backward_expand(ctx0, ctx0, false);

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx0, backend);

    backend->compute(&gf);
    gb.reset();
    backend->compute(&gb);
    ggml_graph_dump_dot(&gf, NULL, "test-grad0-forward.dot");
    ggml_graph_dump_dot(&gb, &gf, "test-grad0-backward.dot");


    for (size_t i = 0; i < x.size(); ++i) {
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
    const int64_t n00 = x0->ne[0];
    const int64_t n10 = x0->ne[1];
    const int64_t n20 = x0->ne[2];
    const int64_t n30 = x0->ne[3];

    const int64_t n01 = x1->ne[0];
    const int64_t n11 = x1->ne[1];
    const int64_t n21 = x1->ne[2];
    const int64_t n31 = x1->ne[3];

    const int64_t n02 = y->ne[0];
    const int64_t n12 = y->ne[1];
    const int64_t n22 = y->ne[2];
    const int64_t n32 = y->ne[3];

    std::println("x0: [{}, {}, {}, {}]", n00, n10, n20, n30);
    for (int j = 0; j < n10; ++j) {
        for (int i = 0; i < n00; ++i) {
            printf("%6.3f ", mat_get(x0, i, j, 0, 0));
        }
        printf("\n");
    }
    std::println();

    std::println("x1: [{}, {}, {}, {}]", n01, n11, n21, n31);
    for (int j = 0; j < n11; ++j) {
        for (int i = 0; i < n01; ++i) {
            printf("%6.3f ", mat_get(x1, i, j, 0, 0));
        }
        printf("\n");
    }
    std::println();

    std::println("y: [{}, {}, {}, {}]", n02, n12, n22, n32);
    for (int j = 0; j < n12; ++j) {
        for (int i = 0; i < n02; ++i) {
            printf("%6.3f ", mat_get(y, i, j, 0, 0));
        }
        printf("\n");
    }

    for (int i3 = 0; i3 < n32; ++i3) {
        for (int i2 = 0; i2 < n22; ++i2) {
            for (int i1 = 0; i1 < n12; ++i1) {
                for (int i0 = 0; i0 < n02; ++i0) {
                    float sum = 0.0f;
                    for (int k = 0; k < n00; ++k) {
                        sum += mat_get(x0, k, i0, i2, i3) * mat_get(x1, k, i1, i2, i3);
                    }
                    if (fabsf(sum - mat_get(y, i0, i1, i2, i3)) > 1e-5) {
                        printf("error: i0=%d, i1=%d, i2=%d, i3=%d, sum=%f, y=%f\n",
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
    int niter = 1;// 500;
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
        printf("test-mul-mat0: iter:%d/%d\n", iter, niter);
        std::unique_ptr<ggml_context> ctx0 = ggml_init();

        get_random_dims(ne, 4);

        ggml_tensor* x[MAX_NARGS];

        // mul_mat
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = create_random_tenser(ctx0.get(), ndims, ne);
                ne[1] = rand() % 4 + 1;
                x[1] = create_random_tenser(ctx0.get(), ndims, ne);

                ggml_set_param(x[0]);

                ggml_tensor* m = ggml_mul_mat(ctx0.get(), x[1], x[0]);
                ggml_tensor* f = ggml_sum(ctx0.get(), m);

                for (auto& tensor : ctx0->tensors) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> distrib(-1.0, 1.0);
                    if (tensor == x[0]) {
                        std::vector<float> data(tensor->nelements());
                        for (auto& v : data) {
                            v = distrib(gen);
                        }
                        //ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                    }
                    else {
                        std::vector<float> data(tensor->nelements());
                        for (auto& v : data) {
                            v = distrib(gen);
                        }
                        //ggml_backend_tensor_set(tensor, data.data(), 0, tensor->nbytes());
                    }
                }

                std::println("testing: mul_mat, [{}, {}, {}, {}] = [{}, {}, {}, {}] * {}, {}, {}, {}]",
                    m->ne[0], m->ne[1], m->ne[2], m->ne[3],
                    x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                    x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient(backend.get(), "mul_mat", ctx0.get(), std::span{ x, nargs }, f, ndims, 1e-3f, 1e-3f, INFINITY);
                }
                else {
                    ggml_cgraph gf;
                    gf.build_forward_expand(m);
                    backend->compute(&gf);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }
#if 0
        // mul_mat (transposed)
        {
            const int nargs = 1;
            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = create_random_tenser(ctx0.get(), ndims, ne);
                ne[1] = ne[0];
                ne[0] = rand() % 4 + 1;
                x[1] = ggml_cont(ctx0.get(), ggml_transpose(ctx0.get(), create_random_tenser(ctx0.get(), ndims, ne)));

                ggml_set_param(x[0]);

                ggml_tensor* m = ggml_mul_mat(ctx0.get(), x[1], x[0]);
                ggml_tensor* f = ggml_sum(ctx0.get(), m);

                std::println("testing: mul_mat, [{}, {}, {}, {}] = [{}, {}, {}, {}] * [{}, {}, {}, {}]",
                    m->ne[0], m->ne[1], m->ne[2], m->ne[3],
                    x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                    x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient("mul_mat", ctx0.get(), x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                }
                else {
                    ggml_cgraph gf;
                    gf.build_forward_expand(m);
                    //ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }
#endif
    }

    return 0;
}
