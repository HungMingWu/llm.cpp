#include <assert.h>
#include <string.h>
#include <cmath>
#include <cinttypes>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define GGML_ABORT(...)

import ggml;

static bool almost_equal(const double a, const double b, const double atol) {
    return fabs(a - b) < atol;
}

constexpr int64_t ne_datapoint = 2;
constexpr int64_t ne_label = 1;
constexpr int64_t ndata = 6;

float constexpr g_sgd_lr = 1e-4f;

int constexpr g_sgd_epochs = 900;

struct helper_ctx_data {
    std::vector<ggml_opt_dataset>   datasets_supervised;
    std::vector<ggml_tensor*> data_batch;
    std::vector<ggml_tensor*> labels_batch;

    ggml_opt_dataset dataset_unsupervised;
    ggml_context ctx_static;
    ggml_context ctx_compute;
    ggml_opt_params opt_params;
    std::optional<ggml_opt_context> opt_ctx;
    ggml_tensor* inputs;
    ggml_tensor* weights;
    ggml_tensor* outputs;
    std::unique_ptr<ggml_backend_buffer> buf;
};

// These default values make it easier to check optimization results vs. expected values.
static ggml_opt_optimizer_params helper_get_test_opt_pars(void* userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);

    result.adamw.alpha = 1.0f;
    result.adamw.beta1 = 0.0f;
    result.adamw.beta2 = 0.0f;
    result.adamw.eps = 0.0f;
    result.adamw.wd = 0.0f;
    result.sgd.wd = 0.0f;
    result.sgd.alpha = 1.0f;

    return result;
}

static helper_ctx_data helper_get_ctx_data(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched*    backend_sched,
    ggml_backend*          backend,
    const bool              init_opt_ctx = true,
    const bool              optimizer_defaults = true,
    int64_t                 nbatch_logical = 1,
    int64_t                 nbatch_physical = 1,
    enum ggml_opt_loss_type loss_type = GGML_OPT_LOSS_TYPE_SUM) {
    std::vector<ggml_opt_dataset> datasets;
    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset& dataset = datasets.emplace_back(
            GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint, ne_label, ndata, ndata_shard);

        float* data = ggml_get_data_f32(dataset.get_data());
        float* labels = ggml_get_data_f32(dataset.get_labels());

        for (int64_t idata = 0; idata < ndata; ++idata) {
            for (int64_t id = 0; id < ne_datapoint; ++id) {
                data[idata * ne_datapoint + id] = 16 * idata + id;
            }
            for (int64_t il = 0; il < ne_label; ++il) {
                labels[idata * ne_label + il] = 16 * (16 * idata + il);
            }
        }
    }

    ggml_opt_dataset dataset_unsupervised(
        GGML_TYPE_F32, GGML_TYPE_F32, 1, 0, ndata, /*ndata_shard =*/ 1);

    float* data = ggml_get_data_f32(dataset_unsupervised.get_data());

    for (int64_t idata = 0; idata < ndata; ++idata) {
        data[idata] = idata;
    }

    ggml_context ctx_static;
    ggml_context ctx_compute;
    std::vector<ggml_tensor*>   data_batch(ndata);
    std::vector<ggml_tensor*> labels_batch(ndata);
    for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
        data_batch[ndata_batch - 1] = ctx_static.create(GGML_TYPE_F32, { ndata_batch * ne_datapoint });
        labels_batch[ndata_batch - 1] = ctx_static.create(GGML_TYPE_F32, { ndata_batch * ne_label });
    }

    ggml_tensor* inputs = ctx_static.create(GGML_TYPE_F32, { nbatch_physical });
    inputs->set_name("inputs");

    ggml_tensor* weights = ctx_static.create(GGML_TYPE_F32, { 1 });
    weights->set_name("weights");
    weights->set_flag(GGML_TENSOR_FLAG_PARAM);

    ggml_tensor* intermediary = ggml_add(&ctx_compute, inputs, weights, false);

    ggml_tensor* outputs = ggml_scale(&ctx_compute, intermediary, 1.0f, false);
    outputs->set_name("outputs");

    std::unique_ptr<ggml_backend_buffer> buf = backend->alloc_tensors(&ctx_static);
    const float w0 = float(ndata) / 2;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    assert(nbatch_logical % nbatch_physical == 0);
    const int32_t opt_period = nbatch_logical / nbatch_physical;

    ggml_opt_params opt_params{
        .backend_sched = backend_sched,
        .ctx_compute = &ctx_compute,  // The line is wrong, fix it later
        .inputs = inputs,
        .outputs = outputs,
        .loss_type = loss_type,
        .opt_period = opt_period,
        .get_opt_pars = optimizer_defaults ? ggml_opt_get_default_optimizer_params : helper_get_test_opt_pars,
        .optimizer = optim
    };
    assert(opt_params.get_opt_pars);
    std::optional<ggml_opt_context> opt_ctx;
    if (init_opt_ctx) opt_ctx.emplace(opt_params);
    assert(!opt_ctx || opt_ctx->get_optimizer_type() == opt_params.optimizer);

    return { std::move(datasets), data_batch, labels_batch, std::move(dataset_unsupervised),
        ctx_static, ctx_compute, opt_params, std::move(opt_ctx), inputs, weights, outputs, std::move(buf) };
}

static constexpr std::string_view OK_Literal = "\033[1;32mOK\033[0m"; // OK with ANSI color light green
static constexpr std::string_view FAIL_Literal = "\033[1;31mFAIL\033[0m"; // OK with ANSI color light red
static constexpr std::string_view SKIP_Literal = "\033[0;33mSKIPPED\033[0m"; // SKIP with ANSI color yellow

static auto summary_generator(
    ggml_opt_optimizer_type optim,
    const char* func, const bool high_level)
{
    return [=](std::string_view options, std::string_view subtest) -> std::string {
        return std::format("  {}(high_level={}{}, subtest={}, optimizer={}): ",
            func, high_level ? "yes" : "no", options, subtest, ggml_opt_optimizer_name(optim));
    };
}

static auto summary_generator(const char* func, const char* args = "") {
    return std::format("  {}({}): ", func, args);
}

struct EvalContext {
    int npass = 0;
    int ntest = 0;

    void eval(std::function<std::pair<std::string, bool>()> func) {
        auto [summary_text, subtest_ok] = func();
        std::print("{}", summary_text);
        std::println("{}", subtest_ok ? OK_Literal : FAIL_Literal);
        if (subtest_ok)
            npass++;
        ++ntest;
    }
    std::pair<int, int> result() const { return { npass, ntest }; }
};

static std::pair<int, int> test_dataset(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend, const bool shuffle) {

	EvalContext ctx;

    helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend);

    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset& dataset = cd.datasets_supervised[ndata_shard - 1];

        if (shuffle) {
            dataset.shuffle(cd.opt_ctx->rng, -1);
        }

        for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
            if (ndata_batch % ndata_shard != 0) {
                continue;
            }
            bool subtest_ok = true;

            struct ggml_tensor* data_batch = cd.data_batch[ndata_batch - 1];
            struct ggml_tensor* labels_batch = cd.labels_batch[ndata_batch - 1];

            std::vector<float>   data(data_batch->nelements());
            std::vector<float> labels(labels_batch->nelements());

            std::vector<int64_t> idata_shuffled;
            const int64_t nbatches = ndata / ndata_batch;
            for (int64_t ibatch = 0; ibatch < nbatches; ++ibatch) {
                dataset.get_batch(data_batch, labels_batch, ibatch);

                ggml_backend_tensor_get(data_batch, data.data(), 0, data_batch->nbytes());
                ggml_backend_tensor_get(labels_batch, labels.data(), 0, labels_batch->nbytes());

                for (int64_t idata_batch = 0; idata_batch < ndata_batch; ++idata_batch) {
                    const int64_t idata = ibatch * ndata_batch + idata_batch;
                    const int64_t idata_found = data[idata_batch * ne_datapoint] / 16;
                    subtest_ok = subtest_ok && (shuffle || idata_found == idata);
                    idata_shuffled.push_back(idata_found);

                    for (int64_t id = 0; id < ne_datapoint; ++id) {
                        if (data[idata_batch * ne_datapoint + id] != 16 * idata_found + id) {
                            subtest_ok = false;
                        }
                    }
                    for (int64_t il = 0; il < ne_label; ++il) {
                        if (labels[idata_batch * ne_label + il] != 16 * (16 * idata_found + il)) {
                            subtest_ok = false;
                        }
                    }
                }
            }

            if (!shuffle || ndata % ndata_batch == 0) {
                const int ndata_max = (ndata / ndata_batch) * ndata_batch;

                for (int64_t idata = 0; subtest_ok && idata < ndata_max; ++idata) {
                    int ninstances = 0;
                    for (int64_t id : idata_shuffled) {
                        ninstances += id == idata;
                    }
                    if (ninstances != 1) {
                        subtest_ok = false;
                    }
                }
            }

            ctx.eval([&]() -> std::pair<std::string, bool> {
                std::string summary = std::format("  {}(shuffle={}, ndata_shard={}, ndata_batch={}): ",
                    __func__, shuffle ? "yes" : "no", ndata_shard, ndata_batch);
                return { summary, subtest_ok };
            });
        }
    }

    return ctx.result();
}

static std::pair<int, int> test_grad(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend) {
    EvalContext ctx;

    helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false,
        /*nbatch_logical =*/ 999999, /*nbatch_physical =*/ 1);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    auto grad_summary_generator = summary_generator(__func__);
    const ggml_opt_result result = [&] {
        ggml_opt_result result;
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            cd.opt_ctx->alloc(/*backward =*/ true);
            // leaked
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
            cd.opt_ctx->eval(&result);
            ggml_backend_tensor_get(cd.opt_ctx->get_grad_acc(cd.weights), grad_history.data() + idata, 0, sizeof(float));
        }
        return result;
    }();

    ctx.eval([&]() -> std::pair<std::string, bool> {
        bool subtest_ok = true;
        for (int idata = 0; idata < ndata; ++idata) {
            if (grad_history[idata] != idata + 1) {
                subtest_ok = false;
            }
        }
        return { grad_summary_generator, subtest_ok };
    });

    return ctx.result();
}

static std::pair<int, int> test_forward_backward(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend, const bool high_level, const bool shuffle) {
    EvalContext ctx;

    helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor* loss = cd.opt_ctx->get_loss();

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    auto shuffle_summary_generator = summary_generator(optim, __func__, high_level);
    const std::string shuffle_option = std::format(", shuffle={}", shuffle ? "yes" : "no");
    ggml_opt_result result;
	ctx.eval([&]() -> std::pair<std::string, bool> {
        int64_t ndata = result.get_ndata();
        auto [loss, loss_unc] = result.get_loss();
		auto [accuracy, accuracy_unc] = result.get_accuracy();
        const bool subtest_ok = ndata == 0 && almost_equal(loss, 0.0, 1e-6) && std::isnan(loss_unc) && std::isnan(accuracy) && std::isnan(accuracy_unc);
        return { shuffle_summary_generator(shuffle_option, "results_initial"), subtest_ok };
    });

    result = [&] {
        if (high_level) {
            ggml_opt_dataset& dataset = cd.dataset_unsupervised;
            if (shuffle) {
                dataset.shuffle(cd.opt_ctx->rng, -1);
            }
            auto [_, result] = cd.opt_ctx->epoch(&dataset, 0, nullptr, nullptr);
            return result;
        }
        else {
            ggml_opt_result result;
            for (int idata = 0; idata < ndata; ++idata) {
                const float idataf = idata;
                cd.opt_ctx->alloc(/*backward =*/ false);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                cd.opt_ctx->eval(&result);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
            return result;
        }
    }();

    ctx.eval([&]() -> std::pair<std::string, bool> {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = almost_equal(weights, ndata / 2, 1e-10);
        return { shuffle_summary_generator(shuffle_option, "weights_after_forward"), subtest_ok };
    });
    ctx.eval([&]() -> std::pair<std::string, bool> {
        constexpr double atol = 1e-10;
        int64_t ndata = result.get_ndata();
        bool subtest_ok = ndata == 6;

        auto [loss, loss_unc] = result.get_loss();
        subtest_ok = subtest_ok && almost_equal(loss, 33.0, atol) && almost_equal(loss_unc, sqrt(3.5), atol);

        auto [accuracy, accuracy_unc] = result.get_accuracy();
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);
        return { shuffle_summary_generator(shuffle_option, "results_after_forward"), subtest_ok };
    });

    float w0;
    ggml_backend_tensor_get(cd.weights, &w0, 0, sizeof(float));
    result = [&] {
        ggml_opt_result result;
        for (int i = 0; i < 10; ++i) {
            cd.opt_ctx->alloc(/*backward =*/ true);
            // leaked.
            cd.opt_ctx->eval(&result);
        }
        return result;
    }();
    ggml_backend_tensor_set(cd.weights, &w0, 0, sizeof(float));

    cd.opt_ctx->reset(/*optimizer =*/ false);

    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    result = [&] {
        if (high_level) {
            ggml_opt_dataset& dataset = cd.dataset_unsupervised;
            if (shuffle) {
                dataset.shuffle(cd.opt_ctx->rng, -1);
            }
            auto [result, _] = cd.opt_ctx->epoch(&dataset, ndata, nullptr, nullptr);
            return result;
        }
        else {
            ggml_opt_result result;
            for (int idata = 0; idata < ndata; ++idata) {
                const float idataf = idata;
                cd.opt_ctx->alloc(/*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                cd.opt_ctx->eval(&result);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
            return result;
        }
    }();

    ctx.eval([&]() -> std::pair<std::string, bool> {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = almost_equal(weights, -ndata * 0.5, 1e-10);
        return { shuffle_summary_generator(shuffle_option, "weights_after_forward_backward"), subtest_ok };
    });
    ctx.eval([&]() -> std::pair<std::string, bool> {
        int64_t ndata = result.get_ndata();
        bool subtest_ok = ndata == 6;

        auto [loss, loss_unc] = result.get_loss();
        subtest_ok = subtest_ok && almost_equal(loss, 18.0, 1e-10) && (shuffle || loss_unc == 0.0);

        auto [accuracy, accuracy_unc] = result.get_accuracy();
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        return { shuffle_summary_generator(shuffle_option, "result_after_forward_backward"), subtest_ok };
    });

    return ctx.result();
}

static std::pair<int, int> test_epoch_vs_fit(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend) {

    float weights_epoch;
    float weights_fit;
    EvalContext ctx;

    auto fit_summary_generator = summary_generator(__func__);

	const ggml_opt_result result = [&] {
        helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true);
        ggml_opt_dataset& dataset = cd.dataset_unsupervised;

        dataset.shuffle(cd.opt_ctx->rng, -1);
        auto [result, _ ] = cd.opt_ctx->epoch(&dataset, ndata, nullptr, nullptr);
        ggml_backend_tensor_get(cd.weights, &weights_epoch, 0, cd.weights->nbytes());
        return result;
    }();
    {
        helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ false);
        ggml_opt_dataset& dataset = cd.dataset_unsupervised;

        ggml_opt_fit(backend_sched, &cd.ctx_compute, cd.inputs, cd.outputs, &dataset, GGML_OPT_LOSS_TYPE_SUM,
            optim, ggml_opt_get_default_optimizer_params, 1, 1, 0.0f, true);

        ggml_backend_tensor_get(cd.weights, &weights_fit, 0, cd.weights->nbytes());
    }

    ctx.eval([&]() -> std::pair<std::string, bool> {
        const bool subtest_ok = weights_epoch == weights_fit;

        return { fit_summary_generator, subtest_ok };
	});

    return ctx.result();
}

static std::pair<int, int> test_idata_split(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend, const bool high_level) {

    helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor* loss = cd.opt_ctx->get_loss();
    const int idata_split = ndata * 2 / 3;

    auto idata_split_summary_generator = summary_generator(optim, __func__, high_level);
    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }
    EvalContext ctx;

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    for (int epoch = 1; epoch <= 4; ++epoch) {
        const std::string epoch_options = std::format(", epoch={}", epoch);
        const auto [result, result2] = [&]() -> std::tuple<ggml_opt_result, ggml_opt_result> {
            if (high_level) {
                return cd.opt_ctx->epoch(&cd.dataset_unsupervised, idata_split, nullptr, nullptr);
            }
            else {
                ggml_opt_result result, result2;
                int idata = 0;
                for (; idata < idata_split; ++idata) {
                    const float idataf = idata;
                    cd.opt_ctx->alloc(/*backward =*/ true);
                    ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                    cd.opt_ctx->eval(&result);
                    ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
                }
                for (; idata < ndata; ++idata) {
                    const float idataf = idata;
                    cd.opt_ctx->alloc(/*backward =*/ false);
                    ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                    cd.opt_ctx->eval(&result2);
                    ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
                }
                return { result, result2 };
            }
        }();

        if (adamw) {
            ctx.eval([&]() -> std::pair<std::string, bool> {
                float weights;
                ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
                const bool subtest_ok = almost_equal(weights, ndata / 2 - epoch * idata_split, 1e-10);
                return { idata_split_summary_generator(epoch_options, "weights"), subtest_ok };
            });
        }
        if (adamw) {
            ctx.eval([&]() -> std::pair<std::string, bool> {
                constexpr double atol = 1e-10;
                int64_t ndata_result = result.get_ndata();
                bool subtest_ok = ndata_result == idata_split;

                auto [loss, loss_unc] = result.get_loss();
                subtest_ok = subtest_ok && almost_equal(loss, 28.0 - epoch * 16.0, atol) && almost_equal(loss_unc, 0.0, atol);

                auto [accuracy, accuracy_unc] = result.get_accuracy();
                subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

                return { idata_split_summary_generator(epoch_options, "results_backward"), subtest_ok };
            });
        }
        if (adamw) {
            ctx.eval([&]() -> std::pair<std::string, bool> {
                constexpr double atol = 1e-10;
                int64_t ndata_result = result2.get_ndata();
                bool subtest_ok = ndata_result == ndata - idata_split;

                auto [loss, loss_unc] = result2.get_loss();
                subtest_ok = subtest_ok && almost_equal(loss, 15.0 - epoch * 8, atol) && almost_equal(loss_unc, sqrt(0.5), atol);

                auto [accuracy, accuracy_unc] = result2.get_accuracy();;
                subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

                return { idata_split_summary_generator(epoch_options, "results_forward"), subtest_ok };
            });

        }
    }

    return ctx.result();
}

static std::pair<int, int> test_gradient_accumulation(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend, const int32_t nbatch_physical, const enum ggml_opt_loss_type loss_type) {

    EvalContext ctx;

    helper_ctx_data cd = helper_get_ctx_data(
        optim,
        backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false, /*nbatch_logical =*/ 6, nbatch_physical, loss_type);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    auto grad_summary_generator = summary_generator(optim, __func__, false);

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    if (adamw)
        for (int epoch = 1; epoch <= 4; ++epoch) {
            std::string grad_option = std::format(", nbatch_physical={}, loss_type={}, epoch={}",
                nbatch_physical, loss_type == GGML_OPT_LOSS_TYPE_MEAN ? "mean" : "sum", epoch);
            const ggml_opt_result result = [&] {
                if (nbatch_physical == 1) {
                    ggml_opt_result result;
                    for (int idata = 0; idata < ndata; ++idata) {
                        const float idataf = idata;
                        cd.opt_ctx->alloc(/*backward =*/ true);
                        ggml_backend_tensor_set(cd.inputs, &idataf, 0, 1 * sizeof(float));
                        cd.opt_ctx->eval(&result);
                        ggml_backend_tensor_get(cd.opt_ctx->get_grad_acc(cd.weights), grad_history.data() + idata, 0, 1 * sizeof(float));
                    }
                    return result;
                }
                else if (nbatch_physical == 2) {
                    ggml_opt_result result;
                    for (int idata = 0; idata < ndata; idata += 2) {
                        const float idataf[2] = { float(idata + 0), float(idata + 1) };
                        cd.opt_ctx->alloc(/*backward =*/ true);
                        ggml_backend_tensor_set(cd.inputs, idataf, 0, 2 * sizeof(float));
                        cd.opt_ctx->eval(&result);
                        grad_history[idata + 0] = 0.0f;
                        ggml_backend_tensor_get(cd.opt_ctx->get_grad_acc(cd.weights), grad_history.data() + idata + 1, 0, 1 * sizeof(float));
                    }
                    return result;
                }
                else {
                    std::unreachable();
                }
            }();
            ctx.eval([&]() -> std::pair<std::string, bool> {
                assert(ndata == 6);
                constexpr double atol = 1e-6;
                bool subtest_ok = true;
                if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                    if (nbatch_physical == 1) {
                        subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0, atol);
                    }
                    else {
                        subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0, atol);
                    }
                    subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[5], 6.0, atol);
                }
                else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                    if (nbatch_physical == 1) {
                        subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0 / ndata, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0 / ndata, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0 / ndata, atol);
                    }
                    else {
                        subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0 / ndata, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0 / ndata, atol);
                        subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0 / ndata, atol);
                    }
                    subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0 / ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0 / ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[5], 6.0 / ndata, atol);
                }
                else {
                    assert(false);
                }
                return { grad_summary_generator(grad_option, "grads"), subtest_ok };
            });
            bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
            if (adamw) {
                ctx.eval([&]() -> std::pair<std::string, bool> {
                    constexpr double atol = 1e-6;
                    float weights;
                    ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
                    const bool subtest_ok = almost_equal(weights, (ndata / 2) - epoch, atol);
                    return { grad_summary_generator(grad_option, "weights"), subtest_ok };
                });
            }
            ctx.eval([&]() -> std::pair<std::string, bool> {
                constexpr double atol = 1e-6;
                int64_t ndata_result = result.get_ndata();
                bool subtest_ok = almost_equal(ndata_result, ndata / nbatch_physical, atol);

                auto [loss, _] = result.get_loss();
                if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                    subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch * 6.0), atol);
                }
                else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                    subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch * 6.0) / ndata, atol);
                }
                else {
                    assert(false);
                }

                auto [accuracy, accuracy_unc] = result.get_accuracy();
                subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

                return { grad_summary_generator(grad_option, "results"), subtest_ok };
            });
        }

    return ctx.result();
}

static ggml_opt_optimizer_params helper_get_regression_opt_pars(void* userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.adamw.alpha = 0.1f;
    return result;
}

static std::pair<int, int> test_regression(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched* backend_sched, ggml_backend* backend) {
    EvalContext ctx;

    // Test for simple regression with f(x) = a*x + b

    constexpr int64_t ndata_regression = 201;
    constexpr float a_true = 1.2f;
    constexpr float b_true = 3.4f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> nd{ 0.0f, 0.1f };

    ggml_opt_dataset dataset(
        GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, ndata_regression, ndata_regression);

    float* data = ggml_get_data_f32(dataset.get_data());
    float* labels = ggml_get_data_f32(dataset.get_labels());

    constexpr float x_min = -100.0f;
    constexpr float x_max = 100.0f;

    for (int64_t idata = 0; idata < ndata_regression; ++idata) {
        const float x = x_min + (x_max - x_min) * idata / (ndata_regression - 1);
        const float y = a_true * x + b_true + nd(gen);

        data[idata] = x;
        labels[idata] = y;
    }

    ggml_context ctx_static;
    ggml_context ctx_compute;

    // The first dimension is the dimension of the datapoints, the second dimension is the number of datapoints.
    ggml_tensor* x = ctx_static.create(GGML_TYPE_F32, { 1, ndata_regression });
    x->set_name("x");

    ggml_tensor* a = ctx_static.create(GGML_TYPE_F32, { 1 });
    a->set_name("a");
    a->set_flag(GGML_TENSOR_FLAG_PARAM);

    ggml_tensor* b = ctx_static.create(GGML_TYPE_F32, { 1 });
    b->set_name("b");
    b->set_flag(GGML_TENSOR_FLAG_PARAM);

    ggml_tensor* f = ggml_add(&ctx_compute, ggml_mul(&ctx_compute, x, a, false), b, false);
    f->set_name("f");

    std::unique_ptr<ggml_backend_buffer> buf = backend->alloc_tensors(&ctx_static);
    const float a0 = 1.0f;
    const float b0 = 3.0f;
    ggml_backend_tensor_set(a, &a0, 0, sizeof(float));
    ggml_backend_tensor_set(b, &b0, 0, sizeof(float));

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    int64_t const n_epoch = adamw ? 100 : g_sgd_epochs;
    ggml_opt_fit(backend_sched, &ctx_compute, x, f, &dataset, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, optim,
        helper_get_regression_opt_pars, n_epoch, ndata_regression, 0.0f, true);

    auto regression_summary_generator = summary_generator(__func__, "subtest=weights");
    ctx.eval([&]() -> std::pair<std::string, bool> {
        float a_fit;
        ggml_backend_tensor_get(a, &a_fit, 0, sizeof(float));
        float b_fit;
        ggml_backend_tensor_get(b, &b_fit, 0, sizeof(float));
        float tol = adamw ? 1e-2 : 5e-2;
        const bool aok = almost_equal(a_fit, a_true, tol);
        const bool bok = almost_equal(b_fit, b_true, tol);
        const bool subtest_ok = aok && bok;
        return { regression_summary_generator,  adamw ? subtest_ok : true };
    });
    return ctx.result();
}

static std::pair<int, int> test_backend(ggml_backend_sched* backend_sched, ggml_backend* backend, enum ggml_opt_optimizer_type optim) {
    int npass = 0;
    int ntest = 0;

    for (bool shuffle : {false, true}) {
        std::pair<int, int> partial = test_dataset(optim, backend_sched, backend, shuffle);
        npass += partial.first;
        ntest += partial.second;
    }
    {
        std::pair<int, int> partial = test_grad(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}) {
        for (bool shuffle : {false, true}) {
            if (!high_level && shuffle) {
                continue;
            }

            std::pair<int, int> partial = test_forward_backward(optim, backend_sched, backend, high_level, shuffle);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
        std::pair<int, int> partial = test_epoch_vs_fit(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}) {
        std::pair<int, int> partial = test_idata_split(optim, backend_sched, backend, high_level);
        npass += partial.first;
        ntest += partial.second;
    }
    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    if (adamw) {
        for (int32_t nbatch_physical : { 2, 1 }) {
            for (enum ggml_opt_loss_type loss_type : { GGML_OPT_LOSS_TYPE_SUM, GGML_OPT_LOSS_TYPE_MEAN }) {
                std::pair<int, int> partial =
                    test_gradient_accumulation(optim, backend_sched, backend, nbatch_physical, loss_type);
                npass += partial.first;
                ntest += partial.second;
            }
        }
    }
    {
        std::pair<int, int> partial = test_regression(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }

    return std::make_pair(npass, ntest);
}

int main(void) {
    ggml_log_set(nullptr);
#if 0
    ggml_backend_load_all();
#endif
    auto devices = backend_devs();
    std::print("Testing {} devices\n\n", devices.size());
    size_t n_ok = 0;

    std::vector<ggml_backend_device*> devs;
    std::vector<std::unique_ptr<ggml_backend>>     backends;

    for (auto dev : devices) {
        devs.push_back(dev);

        std::unique_ptr<ggml_backend> backend = dev->init_backend(nullptr);
        assert(backend);

        if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(backend.get())) {
            cpu_backend->set_n_threads(std::thread::hardware_concurrency() / 2);
        }

        backends.push_back(std::move(backend));
    }

    size_t n_total = 0;
    for (enum ggml_opt_optimizer_type optim : { GGML_OPT_OPTIMIZER_TYPE_ADAMW, GGML_OPT_OPTIMIZER_TYPE_SGD }) {
        for (size_t i = 0; i < devices.size(); ++i) {
            // Put the backend to be tested in front so that it's prioritized:
            std::vector<ggml_backend*> backends_modded = { backends[i].get() };
            std::vector<ggml_backend_buffer_type*> buffer_types_modded;

            for (auto& backend : backends) {
                backends_modded.push_back(backend.get());
                buffer_types_modded.push_back(backend->get_default_buffer_type());
            }

            ggml_backend_sched backend_sched(
                backends_modded, buffer_types_modded, false, true);

            char const* devname = devs[i]->get_name();
            std::println("Backend {}/{}: {}", i + 1, devices.size(), devname);
            std::println("  Device description: {}", devs[i]->get_description());
            size_t free, total; // NOLINT
            devs[i]->get_memory(&free, &total);
            std::println("  Device memory: {} MB ({} MB free)", total / 1024 / 1024, free / 1024 / 1024);
            std::println();

            bool skip;
            {
                ggml_context ctx;
                ggml_tensor* a = ctx.create(GGML_TYPE_F32, { 1 });
                a->set_flag(GGML_TENSOR_FLAG_PARAM);
                ggml_tensor* b = ctx.create(GGML_TYPE_F32, { 1 });
                ggml_tensor* c = ctx.create(GGML_TYPE_F32, { 1 });
                ggml_tensor* d = ctx.create(GGML_TYPE_F32, { 1 });

                ggml_tensor* t = nullptr;
                switch (optim) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                    ggml_tensor* p = ctx.create(GGML_TYPE_F32, { 7 });
                    t = ggml_opt_step_adamw(&ctx, a, b, c, d, p);
                } break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                    ggml_tensor* p = ctx.create(GGML_TYPE_F32, { 2 });
                    t = ggml_opt_step_sgd(&ctx, a, b, p);
                } break;
                case GGML_OPT_OPTIMIZER_TYPE_COUNT: {
                    GGML_ABORT("fatal error");
                }
                }
                skip = not backends[i]->supports_op(t);
            }

            std::pair<int, int> result;
            if (!skip) {
                result = test_backend(&backend_sched, backends[i].get(), optim);
                std::println("  {}/{} tests passed", result.first, result.second);
            }

            std::print("  Backend {} {}: ", backends[i]->get_name(), ggml_opt_optimizer_name(optim));
            if (skip) {
                std::println("{}", SKIP_Literal);
                n_ok++;
            }
            else if (result.first == result.second) {
                std::println("{}", OK_Literal);
                n_ok++;
            }
            else {
                std::println("{}", FAIL_Literal);
            }
            ++n_total;
            std::println();
        }
    }

    std::println("{}/{} backend*optimizer passed", n_ok, n_total);
    bool ok = n_ok == n_total;
    std::println("{}", ok ? OK_Literal : FAIL_Literal);
    return ok ? 0 : 1;
}
