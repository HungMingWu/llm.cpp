#include <assert.h>
#include <cmath>
#include <cinttypes>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

import ggml;

static bool almost_equal(const double a, const double b, const double atol) {
    return fabs(a - b) < atol;
}

constexpr int64_t ne_datapoint = 2;
constexpr int64_t ne_label = 1;
constexpr int64_t ndata = 6;

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
    ggml_opt_result        result;
    ggml_opt_result        result2;
};

// These default values make it easier to check optimization results vs. expected values.
static ggml_opt_optimizer_params helper_get_test_opt_pars(void* userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.adamw.alpha = 1.0f;
    result.adamw.beta1 = 0.0f;
    result.adamw.beta2 = 0.0f;
    result.adamw.eps = 0.0f;
    return result;
}

static helper_ctx_data helper_get_ctx_data(
    ggml_backend_sched_t    backend_sched,
    ggml_backend_t          backend,
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

    ggml_tensor* intermediary = ggml_add(&ctx_compute, inputs, weights);

    ggml_tensor* outputs = ggml_scale(&ctx_compute, intermediary, 1.0f);
    outputs->set_name("outputs");

    std::unique_ptr<ggml_backend_buffer> buf = backend->alloc_tensors(&ctx_static);
    const float w0 = float(ndata) / 2;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    assert(nbatch_logical % nbatch_physical == 0);
    const int32_t opt_period = nbatch_logical / nbatch_physical;

    ggml_opt_params opt_params = ggml_opt_default_params(backend_sched, loss_type);
    // The line is wrong, fix it later
    opt_params.ctx_compute = &ctx_compute;
    opt_params.inputs = inputs;
    opt_params.outputs = outputs;
    opt_params.opt_period = opt_period;
    if (!optimizer_defaults) {
        opt_params.get_opt_pars = helper_get_test_opt_pars;
    }
    std::optional<ggml_opt_context> opt_ctx;
    if (init_opt_ctx) opt_ctx.emplace(opt_params);

    return { std::move(datasets), data_batch, labels_batch, std::move(dataset_unsupervised),
        ctx_static, ctx_compute, opt_params, std::move(opt_ctx), inputs, weights, outputs, std::move(buf) };
}

static void helper_after_test(
    const char* func, const bool high_level, const std::string options,
    const std::string subtest, const bool subtest_ok, int& ntest, int& npass) {
    printf("  %s(high_level=%s%s, subtest=%s): ",
        func, high_level ? "yes" : "no", options.c_str(), subtest.c_str());
    if (subtest_ok) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    }
    else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;
}

static std::pair<int, int> test_dataset(ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend);

    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset& dataset = cd.datasets_supervised[ndata_shard - 1];

        if (shuffle) {
            ggml_opt_dataset_shuffle(&cd.opt_ctx.value(), &dataset, -1);
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
                ggml_opt_dataset_get_batch(&dataset, data_batch, labels_batch, ibatch);

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

            printf("  %s(shuffle=%s, ndata_shard=%" PRId64 ", ndata_batch=%" PRId64 "): ",
                __func__, shuffle ? "yes" : "no", ndata_shard, ndata_batch);
            if (subtest_ok) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            }
            else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }
    }

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_grad(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false,
        /*nbatch_logical =*/ 999999, /*nbatch_physical =*/ 1);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    for (int idata = 0; idata < ndata; ++idata) {
        const float idataf = idata;
        cd.opt_ctx->alloc(/*backward =*/ true);
        ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
        cd.opt_ctx->eval(&cd.result);
        ggml_backend_tensor_get(ggml_opt_grad_acc(&cd.opt_ctx.value(), cd.weights), grad_history.data() + idata, 0, sizeof(float));
    }

    {
        bool subtest_ok = true;
        for (int idata = 0; idata < ndata; ++idata) {
            if (grad_history[idata] != idata + 1) {
                subtest_ok = false;
            }
        }
        printf("  %s(): ", __func__);
        if (subtest_ok) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        }
        else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    return std::make_pair(npass, ntest);
}

static void helper_after_test_forward_backward(
    const char* func, const bool high_level, const bool shuffle,
    const std::string subtest, const bool subtest_ok, int& ntest, int& npass) {
    std::string options = ", shuffle=";
    options += shuffle ? "yes" : "no";
    helper_after_test(func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_forward_backward(
    ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor* loss = cd.opt_ctx->get_loss();

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    {
        int64_t ndata = cd.result.get_ndata();
        auto [loss, loss_unc] = cd.result.get_loss();
		auto [accuracy, accuracy_unc] = cd.result.get_accuracy();
        const bool subtest_ok = ndata == 0 && loss == 0.0 && std::isnan(loss_unc) && std::isnan(accuracy) && std::isnan(accuracy_unc);
        helper_after_test_forward_backward(__func__, high_level, shuffle, "results_initial", subtest_ok, ntest, npass);
    }

    if (high_level) {
        ggml_opt_dataset &dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(&cd.opt_ctx.value(), &dataset, -1);
        }
        ggml_opt_epoch(&cd.opt_ctx.value(), &dataset, nullptr, &cd.result, 0, nullptr, nullptr);
    }
    else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            cd.opt_ctx->alloc(/*backward =*/ false);
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
            cd.opt_ctx->eval(&cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = weights == ndata / 2;
        helper_after_test_forward_backward(__func__, high_level, shuffle, "weights_after_forward", subtest_ok, ntest, npass);
    }
    {
        int64_t ndata = cd.result.get_ndata();
        bool subtest_ok = ndata == 6;

        auto [loss, loss_unc] = cd.result.get_loss();
        subtest_ok = subtest_ok && loss == 33.0 && almost_equal(loss_unc, sqrt(3.5), 1e-10);

		auto [accuracy, accuracy_unc] = cd.result.get_accuracy();
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(__func__, high_level, shuffle, "results_after_forward", subtest_ok, ntest, npass);
    }

    float w0;
    ggml_backend_tensor_get(cd.weights, &w0, 0, sizeof(float));
    for (int i = 0; i < 10; ++i) {
        cd.opt_ctx->alloc(/*backward =*/ true);
        cd.opt_ctx->eval(&cd.result);
    }
    ggml_backend_tensor_set(cd.weights, &w0, 0, sizeof(float));

    cd.opt_ctx->reset(/*optimizer =*/ false);
    cd.result.reset();

    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    if (high_level) {
        ggml_opt_dataset& dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(&cd.opt_ctx.value(), &dataset, -1);
        }
        ggml_opt_epoch(&cd.opt_ctx.value(), &dataset, &cd.result, nullptr, ndata, nullptr, nullptr);
    }
    else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            cd.opt_ctx->alloc(/*backward =*/ true);
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
            cd.opt_ctx->eval(&cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = weights == -ndata / 2;
        helper_after_test_forward_backward(__func__, high_level, shuffle, "weights_after_forward_backward", subtest_ok, ntest, npass);
    }
    {
        int64_t ndata = cd.result.get_ndata();
        bool subtest_ok = ndata == 6;

        auto [loss, loss_unc] = cd.result.get_loss();
        subtest_ok = subtest_ok && loss == 18.0 && (shuffle || loss_unc == 0.0);

		auto [accuracy, accuracy_unc] = cd.result.get_accuracy();
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(__func__, high_level, shuffle, "result_after_forward_backward", subtest_ok, ntest, npass);
    }

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_epoch_vs_fit(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    float weights_epoch;
    float weights_fit;

    {
        helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true);
        ggml_opt_dataset& dataset = cd.dataset_unsupervised;

        ggml_opt_dataset_shuffle(&cd.opt_ctx.value(), &dataset, -1);
        ggml_opt_epoch(&cd.opt_ctx.value(), &dataset, &cd.result, nullptr, ndata, nullptr, nullptr);

        ggml_backend_tensor_get(cd.weights, &weights_epoch, 0, cd.weights->nbytes());
    }
    {
        helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ false);
        ggml_opt_dataset& dataset = cd.dataset_unsupervised;

        ggml_opt_fit(backend_sched, &cd.ctx_compute, cd.inputs, cd.outputs, &dataset,
            GGML_OPT_LOSS_TYPE_SUM, ggml_opt_get_default_optimizer_params, 1, 1, 0.0f, true);

        ggml_backend_tensor_get(cd.weights, &weights_fit, 0, cd.weights->nbytes());
    }

    const bool subtest_ok = weights_epoch == weights_fit;

    printf("  %s(): ", __func__);
    if (subtest_ok) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    }
    else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    return std::make_pair(npass, ntest);
}

static void helper_after_test_idata_split(
    const char* func, const bool high_level, const int epoch,
    const std::string subtest, const bool subtest_ok, int& ntest, int& npass) {
    std::string options = ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_idata_split(ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level) {
    int ntest = 0;
    int npass = 0;

    helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor* loss = cd.opt_ctx->get_loss();
    const int idata_split = ndata * 2 / 3;

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (high_level) {
            ggml_opt_epoch(&cd.opt_ctx.value(), &cd.dataset_unsupervised, &cd.result, &cd.result2, idata_split, nullptr, nullptr);
        }
        else {
            int idata = 0;
            for (; idata < idata_split; ++idata) {
                const float idataf = idata;
                cd.opt_ctx->alloc(/*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                cd.opt_ctx->eval(&cd.result);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
            for (; idata < ndata; ++idata) {
                const float idataf = idata;
                cd.opt_ctx->alloc(/*backward =*/ false);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, cd.inputs->nbytes());
                cd.opt_ctx->eval(&cd.result2);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
        }

        {
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = weights == ndata / 2 - epoch * idata_split;
            helper_after_test_idata_split(__func__, high_level, epoch, "weights", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result = cd.result.get_ndata();
            bool subtest_ok = ndata_result == idata_split;

            auto [loss, loss_unc] = cd.result.get_loss();
            subtest_ok = subtest_ok && loss == 28.0 - epoch * 16.0 && loss_unc == 0.0;

			auto [accuracy, accuracy_unc] = cd.result.get_accuracy();
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(__func__, high_level, epoch, "results_backward", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result = cd.result2.get_ndata();
            bool subtest_ok = ndata_result == ndata - idata_split;

            auto [loss, loss_unc] = cd.result2.get_loss();
            subtest_ok = subtest_ok && loss == 15.0 - epoch * 8 && almost_equal(loss_unc, sqrt(0.5), 1e-10);

            auto [accuracy, accuracy_unc] = cd.result2.get_accuracy();;
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(__func__, high_level, epoch, "results_forward", subtest_ok, ntest, npass);
        }

        cd.result.reset();
        cd.result2.reset();
    }

    return std::make_pair(npass, ntest);
}

static void helper_after_test_gradient_accumulation(
    const char* func, const int nbatch_physical, const enum ggml_opt_loss_type loss_type, const int epoch,
    const std::string subtest, const bool subtest_ok, int& ntest, int& npass) {
    std::string options = ", nbatch_physical=";
    options += std::to_string(nbatch_physical);
    options += ", loss_type=";
    options += loss_type == GGML_OPT_LOSS_TYPE_MEAN ? "mean" : "sum";
    options += ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(func, false, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_gradient_accumulation(
    ggml_backend_sched_t backend_sched, ggml_backend_t backend, const int32_t nbatch_physical, const enum ggml_opt_loss_type loss_type) {
    int ntest = 0;
    int npass = 0;

    helper_ctx_data cd = helper_get_ctx_data(
        backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false, /*nbatch_logical =*/ 6, nbatch_physical, loss_type);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (nbatch_physical == 1) {
            for (int idata = 0; idata < ndata; ++idata) {
                const float idataf = idata;
                cd.opt_ctx->alloc(/*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, 1 * sizeof(float));
                cd.opt_ctx->eval(&cd.result);
                ggml_backend_tensor_get(ggml_opt_grad_acc(&cd.opt_ctx.value(), cd.weights), grad_history.data() + idata, 0, 1 * sizeof(float));
            }
        }
        else if (nbatch_physical == 2) {
            for (int idata = 0; idata < ndata; idata += 2) {
                const float idataf[2] = { float(idata + 0), float(idata + 1) };
                cd.opt_ctx->alloc(/*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, idataf, 0, 2 * sizeof(float));
                cd.opt_ctx->eval(&cd.result);

                grad_history[idata + 0] = 0.0f;
                ggml_backend_tensor_get(ggml_opt_grad_acc(&cd.opt_ctx.value(), cd.weights), grad_history.data() + idata + 1, 0, 1 * sizeof(float));
            }
        }
        else {
            assert(false);
        }

        {
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
            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "grads", subtest_ok, ntest, npass);
        }
        {
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = weights == (ndata / 2) - epoch;
            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "weights", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result = cd.result.get_ndata();
            bool subtest_ok = ndata_result == ndata / nbatch_physical;

            auto [loss, _] = cd.result.get_loss();
            if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                subtest_ok = subtest_ok && loss == (39.0 - epoch * 6.0);
            }
            else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch * 6.0) / ndata, 1e-6);
            }
            else {
                assert(false);
            }

			auto [accuracy, accuracy_unc] = cd.result.get_accuracy();
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "results", subtest_ok, ntest, npass);
        }

        cd.result.reset();
    }

    return std::make_pair(npass, ntest);
}

static ggml_opt_optimizer_params helper_get_regression_opt_pars(void* userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.adamw.alpha = 0.1f;
    return result;
}

static std::pair<int, int> test_regression(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

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

    ggml_tensor* f = ggml_add(&ctx_compute, ggml_mul(&ctx_compute, x, a), b);
    f->set_name("f");

    std::unique_ptr<ggml_backend_buffer> buf = backend->alloc_tensors(&ctx_static);
    const float a0 = 1.0f;
    const float b0 = 3.0f;
    ggml_backend_tensor_set(a, &a0, 0, sizeof(float));
    ggml_backend_tensor_set(b, &b0, 0, sizeof(float));

    ggml_opt_fit(backend_sched, &ctx_compute, x, f, &dataset, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
        helper_get_regression_opt_pars, 100, ndata_regression, 0.0f, true);

    {
        float a_fit;
        ggml_backend_tensor_get(a, &a_fit, 0, sizeof(float));
        float b_fit;
        ggml_backend_tensor_get(b, &b_fit, 0, sizeof(float));
        const bool subtest_ok = almost_equal(a_fit, a_true, 1e-2) && almost_equal(b_fit, b_true, 1e-2);
        printf("  %s(subtest=weights): ", __func__);
        if (subtest_ok) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        }
        else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_backend(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int npass = 0;
    int ntest = 0;

    for (bool shuffle : {false, true}) {
        std::pair<int, int> partial = test_dataset(backend_sched, backend, shuffle);
        npass += partial.first;
        ntest += partial.second;
    }
    {
        std::pair<int, int> partial = test_grad(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}) {
        for (bool shuffle : {false, true}) {
            if (!high_level && shuffle) {
                continue;
            }

            std::pair<int, int> partial = test_forward_backward(backend_sched, backend, high_level, shuffle);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
        std::pair<int, int> partial = test_epoch_vs_fit(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}) {
        std::pair<int, int> partial = test_idata_split(backend_sched, backend, high_level);
        npass += partial.first;
        ntest += partial.second;
    }
    for (int32_t nbatch_physical : {2, 1}) {
        for (enum ggml_opt_loss_type loss_type : {GGML_OPT_LOSS_TYPE_SUM, GGML_OPT_LOSS_TYPE_MEAN}) {
            std::pair<int, int> partial = test_gradient_accumulation(backend_sched, backend, nbatch_physical, loss_type);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
        std::pair<int, int> partial = test_regression(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }

    return std::make_pair(npass, ntest);
}

int main(void) {
    const size_t dev_count = ggml_backend_dev_count();
    printf("Testing %zu devices\n\n", dev_count);
    size_t n_ok = 0;

    std::vector<ggml_backend_dev_t> devs;
    std::vector<std::unique_ptr<ggml_backend>>     backends;

    for (size_t i = 0; i < dev_count; ++i) {
        devs.push_back(ggml_backend_dev_get(i));

        std::unique_ptr<ggml_backend> backend = devs[i]->init_backend(nullptr);
        assert(backend);

#if 0
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, std::thread::hardware_concurrency() / 2);
        }
#endif
        backends.push_back(std::move(backend));
    }
    for (size_t i = 0; i < dev_count; ++i) {
        // Put the backend to be tested in front so that it's prioritized:
        std::vector<ggml_backend_t> backends_modded = { backends[i].get()};
        for (auto& backend : backends)
            backends_modded.push_back(backend.get());

        ggml_backend_sched backend_sched(
            backends_modded.data(), nullptr, backends_modded.size(), false, true);

        printf("Backend %zu/%zu: %s\n", i + 1, dev_count, devs[i]->get_name());
        printf("  Device description: %s\n", devs[i]->get_description());
        size_t free, total; // NOLINT
        devs[i]->get_memory(&free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        std::pair<int, int> result = test_backend(&backend_sched, backends[i].get());

        printf("  %d/%d tests passed\n", result.first, result.second);
        printf("  Backend %s: ", backends[i]->get_name());
        if (result.first == result.second) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        }
        else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");
    }

    printf("%zu/%zu backends passed\n", n_ok, dev_count);
    if (n_ok != dev_count) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
