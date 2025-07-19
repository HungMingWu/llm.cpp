module;
#include <asset.h>
#define assert(...) assert(__VA_ARGS__)

module ggml;

ggml_opt_dataset::ggml_opt_dataset(
    enum ggml_type type_data,
    enum ggml_type type_label,
    int64_t        ne_datapoint,
    int64_t        ne_label,
    int64_t        ndata,
    int64_t        ndata_shard) :ndata(ndata), ndata_shard(ndata_shard) {
    assert(ne_datapoint > 0);
    assert(ne_label >= 0);
    assert(ndata > 0);
    assert(ndata_shard > 0);

    data = ctx.create(type_data, { ne_datapoint, ndata });
    nbs_data = data->nbytes() * ndata_shard / ndata;

    if (ne_label > 0) {
        result->labels = ctx.create(type_label, { ne_label, ndata });
        result->nbs_labels = labels->nbytes() * ndata_shard / ndata;
    }
    else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    buf = ggml_backend_cpu_buffer_type()->alloc_tensors(&ctx);

    const int64_t nshards = ndata / ndata_shard;
    permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        permutation[i] = i;
    }
}

ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void *) {
    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps = 1e-8f;
    result.adamw.wd = 0.0f;

    return result;
}

struct ggml_opt_params ggml_opt_default_params(
    ggml_backend_sched_t      backend_sched,
    enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
    };
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    assert(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    assert((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ggml_set_input(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor* node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        assert(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) +
            (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param * n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    assert(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context* ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
    case GGML_OPT_LOSS_TYPE_MEAN: {
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->loss, "loss_sum");
        const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
        opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
        ggml_set_name(opt_ctx->loss, "loss_mean");
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    case GGML_OPT_LOSS_TYPE_SUM: {
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->loss, "loss_sum");
        opt_ctx->loss_per_datapoint = false;
        break;
    }
    case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
        opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
        ggml_set_input(opt_ctx->labels);
        ggml_set_name(opt_ctx->labels, "labels");
        opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
        ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
        if (opt_ctx->opt_period > 1) {
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
        }
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
        opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
        ggml_set_input(opt_ctx->labels);
        ggml_set_name(opt_ctx->labels, "labels");
        opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
        ggml_set_name(opt_ctx->loss, "loss_error");
        opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
        ggml_set_name(opt_ctx->loss, "loss_squared_error");
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
        ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
        const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
        opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
        ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    }
    else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        assert(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            }
            else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor* node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                }
                else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    }
    else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    assert(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->adamw_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, 7);
    ggml_set_input(opt_ctx->adamw_params);
    ggml_set_name(opt_ctx->adamw_params, "adamw_params");

    for (int i = opt_ctx->gf->n_nodes - 1; i >= 0; --i) {
        struct ggml_tensor* node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor* grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            struct ggml_tensor* m = opt_ctx->grad_m[i];
            struct ggml_tensor* v = opt_ctx->grad_v[i];
            struct ggml_tensor* opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, opt_ctx->adamw_params);

            ggml_set_name(m, (std::string("AdamW m for ") + std::string(node->name)).c_str());
            ggml_set_name(v, (std::string("AdamW v for ") + std::string(node->name)).c_str());
            ggml_set_name(opt_step, (std::string("AdamW step for ") + std::string(node->name)).c_str());

            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}


ggml_opt_context::ggml_opt_context(ggml_opt_params params)
{
    backend_sched = params.backend_sched;
    ctx_compute = params.ctx_compute;
    loss_type = params.loss_type;
    build_type = params.build_type;
    build_type_alloc = params.build_type;
    inputs = params.inputs;
    outputs = params.outputs;
    opt_period = params.opt_period;
    get_opt_pars = params.get_opt_pars;
    get_opt_pars_ud = params.get_opt_pars_ud;

    assert(opt_period >= 1);

    static_graphs = ctx_compute;

    if (!static_graphs) {
        assert(!inputs);
        assert(!outputs);
        return;
    }

    assert(inputs);
    assert(outputs);

    gf.build_forward_expand(outputs);

    ggml_opt_build(this);
}