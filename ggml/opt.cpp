module;
#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <map>
#include <print>
#include <string>
#include <tuple>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_UNUSED(x) (void)(x)
#define GGML_ABORT(...)

module ggml;

ggml_opt_dataset::ggml_opt_dataset(
    enum ggml_type type_data,
    enum ggml_type type_label,
    int64_t        ne_datapoint,
    int64_t        ne_label,
    int64_t        ndata,
    int64_t        ndata_shard) :ndata(ndata), ndata_shard(ndata_shard) {
    GGML_ASSERT(ne_datapoint > 0);
    GGML_ASSERT(ne_label >= 0);
    GGML_ASSERT(ndata > 0);
    GGML_ASSERT(ndata_shard > 0);

    data = ctx.create(type_data, { ne_datapoint, ndata });
    nbs_data = data->nbytes() * ndata_shard / ndata;

    if (ne_label > 0) {
        labels = ctx.create(type_label, { ne_label, ndata });
        nbs_labels = labels->nbytes() * ndata_shard / ndata;
    }
    else {
        labels = nullptr;
        nbs_labels = 0;
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

    result.sgd.alpha = 1e-3f;
    result.sgd.wd = 0.0f;

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
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

static void ggml_opt_build(ggml_opt_context* opt_ctx) {
    assert(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    assert((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const enum ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    opt_ctx->inputs->set_flag(GGML_TENSOR_FLAG_INPUT);
    opt_ctx->outputs->set_flag(GGML_TENSOR_FLAG_OUTPUT);

    int n_param = 0;
    for (const ggml_tensor* node : opt_ctx->gf.nodes) {
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        assert(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }
    
    opt_ctx->ctx_static.getTensors().clear();
    assert(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    opt_ctx->ctx_cpu.getTensors().clear();
    opt_ctx->buf_cpu.reset();

    ggml_context* ctx_results = opt_ctx->static_graphs ? &opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
    case GGML_OPT_LOSS_TYPE_MEAN: {
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
        opt_ctx->loss->set_name("loss_sum");
        const float scale = 1.0f / (opt_ctx->opt_period * opt_ctx->outputs->nelements());
        opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
        opt_ctx->loss->set_name("loss_mean");
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    case GGML_OPT_LOSS_TYPE_SUM: {
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
        opt_ctx->loss->set_name("loss_sum");
        opt_ctx->loss_per_datapoint = false;
        break;
    }
    case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
        opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
        opt_ctx->labels->set_flag(GGML_TENSOR_FLAG_INPUT);
        opt_ctx->labels->set_name("labels");
        opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
        opt_ctx->loss->set_name("loss_cross_entropy");
        if (opt_ctx->opt_period > 1) {
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
            opt_ctx->loss->set_name("loss_cross_entropy_scaled");
        }
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
        opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
        opt_ctx->labels->set_flag(GGML_TENSOR_FLAG_INPUT);
        opt_ctx->labels->set_name("labels");
        opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
        opt_ctx->loss->set_name("loss_error");
        opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
        opt_ctx->loss->set_name("loss_squared_error");
        opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
        opt_ctx->loss->set_name("loss_sum_squared_error");
        const float scale = 1.0f / (opt_ctx->opt_period * opt_ctx->outputs->nelements());
        opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
        opt_ctx->loss->set_name("loss_mean_squared_error");
        opt_ctx->loss_per_datapoint = true;
        break;
    }
    }
    opt_ctx->loss->set_flag(GGML_TENSOR_FLAG_OUTPUT);
    opt_ctx->loss->set_flag(GGML_TENSOR_FLAG_LOSS);
    opt_ctx->gf.build_forward_expand(opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        opt_ctx->pred->set_name("pred");
        opt_ctx->pred->set_flag(GGML_TENSOR_FLAG_OUTPUT);
        opt_ctx->gf.build_forward_expand(opt_ctx->pred);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        opt_ctx->ncorrect->set_name("ncorrect");
        opt_ctx->ncorrect->set_flag(GGML_TENSOR_FLAG_OUTPUT);
        opt_ctx->gf.build_forward_expand(opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    }
    else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = opt_ctx->backend_sched->get_backend(0)->alloc_tensors(
            &opt_ctx->ctx_static);
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        assert(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf.nodes.size();
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = opt_ctx->gf.nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                const auto& ne = node->ne;
                opt_ctx->grad_accs[i] = opt_ctx->ctx_static.create(GGML_TYPE_F32, { ne[0], ne[1], ne[2], ne[3] });
            }
            else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (size_t i = 0; i < opt_ctx->gf.nodes.size(); i++) {
                const ggml_tensor* node = opt_ctx->gf.nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    const auto& ne = node->ne;
                    opt_ctx->grad_m[i] = opt_ctx->ctx_static.create(GGML_TYPE_F32, { ne[0], ne[1], ne[2], ne[3] });
                    opt_ctx->grad_v[i] = opt_ctx->ctx_static.create(GGML_TYPE_F32, { ne[0], ne[1], ne[2], ne[3] });
                }
                else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = opt_ctx->gf;
    opt_ctx->gb_grad.build_backward_expand(opt_ctx->ctx_compute, opt_ctx->grad_accs);
    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    }
    else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = opt_ctx->backend_sched->get_backend(0)->alloc_tensors(&opt_ctx->ctx_static);
        opt_ctx->gb_grad.reset();
    }

    assert(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = opt_ctx->gb_grad;

    opt_ctx->opt_step_params = opt_ctx->ctx_cpu.create(GGML_TYPE_F32, { need_momenta ? 7 : 2 });
    ggml_tensor* adamw_params = opt_ctx->opt_step_params;
    adamw_params->set_flag(GGML_TENSOR_FLAG_INPUT);
    const char* optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    adamw_params->set_name(std::format("{}_params", optimizer_name));

    for (int i = opt_ctx->gf.nodes.size() - 1; i >= 0; --i) {
        ggml_tensor* node = opt_ctx->gb_opt.nodes[i];
        ggml_tensor* grad = ggml_graph_get_grad(&opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            ggml_tensor* m = nullptr;
            ggml_tensor* v = nullptr;
            if (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                m->set_name(std::format("AdamW m for {}", node->get_name()));
                v->set_name(std::format("AdamW v for {}", node->get_name()));
            }
            struct ggml_tensor* opt_step;
            switch (optimizer) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD:
                opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
                break;
            default:
                GGML_ABORT("fatal error");
            }
            opt_step->set_name(std::format("{} step for {}", optimizer_name, node->get_name()));
            opt_ctx->gb_opt.build_forward_expand(opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = opt_ctx->backend_sched->get_backend(0)->alloc_tensors(
            &opt_ctx->ctx_static);
        opt_ctx->gb_opt.reset();
    }

    opt_ctx->buf_cpu = ggml_backend_cpu_buffer_type()->alloc_tensors(&opt_ctx->ctx_cpu);
}

static ggml_tensor* map_tensor(std::map<ggml_tensor*, ggml_tensor*>& tensor_map, ggml_context* ctx, ggml_tensor* tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor* new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    new_tensor->name = tensor->name;
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (auto &src : tensor->src) {
        new_tensor->src.push_back(map_tensor(tensor_map, ctx, src));
    }

    return new_tensor;
}

static ggml_cgraph dup_graph(ggml_context* ctx, ggml_cgraph* src) {
    std::map<ggml_tensor*, ggml_tensor*> tensor_map;

    ggml_cgraph dst;

    for (auto &leaf : src->leafs) {
        dst.build_forward_expand(map_tensor(tensor_map, ctx, leaf));
    }
    GGML_ASSERT(dst.leafs.size() == src->leafs.size());
    for (auto &node : src->nodes) {
        dst.build_forward_expand(map_tensor(tensor_map, ctx, node));
    }
    GGML_ASSERT(dst.nodes.size() == src->nodes.size());
    for (size_t i = 0; i < src->nodes.size(); ++i) {
        ggml_tensor* src_node = src->nodes[i];
        ggml_tensor* dst_node = dst.nodes[i];
        dst.grads[dst_node] = src->grads[src_node];
        dst.grad_accs[dst_node] = src->grad_accs[src_node];
    }
    return dst;
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

void ggml_opt_context::alloc(bool backward) {
    assert(!eval_ready);
    if (build_type == GGML_OPT_BUILD_TYPE_OPT && opt_period > 1 && opt_i == 0) {
        gb_grad.reset();
    }
    if (backward) {
        const int32_t opt_i_next = (opt_i + 1) % opt_period;
        build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    }
    else {
        build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!static_graphs) {
        ggml_opt_build(this);
    }

    ggml_cgraph* graph = nullptr;
    switch (build_type) {
    case GGML_OPT_BUILD_TYPE_FORWARD: {
        graph = &gf;
    } break;
    case GGML_OPT_BUILD_TYPE_GRAD: {
        graph = &gb_grad;
    } break;
    case GGML_OPT_BUILD_TYPE_OPT: {
        graph = &gb_opt;
    } break;
    }
    assert(graph);

    if (allocated_graph == graph) {
        eval_ready = true;
        return;
    }

    backend_sched->reset(); // clear allocation of previous graph

    if (static_graphs) {
        ctx_copy.getTensors().clear();

        dup_static_graph = dup_graph(&ctx_copy, graph);
        allocated_graph_copy = &dup_static_graph;
    }
    else {
        allocated_graph_copy = graph;
    }

    backend_sched->alloc_graph(*allocated_graph_copy);
    allocated_graph = graph;

    eval_ready = true;
}

std::tuple<double, double> ggml_opt_result::get_loss() const {
    const int64_t nbatches = this->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        return { 0.0, std::numeric_limits<double>::quiet_NaN() };
    }

    double sum = 0.0;
    double sum_squared = 0.0;

    for (const float& loss : this->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = loss_per_datapoint ? loss * opt_period : loss;
        sum += loss_scaled;
        sum_squared += loss_scaled * loss_scaled;
    }

    const double mean = sum / nbatches;
    double loss = loss_per_datapoint ? mean : sum;

    if (nbatches < 2) {
        return { loss, std::numeric_limits<double>::quiet_NaN() };
    }

    const double var_sum = sum_squared / nbatches - mean * mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    double unc = loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches / (nbatches - 1));
    return { loss, unc };
}

void ggml_opt_context::eval(ggml_opt_result* result) {
    GGML_ASSERT(eval_ready);
    if (allocated_graph == &gb_opt) {
        const ggml_opt_optimizer_params& opt_pars = get_opt_pars(get_opt_pars_ud);

        switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
            GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
            GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
            GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
            GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
            GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
            GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
            GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
            GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

            // beta1, beta2 after applying warmup
            const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, iter));
            const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, iter));

            float* adamw_par_data = ggml_get_data_f32(opt_step_params);
            adamw_par_data[0] = opt_pars.adamw.alpha;
            adamw_par_data[1] = opt_pars.adamw.beta1;
            adamw_par_data[2] = opt_pars.adamw.beta2;
            adamw_par_data[3] = opt_pars.adamw.eps;
            adamw_par_data[4] = opt_pars.adamw.wd;
            adamw_par_data[5] = beta1h;
            adamw_par_data[6] = beta2h;
        } break;
        case GGML_OPT_OPTIMIZER_TYPE_SGD: {
            GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
            GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
            GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
            float* sgd = ggml_get_data_f32(opt_step_params);
            sgd[0] = opt_pars.sgd.alpha;
            sgd[1] = opt_pars.sgd.wd;
        } break;
        default:
            GGML_ABORT("fatal error");
        }
    }

    backend_sched->graph_compute(*allocated_graph_copy);
    iter += allocated_graph == &gb_opt;
    opt_i = (opt_i + 1) % opt_period;

    if (!static_graphs) {
        // TODO
#if 0
        gf = nullptr;
        gb_grad = nullptr;
        gb_opt = nullptr;
        allocated_graph = nullptr;
        allocated_graph_copy = nullptr;
#endif
    }

    eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = loss_per_datapoint;
        result->opt_period = opt_period;
    }
    else {
        GGML_ASSERT(result->loss_per_datapoint == loss_per_datapoint);
        GGML_ASSERT(result->opt_period == opt_period);
    }

    const int64_t ndata = outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata * int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(this->loss));
    GGML_ASSERT(this->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(this->loss, &loss, 0, this->loss->nbytes());
    result->loss.push_back(loss);

    if (pred) {
        GGML_ASSERT(pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(this->pred, pred.data(), 0, this->pred->nbytes());
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(this->ncorrect));
    GGML_ASSERT(this->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(this->ncorrect, &ncorrect, 0, this->ncorrect->nbytes());
    result->ncorrect += ncorrect;
}

void ggml_opt_context::reset(bool optimizer) {
    if (optimizer) {
        gb_opt.reset();
        iter = 1;
    }
    else {
        gb_grad.reset();
    }
}

ggml_tensor* ggml_opt_grad_acc(ggml_opt_context* opt_ctx, ggml_tensor* node) {
    return ggml_graph_get_grad_acc(&opt_ctx->gb_opt, node);
}

std::tuple<double, double> ggml_opt_result::get_accuracy() const {
    double accuracy = ncorrect >= 0 ? double(ncorrect) / double(ndata) : std::numeric_limits<double>::quiet_NaN();
    double unc = ncorrect >= 0 && ndata >= 2 ?
        sqrt(accuracy * (1.0 - accuracy) / double(ndata - 1)) : std::numeric_limits<double>::quiet_NaN();
	return { accuracy, unc };
}

void ggml_opt_result::reset() {
    ndata = 0;
    loss.clear();
    pred.clear();
    ncorrect = 0;
}

void ggml_opt_epoch_callback_progress_bar(
    bool               train,
    ggml_opt_context* opt_ctx,
    ggml_opt_dataset* dataset,
    ggml_opt_result*  result,
    int64_t            ibatch,
    int64_t            ibatch_max,
    int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if (ibatch_max * (8 * j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        }
        else if (ibatch_max * (8 * j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        }
        else if (ibatch_max * (8 * j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        }
        else if (ibatch_max * (8 * j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        }
        else if (ibatch_max * (8 * j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        }
        else if (ibatch_max * (8 * j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        }
        else if (ibatch_max * (8 * j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        }
        else if (ibatch_max * (8 * j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        }
        else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = opt_ctx->get_inputs()->ne[1];
    const int64_t idata = ibatch * batch_size;
    const int64_t idata_max = ibatch_max * batch_size;

    auto [loss, loss_unc] = result->get_loss();

    auto [accuracy, accuracy_unc] = result->get_accuracy();

    const int64_t t_ibatch_us = 0;// ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch) / ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    std::print(stderr, "] data={:07}/{:07} loss={:.5f}±{:.5f} acc={:.2f}±{:.2f}% "
                             "t={:02}:{:02}:{:02} ETA={:02}:{:02}:{:02} \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
    ggml_backend_sched_t            backend_sched,
    ggml_context* ctx_compute,
    ggml_tensor* inputs,
    ggml_tensor* outputs,
    ggml_opt_dataset*               dataset,
    enum ggml_opt_loss_type         loss_type,
    enum ggml_opt_optimizer_type    optimizer,
    ggml_opt_get_optimizer_params   get_opt_pars,
    int64_t                         nepoch,
    int64_t                         nbatch_logical,
    float                           val_split,
    bool                            silent) {
    //ggml_time_init();
    //const int64_t t_start_us = ggml_time_us();

    const int64_t ndata = dataset->get_data()->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata % nbatch_logical == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split < 1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute = ctx_compute;
    params.inputs = inputs;
    params.outputs = outputs;
    params.opt_period = opt_period;
    params.get_opt_pars = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer = optimizer;
    ggml_opt_context opt_ctx(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(&opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result result_train;
    ggml_opt_result result_val;

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(&opt_ctx, dataset, idata_split);
        }

        result_train.reset();
        result_val.reset();

        if (!silent) {
            std::println(stderr, "{}: epoch {:04}/{:04}:", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(&opt_ctx, dataset, &result_train, &result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = 0;// (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        std::println(stderr, "{}: training took {:02}:{:02}:{:02}", __func__, t_total_h, t_total_m, t_total_s);
    }
}

void ggml_opt_dataset_shuffle(ggml_opt_context* opt_ctx, ggml_opt_dataset* dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset* dataset, ggml_tensor* data_batch, ggml_tensor* labels_batch, int64_t ibatch) {
    GGML_ASSERT(data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = data_batch->nbytes();
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = labels_batch->nbytes();
        GGML_ASSERT(nb_labels_batch == shards_per_batch * dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1) * shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch * shards_per_batch + ishard_batch];

        const char* ptr_data = (const char*)dataset->data->data + ishard * dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch * dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char* ptr_labels = (const char*)dataset->labels->data + ishard * dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch * dataset->nbs_labels, dataset->nbs_labels);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context* opt_ctx) {
    return opt_ctx->static_graphs;
}

void ggml_opt_epoch(
    ggml_opt_context*      opt_ctx,
    ggml_opt_dataset*      dataset,
    ggml_opt_result*       result_train,
    ggml_opt_result*       result_eval,
    int64_t                 idata_split,
    ggml_opt_epoch_callback callback_train,
    ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    ggml_tensor* inputs = opt_ctx->get_inputs();
    ggml_tensor* labels = opt_ctx->get_labels();
    ggml_tensor* data = dataset->get_data();
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata = data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata / ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = 0; // ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        opt_ctx->alloc(/*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        opt_ctx->eval(result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch + 1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = 0;// ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        opt_ctx->alloc(/*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        opt_ctx->eval(result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch + 1 - ibatch_split, nbatches - ibatch_split, t_loop_start);
        }
    }
}

const char* ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
    case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
        return "adamw";
    case GGML_OPT_OPTIMIZER_TYPE_SGD:
        return "sgd";
    default:
        return "undefined";
    };
}