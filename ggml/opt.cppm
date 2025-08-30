module;
#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

export module ggml:opt;
import :ds;

export
{
    // built-in loss types, i.e. the built-in quantities minimized by the optimizer
    // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
    enum ggml_opt_loss_type {
        GGML_OPT_LOSS_TYPE_MEAN,
        GGML_OPT_LOSS_TYPE_SUM,
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
        GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
    };

    class ggml_opt_dataset {
        // Modify it later
    public:
        ggml_context ctx;
        std::unique_ptr<ggml_backend_buffer> buf;
        ggml_tensor* data = nullptr;
        ggml_tensor* labels = nullptr;

        int64_t ndata = -1;
        int64_t ndata_shard = -1;
        size_t  nbs_data = -1;
        size_t  nbs_labels = -1;

        std::vector<int64_t> permutation;
    public:
        ggml_opt_dataset(
            enum ggml_type type_data,    // the type for the internal data tensor
            enum ggml_type type_label,   // the type for the internal labels tensor
            int64_t        ne_datapoint, // number of elements per datapoint
            int64_t        ne_label,     // number of elements per label
            int64_t        ndata,        // total number of datapoints/labels
            int64_t        ndata_shard); // number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
        ggml_opt_dataset(const ggml_opt_dataset&) = delete;
        ggml_opt_dataset(ggml_opt_dataset&&) = default;

        ggml_tensor* get_data() { return data; }
        ggml_tensor* get_labels() { return labels; }
        void shuffle(std::mt19937& rng, int64_t idata);
        void get_batch(ggml_tensor* data_batch, ggml_tensor* labels_batch, int64_t ibatch);
    };

    // callback to calculate optimizer parameters prior to a backward pass
    // userdata can be used to pass arbitrary data
    typedef struct ggml_opt_optimizer_params(*ggml_opt_get_optimizer_params)(void* userdata);

    enum ggml_opt_build_type {
        GGML_OPT_BUILD_TYPE_FORWARD = 10,
        GGML_OPT_BUILD_TYPE_GRAD = 20,
        GGML_OPT_BUILD_TYPE_OPT = 30,
    };

    enum ggml_opt_optimizer_type {
        GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        GGML_OPT_OPTIMIZER_TYPE_SGD,

        GGML_OPT_OPTIMIZER_TYPE_COUNT
    };

    ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void*);

    // parameters for initializing a new optimization context
    struct ggml_opt_params {
        ggml_backend_sched* backend_sched; // defines which backends are used to construct the compute graphs

        // by default the forward graph needs to be reconstructed for each eval
        // if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
        ggml_context* ctx_compute = nullptr;
        ggml_tensor* inputs = nullptr;
        ggml_tensor* outputs = nullptr;

        ggml_opt_loss_type  loss_type;
        ggml_opt_build_type build_type = GGML_OPT_BUILD_TYPE_OPT;

        int32_t opt_period = 1; // after how many gradient accumulation steps an optimizer step should be done

        ggml_opt_get_optimizer_params get_opt_pars = ggml_opt_get_default_optimizer_params; // callback for calculating optimizer parameters
        void* get_opt_pars_ud = nullptr;                     // userdata for calculating optimizer parameters

        // only GGML_OPT_OPTIMIZER_TYPE_ADAMW needs m, v momenta per parameter tensor
        ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    };

    struct ggml_opt_result {
        int64_t              ndata = 0;
        std::vector<float>   loss;
        std::vector<int32_t> pred;
        int64_t              ncorrect = 0;

        int64_t opt_period = -1;
        bool    loss_per_datapoint = false;
    public:
        int64_t get_ndata() const { return ndata; }
        std::tuple<double, double> get_loss() const;
        std::tuple<double, double> get_accuracy() const;
    };

    struct ggml_opt_context;

    // signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
    typedef void (*ggml_opt_epoch_callback)(
        bool               train,       // true after training evaluation, false after validation evaluation
        ggml_opt_context* opt_ctx,
        ggml_opt_dataset* dataset,
        ggml_opt_result* result,      // result associated with the dataset subsection
        int64_t            ibatch,      // number of batches that have been evaluated so far
        int64_t            ibatch_max,  // total number of batches in this dataset subsection
        int64_t            t_start_us); // time at which the evaluation on the dataset subsection was started

    struct ggml_opt_context {
        ggml_backend_sched*       backend_sched = nullptr;
        ggml_cgraph* allocated_graph = nullptr;
        ggml_cgraph* allocated_graph_copy = nullptr;

        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        ggml_context ctx_static;

        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ggml_context ctx_cpu;
        ggml_context* ctx_compute = nullptr;
        ggml_context ctx_copy;
        std::unique_ptr<ggml_backend_buffer>      buf_static;
        std::unique_ptr<ggml_backend_buffer>      buf_cpu;
        std::mt19937               rng;
        enum ggml_opt_loss_type    loss_type;
        enum ggml_opt_build_type   build_type;
        enum ggml_opt_build_type   build_type_alloc;

        ggml_tensor* inputs = nullptr;
        ggml_tensor* outputs = nullptr;
        ggml_tensor* labels = nullptr;

        ggml_tensor* loss = nullptr;
        ggml_tensor* pred = nullptr;
        ggml_tensor* ncorrect = nullptr;

        ggml_cgraph gf;
        ggml_cgraph gb_grad;
        ggml_cgraph gb_opt;
        ggml_cgraph dup_static_graph;
        bool static_graphs = false;
        bool eval_ready = false;
        std::vector<ggml_tensor*> grad_accs;
        std::vector<ggml_tensor*> grad_m;
        std::vector<ggml_tensor*> grad_v;

        int64_t iter = 1;
        int32_t opt_period = 1;
        int32_t opt_i = 0;
        bool    loss_per_datapoint = false;

        ggml_opt_get_optimizer_params get_opt_pars = nullptr;
        void* get_opt_pars_ud = nullptr;
        ggml_tensor* opt_step_params = nullptr; // Stores output of get_opt_pars.

        enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    private:
        bool is_static_graphs() const;
        void build();
    public:
        ggml_opt_context(ggml_opt_params params);
        void alloc(bool backward);
        ggml_tensor* get_loss() { return loss; }
        ggml_tensor* get_inputs() { return inputs; }
        ggml_tensor* get_labels() { return labels; }
        ggml_opt_optimizer_type get_optimizer_type() const { return optimizer; }
        void reset(bool optimizer);
        void eval(ggml_opt_result* result);
        ggml_tensor* get_grad_acc(ggml_tensor* node);

        // (result_train, result_eval)
        std::tuple<ggml_opt_result, ggml_opt_result> epoch(
            ggml_opt_dataset* dataset,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);
    };

    // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
    struct ggml_opt_optimizer_params {
        struct {
            float alpha; // learning rate
            float beta1; // first AdamW momentum
            float beta2; // second AdamW momentum
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay - 0.0f to disable
        } adamw;
        struct {
            float alpha; // learning rate
            float wd;    // weight decay
        } sgd;
    };

    ggml_tensor* ggml_opt_grad_acc(ggml_opt_context* opt_ctx, ggml_tensor* node);

    void ggml_opt_fit(
        ggml_backend_sched*           backend_sched,
        ggml_context* ctx_compute,
        ggml_tensor* inputs,
        ggml_tensor* outputs,
        ggml_opt_dataset* dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent);

    const char* ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o);
}
