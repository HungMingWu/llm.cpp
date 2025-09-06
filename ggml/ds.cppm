module;
#include <stdint.h>
#include <array>
#include <condition_variable>
#include <format>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "internal_ds.h"
#include "inplace_vector.hpp"

export module ggml:ds;
constexpr size_t GGML_MAX_OP_PARAMS = 64;
constexpr size_t GGML_MAX_SRC = 10;

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

export {
    using ggml_type = ggml_type;
    using ggml_prec = ggml_prec;
    using ggml_op_pool = ggml_op_pool;
    using ggml_sort_order = ggml_sort_order;
    using ggml_scale_mode = ggml_scale_mode;
    using ggml_scale_flag = ggml_scale_flag;

    constexpr size_t GGML_MAX_DIMS = 4;
    constexpr size_t GGML_MAX_NAME = 64;
    constexpr size_t GGML_DEFAULT_N_THREADS = 4;
    constexpr size_t GGML_MAX_N_THREADS = 512;
    constexpr uint32_t GGML_KQ_MASK_PAD = 64;

    // required for mmap as gguf only guarantees 32-byte alignment
    constexpr uint32_t TENSOR_ALIGNMENT = 32;

    constexpr size_t GGML_MROPE_SECTIONS = 4;

    // this tensor...
    enum ggml_tensor_flag {
        GGML_TENSOR_FLAG_INPUT = 1, // ...is an input for the GGML compute graph
        GGML_TENSOR_FLAG_OUTPUT = 2, // ...is an output for the GGML compute graph
        GGML_TENSOR_FLAG_PARAM = 4, // ...contains trainable parameters
        GGML_TENSOR_FLAG_LOSS = 8, // ...defines loss for numerical optimization (multiple loss tensors add up)
    };

    // available tensor operations:
    enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD_ID,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_COUNT_EQUAL,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_L2_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_SET_ROWS,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_IM2COL_BACK,
        GGML_OP_IM2COL_3D,
        GGML_OP_CONV_2D,
        GGML_OP_CONV_3D,
        GGML_OP_CONV_2D_DW,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_POOL_2D_BACK,
        GGML_OP_UPSCALE,
        GGML_OP_PAD,
        GGML_OP_PAD_REFLECT_1D,
        GGML_OP_ROLL,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_LEAKY_RELU,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_RWKV_WKV6,
        GGML_OP_GATED_LINEAR_ATTN,
        GGML_OP_RWKV_WKV7,

        GGML_OP_UNARY,

        GGML_OP_CUSTOM,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        GGML_OP_OPT_STEP_ADAMW,
        GGML_OP_OPT_STEP_SGD,

        GGML_OP_GLU,

        GGML_OP_COUNT,
    };

    // model file types
    enum ggml_ftype {
        GGML_FTYPE_UNKNOWN = -1,
        GGML_FTYPE_ALL_F32 = 0,
        GGML_FTYPE_MOSTLY_F16 = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XS = 16, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_S = 18, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_NL = 19, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_S = 20, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_S = 21, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_XS = 22, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_M = 23, // except 1d tensors
        GGML_FTYPE_MOSTLY_BF16 = 24, // except 1d tensors
        GGML_FTYPE_MOSTLY_MXFP4 = 25, // except 1d tensors
    };

    enum ggml_glu_op {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU,
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_SWIGLU_OAI,
        GGML_GLU_OP_GEGLU_ERF,
        GGML_GLU_OP_GEGLU_QUICK,

        GGML_GLU_OP_COUNT,
    };

    //
    // Backend device
    //
    enum ggml_backend_dev_type {
        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL
    };

    enum ggml_backend_type {
        GGML_BACKEND_TYPE_CPU = 0,
        GGML_BACKEND_TYPE_GPU = 10,
        GGML_BACKEND_TYPE_GPU_SPLIT = 20,
    };

    using ggml_backend_score_t = int (*)();

    using ggml_backend_reg_t = struct ggml_backend_reg*;
    using ggml_backend_init_t = ggml_backend_reg_t(*)();

    struct ggml_tensor;
    struct tensor_traits;
    struct ggml_context;
    struct ggml_backend_device;

    // Use for reflection later
    struct ggml_backend_reg {
        int api_version; // initialize to GGML_BACKEND_API_VERSION
        void* context;
    public:
        ggml_backend_reg(int api_version, void* context) : api_version(api_version), context(context) {}
        virtual ~ggml_backend_reg() = default;
        virtual std::string_view get_name() = 0;
        virtual std::span<ggml_backend_device*> get_devices() = 0;
        virtual size_t get_device_count() { return 1; }
        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        virtual void* get_proc_address(std::string_view name) { return nullptr; }
    };

    struct ggml_backend;
    struct ggml_backend_buffer;
    struct ggml_backend_buffer_type;
    struct ggml_backend_event;

    class ggml_backend_device {
        ggml_backend_reg_t reg;
    public:
        ggml_backend_device(ggml_backend_reg_t reg) : reg(reg) {}
        virtual ~ggml_backend_device() = default;
        ggml_backend_reg_t get_backend_reg() { return reg; }
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        virtual const char* get_name() = 0;
        // device description: short informative description of the device, could be the model name
        virtual const char* get_description() = 0;
        // device memory in bytes
        virtual void get_memory(size_t* free, size_t* total) = 0;
        // device type
        virtual enum ggml_backend_dev_type get_type() = 0;
        // device properties
        virtual void get_props(struct ggml_backend_dev_props* props) = 0;
        // backend (stream) initialization
        virtual std::unique_ptr<ggml_backend> init_backend(const char* params) = 0;
        // preferred buffer type
        virtual ggml_backend_buffer_type* get_buffer_type() = 0;
        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        virtual ggml_backend_buffer_type* get_host_buffer_type()
        {
            return {};
        }
        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        virtual ggml_backend_buffer* buffer_from_host_ptr(void* ptr, size_t size, size_t max_tensor_size)
        {
            return {};
        }
        // check if the backend can compute an operation
        virtual bool supports_op(const ggml_tensor* op) = 0;
        // check if the backend can use tensors allocated in a buffer type
        virtual bool supports_buft(ggml_backend_buffer_type* buft) = 0;
        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        virtual bool offload_op(const ggml_tensor* op)
        {
            return false;
        }
        // (optional) event synchronization
        virtual ggml_backend_event* event_new()
        {
            return nullptr;
        }
        virtual void event_free(ggml_backend_event* event)
        {

        }
        virtual void event_synchronize(ggml_backend_event* event)
        {

        }
        // (optional) get extra bufts
        virtual std::span<ggml_backend_buffer_type*> get_extra_bufts() {
            return {};
        }
    };

    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    enum ggml_status {
        GGML_STATUS_ALLOC_FAILED = -2,
        GGML_STATUS_FAILED = -1,
        GGML_STATUS_SUCCESS = 0,
        GGML_STATUS_ABORTED = 1,
    };

    using ggml_custom_op_cb = std::function<void(ggml_tensor*, int, int)>;

    struct custom_op_hook {
        ggml_custom_op_cb func;
        std::optional<uint32_t> n_tasks;
    };

    // n-dimensional tensor
    struct ggml_tensor {
        ggml_type type;

        ggml_backend_buffer* buffer = nullptr;

        std::array<int64_t, GGML_MAX_DIMS> ne { 1, 1, 1, 1 }; // number of elements
        std::array<size_t, GGML_MAX_DIMS> nb { 0, 0, 0, 0 }; // stride in bytes:
        // nb[0] = ggml_type_size(type)
        // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
        // nb[i] = nb[i-1] * ne[i-1]

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]{ 0 };
        custom_op_hook hook;

        int32_t flags = 0;

        // source tensor and offset for views
        ggml_tensor* view_src = nullptr;
        size_t view_offs = 0;

        void* data = nullptr;

        std::string name;

        void* extra = nullptr; // extra things e.g. for ggml-cuda.cu

        // compute data
        ggml_op op = GGML_OP_NONE;

        cpp26::inplace_vector<ggml_tensor*, GGML_MAX_SRC> src;

        char padding[8] { 0 };
    public:
        std::string_view get_name() const { return name; }

        template<typename... Args>
        auto set_name(std::format_string<Args...> fmt, Args&&... args)
        {
            name = std::vformat(fmt.get(), std::make_format_args(args...));
        }

        void set_name(std::string new_name)
        {
            //assert(new_name.length() < GGML_MAX_NAME);
            name = std::move(new_name);
        }

        size_t nbytes() const;
        int64_t nelements() const;
        void set_flag(int32_t flag);
    };

    struct ggml_cgraph;

    struct ggml_context {
    private:        
        ggml_tensor* create_new_tensor_impl(
            ggml_type type,
            std::span<int64_t> ne,
            struct ggml_tensor* view_src,
            size_t  view_offs);
    public:
        std::list<ggml_tensor*> tensors;

        explicit ggml_context();
        ~ggml_context();

        template <typename Self>
        auto& getTensors(this Self&& self) { return self.tensors; }
        ggml_tensor* create(ggml_type type, std::initializer_list<int64_t> ne);
        ggml_tensor* create(ggml_type type, std::initializer_list<int64_t> ne, ggml_tensor* view_src, size_t view_offset);
		ggml_tensor* find(std::string_view name);
    };

    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
        GGML_UNARY_OP_GELU_ERF,

        GGML_UNARY_OP_COUNT,
    };

    constexpr int GGML_ROPE_TYPE_NEOX = 2;
    constexpr int GGML_ROPE_TYPE_MROPE = 8;
    constexpr int GGML_ROPE_TYPE_VISION = 24;

    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    // all the device properties
    struct ggml_backend_dev_props {
        // device name
        const char* name;
        // device description
        const char* description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum ggml_backend_dev_type type;
        // device id
        //   for PCI devices, this should be the PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:01:00.0")
        //   if the id is unknown, this should be NULL
        const char* device_id;
        // device capabilities
        struct ggml_backend_dev_caps caps;
    };

    // dynamic tensor allocator

    struct free_block {
        size_t offset;
        size_t size;
    };

    constexpr size_t MAX_FREE_BLOCKS = 256;

    struct ggml_dyn_tallocr {
        size_t alignment;
        int n_free_blocks = 0;
        free_block free_blocks[MAX_FREE_BLOCKS] {};
        size_t max_size = 0;

#ifdef GGML_ALLOCATOR_DEBUG
        struct {
            const struct ggml_tensor* tensor;
            size_t offset;
        } allocated_tensors[1024]{};
#endif
    public:
        ggml_dyn_tallocr(size_t alignment);
        size_t get_max_size() const { return max_size; }
        void reset();
        size_t alloc(size_t size, const ggml_tensor* tensor);
        void free_tensor(size_t offset, size_t size, const ggml_tensor* tensor);
    };

    struct hash_node {
        int n_children;
        int n_views;
        int buffer_id;
        size_t offset; // offset within the buffer
        bool allocated;
    };

    struct tensor_alloc {
        int buffer_id{};
        size_t offset{};
        size_t size_max{}; // 0 = pre-allocated, unused, or view
    };

    struct leaf_alloc {
        tensor_alloc leaf;
    };

    struct node_alloc {
        tensor_alloc dst;
        tensor_alloc src[GGML_MAX_SRC];
    };

    struct ggml_gallocr {
    private:
        std::vector<ggml_backend_buffer_type*> bufts; // [n_buffers]
        std::vector<std::shared_ptr<ggml_backend_buffer>> buffers; // [n_buffers]
        std::vector<std::shared_ptr<ggml_dyn_tallocr>> buf_tallocs; // [n_buffers]
        std::unordered_map<ggml_tensor*, hash_node> hash_map;
        std::vector<node_alloc> node_allocs; // [n_nodes]
        std::vector<leaf_alloc> leaf_allocs; // [n_leafs]
        bool is_allocated(ggml_tensor* t);
        bool is_own(ggml_tensor* t);
        void free_node(ggml_tensor* node);
        void allocate_node(ggml_tensor* node, int buffer_id);
        bool needs_realloc(const ggml_cgraph& graph);
        bool node_needs_realloc(ggml_tensor* node, tensor_alloc* talloc);
        void init_tensor(ggml_tensor* tensor, tensor_alloc* tensor_alloc);
        void alloc_graph_impl(const ggml_cgraph &graph,
            std::span<const int> node_buffer_ids, std::span<const int> leaf_buffer_ids);
    public:
        ggml_gallocr(std::span<ggml_backend_buffer_type*> bufts);
        ggml_gallocr(ggml_backend_buffer_type* buft) :
            ggml_gallocr(std::span<ggml_backend_buffer_type*>{ &buft, 1 }) {}
        ~ggml_gallocr() = default;
        size_t get_buffer_size(int buffer_id);
        bool alloc_graph(ggml_cgraph* graph);
        bool reserve(const ggml_cgraph& graph, std::span<const int> node_buffer_ids, std::span<const int> leaf_buffer_ids);
        bool reserve(const ggml_cgraph* graph);
    };

    using ggml_bitset_t = uint32_t;
    static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
    constexpr size_t BITSET_SHR = 5; // log2(sizeof(ggml_bitset_t)*8)
    constexpr size_t BITSET_MASK = (sizeof(ggml_bitset_t) * 8 - 1);

    size_t ggml_bitset_size(size_t n) {
        return (n + BITSET_MASK) >> BITSET_SHR;
    }

    inline bool ggml_bitset_get(const ggml_bitset_t* bitset, size_t i) {
        return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
    }

    inline void ggml_bitset_set(ggml_bitset_t* bitset, size_t i) {
        bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
    }

    inline void ggml_bitset_clear(ggml_bitset_t* bitset, size_t i) {
        bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
    }

    // computation graph

    enum ggml_cgraph_eval_order {
        GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        GGML_CGRAPH_EVAL_ORDER_COUNT
    };

    struct ggml_cgraph {
    public: // Use for develop
        std::vector<ggml_tensor*> nodes;     // tensors with data that can change if the graph is evaluated
        std::unordered_map<const ggml_tensor*, ggml_tensor*> grads;     // the outputs of these tensors are the gradients of the nodes
        std::unordered_map<const ggml_tensor*, ggml_tensor*> grad_accs; // accumulators for node gradients
        std::vector<ggml_tensor*> leafs;     // tensors with constant data
        std::unordered_map<const ggml_tensor*, int32_t> use_counts; // number of uses of each tensor
        std::unordered_set<ggml_tensor*> visited_hash_set;

        enum ggml_cgraph_eval_order order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
        void visit_parents(ggml_tensor*);
    public:
        void build_forward_expand(ggml_tensor*);
        void build_backward_expand(ggml_context*, std::span<ggml_tensor*> grad_accs = {});
        void add_node(ggml_tensor*);
        size_t getGraphSize() const { return std::max(nodes.size(), leafs.size()); }
        std::span<ggml_tensor*> getNodes() { return nodes; }
        void reset();
        ggml_tensor* get_tensor(std::string_view name);
    };

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    using ggml_backend_sched_eval_callback = std::function<bool(ggml_tensor* t, bool ask)>;

    struct ggml_backend_sched_split {
        int backend_id;
        int i_start;
        int i_end;
        cpp26::inplace_vector<ggml_tensor*, GGML_SCHED_MAX_SPLIT_INPUTS> inputs;
        // graph view of this split
        ggml_cgraph graph;
    };

    struct ggml_backend_sched {
        bool is_reset = false; // true if the scheduler has been reset since the last graph split
        bool is_alloc = false;

        int n_backends;

        ggml_backend* backends[GGML_SCHED_MAX_BACKENDS];
        ggml_backend_buffer_type* bufts[GGML_SCHED_MAX_BACKENDS];
        std::unique_ptr<ggml_gallocr> galloc;

        // hash map of the nodes in the graph
        std::unordered_map<ggml_tensor*, int> hv_tensor_backend_ids; // [hash_set.size]
        std::unordered_map<ggml_tensor*,
            std::unordered_map<int, 
                std::vector<ggml_tensor*>>> hv_tensor_copies; // [hash_set.size][n_backends][n_copies]  

        std::vector<int> node_backend_ids; // [graph_size]
        std::vector<int> leaf_backend_ids; // [graph_size]

        std::vector<int> prev_node_backend_ids; // [graph_size]
        std::vector<int> prev_leaf_backend_ids; // [graph_size]

        // copy of the graph with modified inputs
        ggml_cgraph graph;

        // graph splits
        std::list<ggml_backend_sched_split> splits;

        // pipeline parallelism support
        int n_copies;
        int cur_copy = 0;

        std::array<
            std::array<std::unique_ptr<ggml_backend_event>, GGML_SCHED_MAX_COPIES>,
                GGML_SCHED_MAX_BACKENDS> events;

        cpp26::inplace_vector<ggml_tensor*, GGML_SCHED_MAX_SPLIT_INPUTS> graph_inputs;

        std::unique_ptr<ggml_context> ctx;

        ggml_backend_sched_eval_callback callback_eval;

        bool op_offload;

        int debug;
    private:
        int backend_from_buffer(const ggml_tensor* tensor, const ggml_tensor* op);
        int backend_id_from_cur(ggml_tensor* tensor);
        bool buffer_supported(ggml_tensor* t, int backend_id);
        ggml_backend* get_tensor_backend(ggml_tensor* node);
        std::optional<int> get_backend_id(ggml_backend* backend);
        // Split graph without allocating it
        void split_graph(const ggml_cgraph& graph);
        void print_assignments(const ggml_cgraph& graph);
        ggml_status graph_compute_async(const ggml_cgraph& graph);
        bool alloc_splits();
        ggml_status compute_splits();
    public:
        ggml_backend_sched(ggml_backend** backends,
            ggml_backend_buffer_type** bufts,
            int n_backends,
            bool parallel,
            bool op_offload);
        void reset();
        size_t get_buffer_size(ggml_backend* backend);
        bool reserve(const ggml_cgraph* measure_graph);
        ggml_status graph_compute(const ggml_cgraph& graph);
        void set_eval_callback(ggml_backend_sched_eval_callback callback) {
            callback_eval = callback;
        }
        void set_tensor_backend(ggml_tensor* node, ggml_backend* backend);
        void dump_dot(const ggml_cgraph* graph, const char* filename);
        // not sure move to public is right direction
        bool alloc_graph(const ggml_cgraph& graph);
        void synchronize();
        ggml_backend* get_backend(int i) { return backends[i]; }
    };

    // GUID types
    using ggml_guid = uint8_t[16];
    using ggml_guid_t = ggml_guid*;

    //
    // Backend (stream)
    //
    struct ggml_backend {
    private:
        ggml_backend_device* device;
    public:
        ggml_guid_t guid;
    protected:
        virtual void set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size);
        virtual void get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size);

        // compute graph (always async if supported by the backend)
        virtual ggml_status graph_compute_impl(ggml_cgraph* cgraph) = 0;
    public:
        ggml_backend(ggml_backend_device* device) : device(device) {}
        virtual ~ggml_backend() = default;
        virtual const char* get_name() = 0;

        void set_tensor_async(ggml_tensor* tensor, const void* data, size_t offset, size_t size);
        void get_tensor_async(const ggml_tensor* tensor, void* data, size_t offset, size_t size);
        virtual bool cpy_tensor_async(ggml_backend* backend_src, const ggml_tensor* src, ggml_tensor* dst) { return false; }

        // (optional) complete all pending operations (required if the backend supports async operations)
        virtual void synchronize() {}

        ggml_status graph_compute(ggml_cgraph* cgraph);
        std::unique_ptr<ggml_backend_buffer> alloc_tensors(const ggml_context* ctx);

        // (optional) event synchronization
        // record an event on this stream
        virtual void event_record(ggml_backend_event* event) {}
        // wait for an event on on a different stream
        virtual void event_wait(ggml_backend_event* event) {}

        ggml_status compute(ggml_cgraph* cgraph)
        {
            ggml_status err = graph_compute(cgraph);
            synchronize();
            return err;
        }

        virtual void graph_optimize(ggml_cgraph* cgraph) {}

        // helper function
        ggml_backend_device* get_device() const
        {
            return device;
        }
        ggml_backend_buffer_type* get_default_buffer_type() const
        {
            return device->get_buffer_type();
        }
        bool supports_buft(ggml_backend_buffer_type* buft) const
        {
            return device->supports_buft(buft);
        }
        bool supports_op(const ggml_tensor* op)
        {
            return device->supports_op(op);
        }
        bool offload_op(const ggml_tensor* op)
        {
            return device->offload_op(op);
        }

        std::unique_ptr<ggml_backend_buffer> alloc_buffer(size_t size);
    };

    struct ggml_threadpool;     // forward declaration, see ggml.c

    using ggml_threadpool_t = struct ggml_threadpool*;

    // Abort callback
    // If not NULL, called before ggml computation
    // If it returns true, the computation is aborted
    using ggml_abort_callback = std::function<bool()>;

    // Set the abort callback for the backend
    using ggml_backend_set_abort_callback_t = void (*)(ggml_backend* backend, ggml_abort_callback abort_callback);

    // Tensor allocator
    struct ggml_tallocr {
        ggml_backend_buffer* buffer;
        void* base;
        size_t alignment;
        size_t offset;
    public:
        ggml_tallocr(ggml_backend_buffer*);
        void alloc(ggml_tensor* tensor);
    };

    struct ggml_backend_event {
        ggml_backend_device* device;
        void* context;

        void synchronize() {
            return device->event_synchronize(this);
        }
    };

    using ggml_backend_eval_callback = std::function<bool(ggml_tensor*, ggml_tensor*)>;
}
