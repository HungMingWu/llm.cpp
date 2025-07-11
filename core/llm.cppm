module;
#include <filesystem>
#include <variant>
#include <vector>

export module llm;
import ggml;

export
{
    constexpr uint32_t LLAMA_DEFAULT_SEED = 0xFFFFFFFF;

    // scheduling priorities
    enum ggml_sched_priority {
        GGML_SCHED_PRIO_LOW = -1,
        GGML_SCHED_PRIO_NORMAL,
        GGML_SCHED_PRIO_MEDIUM,
        GGML_SCHED_PRIO_HIGH,
        GGML_SCHED_PRIO_REALTIME
    };

    struct cpu_params {
        int      n_threads = -1;
        bool     cpumask[GGML_MAX_N_THREADS] = { false }; // CPU affinity mask.
        bool     mask_valid = false;   // Default: any CPU
        enum ggml_sched_priority  priority = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
        bool     strict_cpu = false;   // Use strict CPU placement
        uint32_t poll = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
    };

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE = 2,
        GGML_NUMA_STRATEGY_NUMACTL = 3,
        GGML_NUMA_STRATEGY_MIRROR = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    enum common_sampler_type {
        COMMON_SAMPLER_TYPE_NONE = 0,
        COMMON_SAMPLER_TYPE_DRY = 1,
        COMMON_SAMPLER_TYPE_TOP_K = 2,
        COMMON_SAMPLER_TYPE_TOP_P = 3,
        COMMON_SAMPLER_TYPE_MIN_P = 4,
        //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
        COMMON_SAMPLER_TYPE_TYPICAL_P = 6,
        COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
        COMMON_SAMPLER_TYPE_XTC = 8,
        COMMON_SAMPLER_TYPE_INFILL = 9,
        COMMON_SAMPLER_TYPE_PENALTIES = 10,
    };

    using llama_token = int32_t;

    struct llama_logit_bias {
        llama_token token;
        float bias;
    };

    // sampling parameters
    struct common_params_sampling {
        uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

        int32_t n_prev = 64;    // number of previous tokens to remember
        int32_t n_probs = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
        int32_t min_keep = 0;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
        int32_t top_k = 40;    // <= 0 to use vocab size
        float   top_p = 0.95f; // 1.0 = disabled
        float   min_p = 0.05f; // 0.0 = disabled
        float   xtc_probability = 0.00f; // 0.0 = disabled
        float   xtc_threshold = 0.10f; // > 0.5 disables XTC
        float   typ_p = 1.00f; // typical_p, 1.0 = disabled
        float   temp = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities
        float   dynatemp_range = 0.00f; // 0.0 = disabled
        float   dynatemp_exponent = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
        int32_t penalty_last_n = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
        float   penalty_repeat = 1.00f; // 1.0 = disabled
        float   penalty_freq = 0.00f; // 0.0 = disabled
        float   penalty_present = 0.00f; // 0.0 = disabled
        float   dry_multiplier = 0.0f;  // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
        float   dry_base = 1.75f; // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
        int32_t dry_allowed_length = 2;     // tokens extending repetitions beyond this receive penalty
        int32_t dry_penalty_last_n = -1;    // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        int32_t mirostat = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float   mirostat_tau = 5.00f; // target entropy
        float   mirostat_eta = 0.10f; // learning rate
        bool    ignore_eos = false;
        bool    no_perf = false; // disable performance metrics
        bool    timing_per_token = false;

        std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };     // default sequence breakers for DRY


        std::vector<enum common_sampler_type> samplers = {
            COMMON_SAMPLER_TYPE_PENALTIES,
            COMMON_SAMPLER_TYPE_DRY,
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TYPICAL_P,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_MIN_P,
            COMMON_SAMPLER_TYPE_XTC,
            COMMON_SAMPLER_TYPE_TEMPERATURE,
        };

        std::string grammar; // optional BNF-like grammar to constrain sampling

        std::vector<llama_logit_bias> logit_bias; // logit biases to apply

        // print the parameters into a string
        std::string print() const;
    };

    struct common_params_speculative {
        std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

        int32_t n_ctx = 0; // draft context size
        int32_t n_max = 16; // maximum number of tokens to draft during speculative decoding
        int32_t n_min = 5; // minimum number of draft tokens to use for speculative decoding
        int32_t n_gpu_layers = -1; // number of layers to store in VRAM for the draft model (-1 - use default)
        float   p_split = 0.1f; // speculative decoding split probability
        float   p_min = 0.9f; // minimum speculative decoding probability (greedy)

        struct cpu_params cpuparams;
        struct cpu_params cpuparams_batch;

        std::string model = ""; // draft model for speculative decoding                          // NOLINT
    };


    struct common_params_vocoder {
        std::string hf_repo = ""; // HF repo                                                     // NOLINT
        std::string hf_file = ""; // HF file                                                     // NOLINT

        std::string model = ""; // model path                                                // NOLINT
        std::string model_url = ""; // model url to download                                     // NOLINT
    };

    struct llama_model_kv_override {
        char key[128];

        std::variant<int64_t, double, bool, std::string> val;
    };

    struct common_lora_adapter_info {
        std::string path;
        float scale;
    };

    struct common_lora_adapter_container : common_lora_adapter_info {
        struct llama_lora_adapter* adapter;
    };

    struct common_control_vector_load_info {
        float strength;

        std::string fname;
    };

    // dimensionality reduction methods, used by cvector-generator
    enum dimre_method {
        DIMRE_METHOD_PCA,
        DIMRE_METHOD_MEAN,
    };

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1,
    };

    struct common_init_result {
        struct llama_model* model = nullptr;
        struct llama_context* context = nullptr;
        std::vector<common_lora_adapter_container> lora_adapters;
    };

    struct common_params {
        int32_t n_predict = -1; // new tokens to predict
        int32_t n_ctx = 4096; // context size
        int32_t n_batch = 2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_ubatch = 512; // physical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_keep = 0; // number of tokens to keep from initial prompt
        int32_t n_chunks = -1; // max number of chunks to process (-1 = unlimited)
        int32_t n_parallel = 1; // number of parallel sequences to decode
        int32_t n_sequences = 1; // number of sequences to decode
        int32_t grp_attn_n = 1; // group-attention factor
        int32_t grp_attn_w = 512; // group-attention width
        int32_t n_print = -1; // print token count every n tokens (-1 = disabled)
        float   rope_freq_base = 0.0f; // RoPE base frequency
        float   rope_freq_scale = 0.0f; // RoPE frequency scaling factor
        float   yarn_ext_factor = -1.0f; // YaRN extrapolation mix factor
        float   yarn_attn_factor = 1.0f; // YaRN magnitude scaling factor
        float   yarn_beta_fast = 32.0f; // YaRN low correction dim
        float   yarn_beta_slow = 1.0f; // YaRN high correction dim
        int32_t yarn_orig_ctx = 0; // YaRN original context length
        float   defrag_thold = 0.1f; // KV cache defragmentation threshold

        // offload params
        std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

        int32_t n_gpu_layers = -1;  // number of layers to store in VRAM (-1 - use default)
        int32_t main_gpu = 0;   // the GPU that is used for scratch and small tensors
        float   tensor_split[128] = { 0 }; // how split tensors should be distributed across GPUs

        enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs

        struct cpu_params cpuparams;
        struct cpu_params cpuparams_batch;

        ggml_backend_sched_eval_callback cb_eval = nullptr;

        ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

        enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        enum llama_pooling_type      pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
        enum llama_attention_type    attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings

        struct common_params_sampling    sampling;
        struct common_params_speculative speculative;
        struct common_params_vocoder     vocoder;

        std::string model = ""; // model path                                                    // NOLINT
        std::string model_alias = ""; // model alias                                                   // NOLINT
        std::string model_url = ""; // model url to download                                         // NOLINT
        std::string hf_token = ""; // HF token                                                      // NOLINT
        std::string hf_repo = ""; // HF repo                                                       // NOLINT
        std::string hf_file = ""; // HF file                                                       // NOLINT
        std::string prompt = "";                                                                  // NOLINT
        std::string prompt_file = ""; // store the external prompt file name                           // NOLINT
        std::string path_prompt_cache = ""; // path to file for saving/loading prompt eval state             // NOLINT
        std::string input_prefix = ""; // string to prefix user inputs with                             // NOLINT
        std::string input_suffix = ""; // string to suffix user inputs with                             // NOLINT
        std::string lookup_cache_static = ""; // path of static ngram cache file for lookup decoding           // NOLINT
        std::string lookup_cache_dynamic = ""; // path of dynamic ngram cache file for lookup decoding          // NOLINT
        std::string logits_file = ""; // file for saving *all* logits                                  // NOLINT
        std::string rpc_servers = ""; // comma separated list of RPC servers                           // NOLINT

        std::vector<std::string> in_files;   // all input files
        std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
        std::vector<llama_model_kv_override> kv_overrides;

        bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
        std::vector<common_lora_adapter_info> lora_adapters; // lora adapter path with user defined scale

        std::vector<common_control_vector_load_info> control_vectors; // control vector with user defined scale

        int32_t verbosity = 0;
        int32_t control_vector_layer_start = -1; // layer range for control vector
        int32_t control_vector_layer_end = -1; // layer range for control vector

        int32_t ppl_stride = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
        int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
        //                                       (which is more convenient to use for plotting)
        //
        bool   hellaswag = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
        size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

        bool   winogrande = false; // compute Winogrande score over random tasks from datafile supplied in prompt
        size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

        bool   multiple_choice = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
        size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

        bool   kl_divergence = false; // compute KL divergence

        bool usage = false; // print usage
        bool use_color = false; // use color to distinguish generations and inputs
        bool special = false; // enable special token output
        bool interactive = false; // interactive mode
        bool interactive_first = false; // wait for user input immediately
        bool conversation = false; // conversation mode (does not print special tokens and suffix/prefix)
        bool prompt_cache_all = false; // save user input and generations to prompt cache
        bool prompt_cache_ro = false; // open the prompt cache read-only and do not update it

        bool escape = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
        bool multiline_input = false; // reverse the usage of `\`
        bool simple_io = false; // improves compatibility with subprocesses and limited consoles
        bool cont_batching = true;  // insert new sequences for decoding on-the-fly
        bool flash_attn = false; // flash attention
        bool no_perf = false; // disable performance metrics
        bool ctx_shift = true;  // context shift on inifinite text generation

        bool input_prefix_bos = false; // prefix BOS to user inputs, preceding input_prefix
        bool logits_all = false; // return logits for all tokens in the batch
        bool use_mmap = true;  // use mmap for faster loads
        bool use_mlock = false; // use mlock to keep model in memory
        bool verbose_prompt = false; // print prompt tokens before generation
        bool display_prompt = true;  // print prompt before generation
        bool dump_kv_cache = false; // dump the KV cache contents for debugging purposes
        bool no_kv_offload = false; // disable KV offloading
        bool warmup = true;  // warmup run
        bool check_tensors = false; // validate tensor data

        ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
        ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V

        // multimodal models (see examples/llava)
        std::string mmproj = "";        // path to multimodal projector                                         // NOLINT
        std::vector<std::string> image; // path to image file(s)

        // embedding
        bool embedding = false; // get only sentence embedding
        int32_t embd_normalize = 2;     // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
        std::string embd_out = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
        std::string embd_sep = "\n";  // separator of embeddings
        bool reranking = false; // enable reranking support on server

        // server params
        int32_t port = 8080;         // server listens on this network port
        int32_t timeout_read = 600;          // http read timeout in seconds
        int32_t timeout_write = timeout_read; // http write timeout in seconds
        int32_t n_threads_http = -1;           // number of threads to process HTTP requests (TODO: support threadpool)
        int32_t n_cache_reuse = 0;            // min chunk size to reuse from the cache via KV shifting

        std::string hostname = "127.0.0.1";
        std::string public_path = "";                                                                         // NOLINT
        std::string chat_template = "";                                                                         // NOLINT
        bool enable_chat_template = true;

        std::vector<std::string> api_keys;

        std::string ssl_file_key = "";                                                                         // NOLINT
        std::string ssl_file_cert = "";                                                                         // NOLINT

        // "advanced" endpoints are disabled by default for better security
        bool webui = true;
        bool endpoint_slots = false;
        bool endpoint_props = false; // only control POST requests, not GET
        bool endpoint_metrics = false;

        bool log_json = false;

        std::string slot_save_path;

        float slot_prompt_similarity = 0.5f;

        // batched-bench params
        bool is_pp_shared = false;

        std::vector<int32_t> n_pp;
        std::vector<int32_t> n_tg;
        std::vector<int32_t> n_pl;

        // retrieval params
        std::vector<std::string> context_files; // context files to embed

        int32_t chunk_size = 64; // chunk size for context embedding

        std::string chunk_separator = "\n"; // chunk separator for context embedding

        // passkey params
        int32_t n_junk = 250; // number of times to repeat the junk text
        int32_t i_pos = -1;  // position of the passkey in the junk text

        // imatrix params
        std::string out_file = "imatrix.dat"; // save the resulting imatrix to this file

        int32_t n_out_freq = 10; // output the imatrix every n_out_freq iterations
        int32_t n_save_freq = 0; // save the imatrix every n_save_freq iterations
        int32_t i_chunk = 0; // start processing from this chunk

        bool process_output = false; // collect data for the output tensor
        bool compute_ppl = true;  // whether to compute perplexity

        // cvector-generator params
        int n_pca_batch = 100;
        int n_pca_iterations = 1000;
        dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
        std::string cvector_outfile = "control_vector.gguf";
        std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
        std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

        bool spm_infill = false; // suffix/prefix/middle pattern for infill

        std::string lora_outfile = "ggml-lora-merged-f16.gguf";

        // batched-bench params
        bool batched_bench_output_jsonl = false;
    };

    common_init_result common_init_from_params(common_params& params);
}
