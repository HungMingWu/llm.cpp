module;

module ggml:cuda.config;
#ifdef USE_CUDA_GRAPH
constexpr bool use_cuda_graph_v = true;
#else
constexpr bool use_cuda_graph_v = false;
#endif

#ifdef GGML_CUDA_FORCE_CUBLAS
constexpr bool force_enable_cuda_blas_v = true;
#else
constexpr bool force_enable_cuda_blas_v = false;
#endif

#ifdef GGML_CUDA_FORCE_MMQ
constexpr bool force_enable_cuda_mmq_v = true;
#else
constexpr bool force_enable_cuda_mmq_v = false;
#endif