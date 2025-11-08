cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
set(CUDA_FLAGS --expt-relaxed-constexpr -extended-lambda -Xcudafe --diag_suppress=737)

file(READ "${CUDAToolkit_INCLUDE_DIRS}/cuda_runtime_api.h" CUDART_HEADER_CONTENTS)
string(REGEX MATCH "#define CUDART_VERSION[ \t]+([0-9]+)" _ ${CUDART_HEADER_CONTENTS})
set(CUDART_VERSION "${CMAKE_MATCH_1}")
message(STATUS "CUDART_VERSION: ${CUDART_VERSION}")

if (CUDART_VERSION LESS 10000)
    message(FATAL_ERROR  "Minimum cuda version needs 10")
endif()

if (NOT DEFINED GGML_USE_HIP AND NOT DEFINED GGML_USE_MUSA AND CUDART_VERSION GREATER_EQUAL 11010)
    set(CUDA_USE_CUB TRUE)
    target_compile_definitions(libggml PRIVATE GGML_CUDA_USE_CUB)
else()
    set(CUDA_USE_CUB FALSE)
endif()

target_sources(libggml
  PUBLIC
    FILE_SET primary_cuda_interface
    TYPE CXX_MODULES
    FILES
      backend.cppm
      buffer.cppm
      buffer_type.cppm
      registry.cppm
  PRIVATE
    FILE_SET cuda_hidden_interface
    TYPE CXX_MODULES
    FILES
      op.cppm
      fused.cppm
  PRIVATE
      backend.cpp
      buffer.cpp
      buffer_type.cpp
      common.cpp
      fused.cpp
      registry.cpp
      fused/moe-expert-reduce.cu
      fused/softcap.cu
      fused/topk-moe.cu
      common.cuh
      op/acc.cu
      op/add-id.cu
      op/arange.cu
      op/argmax.cu
      op/argsort.cu
      op/binbcast.cu
      op/clamp.cu
      op/compute_batched_ptrs.cu
      op/concat.cu
      op/conv-transpose-1d.cu
      op/convert.cu
      op/convert.cuh
      op/conv2d-dw.cu
      op/conv2d-transpose.cu
      op/conv2d.cu
      op/count-equal.cu
      op/cpy.cu
      op/cross-entropy-loss.cu
      op/diagmask.cu
      op/fattn-tile.cu
      op/fattn-wmma-f16.cu
      op/fattn.cu
      op/getrows.cu
      op/gla.cu
      op/im2col.cu
      op/mean_common.cu
      op/mmid.cu
      op/mmf.cu
      op/mmq.cu
      op/mmvf.cu
      op/mmvq.cu
      op/moe-expert-reduce.cu
      op/norm.cu
      op/opt-step-adamw.cu
      op/opt-step-sgd.cu
      op/pad.cu
      op/pad_reflect_1d.cu
      op/pool2d.cu
      op/quantize.cu
      op/roll.cu
      op/rope.cu
      op/scale.cu
      op/set-rows.cu
      op/softmax.cu
      op/ssm-conv.cu
      op/ssm-scan.cu
      op/sumrows.cu
      op/tsembd.cu
      op/unary.cu
      op/upscale.cu
      op/wkv.cu
)

if (CUDA_USE_CUB)
target_sources(libggml
PRIVATE
      op/argsort_cub.cu
      op/mean_cub.cu
      op/sum_cub.cu
)
else()
target_sources(libggml
PRIVATE
      op/argsort_fallback.cu
      op/mean_fallback.cu
      op/sum_fallback.cu
)
endif()

target_include_directories(libggml
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/
    ./
)

target_link_libraries(libggml PRIVATE CUDA::cudart_static CUDA::cuda_driver CUDA::cublas)
target_compile_options(libggml PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
