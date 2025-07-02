cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
set(CUDA_FLAGS --expt-relaxed-constexpr -Xcudafe --diag_suppress=737)

configure_file (${CMAKE_CURRENT_LIST_DIR}/cuda_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/cuda_config.h)
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
  PRIVATE
      backend.cpp
      buffer.cpp
      buffer_type.cpp
      common.cpp
      cu/acc.cu
      cu/arange.cu
      cu/argmax.cu
      cu/argsort.cu
      cu/binbcast.cu
      cu/clamp.cu
      cu/common.cuh
      cu/compute_batched_ptrs.cu
      cu/concat.cu
      cu/conv-transpose-1d.cu
      cu/convert.cu
      cu/convert.cuh
      cu/conv2d-dw.cu
      cu/conv2d-transpose.cu
      cu/count-equal.cu
      cu/cpy.cu
      cu/cross-entropy-loss.cu
      cu/diagmask.cu
      cu/fattn-tile-f16.cu
      cu/fattn-tile-f32.cu
      cu/fattn-wmma-f16.cu
      cu/fattn.cu
      cu/getrows.cu
      cu/gla.cu
      cu/im2col.cu
      cu/mean.cu
      cu/mmv.cu
      cu/mmvq.cu
      cu/norm.cu
      cu/opt-step-adamw.cu
      cu/pad.cu
      cu/pool2d.cu
      cu/quantize.cu
      cu/rope.cu
      cu/scale.cu
      cu/softmax.cu
      cu/ssm-conv.cu
      cu/ssm-scan.cu
      cu/sum.cu
      cu/sumrows.cu
      cu/tsembd.cu
      cu/unary.cu
      cu/upscale.cu
      cu/wkv.cu
)

target_include_directories(libggml
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/
)

target_link_libraries(libggml PRIVATE CUDA::cudart_static CUDA::cuda_driver CUDA::cublas)
target_compile_options(libggml PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
