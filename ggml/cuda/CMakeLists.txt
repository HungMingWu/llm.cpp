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
      op/acc.cu
      op/add-id.cu
      op/arange.cu
      op/argmax.cu
      op/argsort.cu
      op/binbcast.cu
      op/clamp.cu
      op/common.cuh
      op/compute_batched_ptrs.cu
      op/concat.cu
      op/conv-transpose-1d.cu
      op/convert.cu
      op/convert.cuh
      op/conv2d-dw.cu
      op/conv2d-transpose.cu
      op/count-equal.cu
      op/cpy.cu
      op/cross-entropy-loss.cu
      op/diagmask.cu
      op/fattn-tile-f16.cu
      op/fattn-tile-f32.cu
      op/fattn-wmma-f16.cu
      op/fattn.cu
      op/getrows.cu
      op/gla.cu
      op/im2col.cu
      op/mean.cu
      op/mmf.cu
      op/mmvf.cu
      op/mmvq.cu
      op/norm.cu
      op/opt-step-adamw.cu
      op/opt-step-sgd.cu
      op/pad.cu
      op/pool2d.cu
      op/quantize.cu
      op/roll.cu
      op/rope.cu
      op/scale.cu
      op/set-rows.cu
      op/softcap.cu
      op/softmax.cu
      op/ssm-conv.cu
      op/ssm-scan.cu
      op/sum.cu
      op/sumrows.cu
      op/tsembd.cu
      op/unary.cu
      op/upscale.cu
      op/wkv.cu
)

target_include_directories(libggml
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/
)

target_link_libraries(libggml PRIVATE CUDA::cudart_static CUDA::cuda_driver CUDA::cublas)
target_compile_options(libggml PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
