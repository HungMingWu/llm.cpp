cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
set(CUDA_FLAGS --expt-relaxed-constexpr -Xcudafe --diag_suppress=737)

target_sources(libggml
  PUBLIC
    FILE_SET primary_cuda_interface
    TYPE CXX_MODULES
    FILES
      backend.cppm
      buffer.cppm
      registry.cppm
  PRIVATE
    FILE_SET cuda_hidden_interface
    TYPE CXX_MODULES
    FILES
      op.cppm
      config.cppm
  PRIVATE
      backend.cpp
      buffer.cpp
      common.cpp
      cu/arange.cu
      cu/argmax.cu
      cu/binbcast.cu
      cu/clamp.cu
      cu/common.cuh
      cu/compute_batched_ptrs.cu
      cu/conv-transpose-1d.cu
      cu/convert.cu
      cu/convert.cuh
      cu/count-equal.cu
      cu/cpy.cu
      cu/diagmask.cu
      cu/getrows.cu
      cu/gla.cu
      cu/im2col.cu
      cu/misc.cu
      cu/mmv.cu
      cu/mmvq.cu
      cu/norm.cu
      cu/pool2d.cu
      cu/quantize.cu
      cu/scale.cu
      cu/softmax.cu
      cu/unary.cu
      cu/wkv6.cu
)

target_link_libraries(libggml PRIVATE CUDA::cudart_static CUDA::cuda_driver CUDA::cublas)
target_compile_options(libggml PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
