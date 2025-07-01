cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

target_sources(libggml
  PUBLIC
    FILE_SET primary_cpu_interface
    TYPE CXX_MODULES
    FILES
      aarch64.cppm
      backend.cppm
      buffer_type.cppm
      device.cppm
      ds.cppm
      func.cppm
      from_float.cppm
      to_float.cppm
      traits.cppm
      registry.cppm
  PRIVATE
    FILE_SET cpu_op_interface
    TYPE CXX_MODULES
    FILES
      vec_dot.cppm
      op/argsort.cppm
      op/concat.cppm
      op/leaky_relu.cppm
      op/norm.cppm
      op/op.cppm
      op/repeat_back.cppm
      op/scale.cppm
      op/unary.cppm
      op/upscale.cppm
  PRIVATE
      backend.cpp
      buffer_type.cpp
      device.cpp
      func.cpp
      plan.cpp
      to_float.cpp
      vec_dot.cpp
)

if (GGML_LLAMAFILE)
    message("Enable LLAMAFILE support")
    target_sources(libggml
      PRIVATE
        FILE_SET cpu_llamafile_interface
        TYPE CXX_MODULES
        FILES
          llamafile/sgemm.cppm
      PRIVATE
          llamafile/sgemm.cpp
    )
    target_compile_definitions(libggml PRIVATE GGML_USE_LLAMAFILE=1)
endif()