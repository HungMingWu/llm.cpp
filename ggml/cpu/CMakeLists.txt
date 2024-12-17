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
      vec_dot.cppm
      registry.cppm
  PRIVATE
    FILE_SET cpu_op_interface
    TYPE CXX_MODULES
    FILES
      op/argsort.cppm
      op/concat.cppm
      op/norm.cppm
      op/op.cppm
      op/scale.cppm
      op/unary.cppm
      op/upscale.cppm
  PRIVATE
      backend.cpp
      device.cpp
      func.cpp
      plan.cpp
 "buffer_type.cpp")