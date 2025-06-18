cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

target_sources(libggml
  PUBLIC
    FILE_SET primary_rpc_interface
    TYPE CXX_MODULES
    FILES
      rpc.cppm
  PRIVATE
    FILE_SET private_rpc_interface
    TYPE CXX_MODULES
    FILES
      backend.cppm
      buffer.cppm
      buffer_type.cppm
      ds.cppm
      helper.cppm
      rpc_server.cppm
      socket.cppm
  PRIVATE
      backend.cpp
      buffer.cpp
      buffer_type.cpp
      rpc.cpp
      rpc_server.cpp
      socket.cpp
)
