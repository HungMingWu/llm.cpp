cmake_minimum_required(VERSION 3.28)
project(llm.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /constexpr:steps 33554432")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=33554432")
endif()

option(GGML_CUDA                            "ggml: use CUDA"                                  ON)
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             OFF)
add_subdirectory(ggml)
add_subdirectory(core)

add_executable(tutorial)

target_sources(tutorial
  PRIVATE
    main.cpp
)

target_link_libraries(tutorial
  PRIVATE
    libllm
)

add_subdirectory(chatllm)