cmake_minimum_required(VERSION 3.28)
project(llm.cpp)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

option(GGML_CUDA                            "ggml: use CUDA"                                  ON)
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