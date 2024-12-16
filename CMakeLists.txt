cmake_minimum_required(VERSION 3.28)
project(llm.cpp)

add_subdirectory(core)

add_executable(tutorial)

target_sources(tutorial
  PRIVATE
    main.cpp
)

target_link_libraries(tutorial
  PRIVATE
    MathFunctions
)
