function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "${backend}" backend_target)
        string(TOUPPER "GGML_USE_${backend}" backend_use)
        message(STATUS "Including ${backend} backend")
        add_subdirectory(${backend_target})
        target_compile_definitions(libggml PUBLIC ${backend_use})
    endif()
endfunction()

add_library(libggml OBJECT)

add_subdirectory(cpu)
add_subdirectory(stdexec)
find_package(CUDAToolkit)

if (GGML_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, build CUDA Backend")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 52     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set_target_properties(libggml PROPERTIES CUDA_ARCHITECTURES "native")
        elseif(GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            set_target_properties(libggml PROPERTIES CUDA_ARCHITECTURES "60;61;70;75")
        else()
            set_target_properties(libggml PROPERTIES CUDA_ARCHITECTURES "52;61;70;75")
        endif()
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    ggml_add_backend(cuda)
else()
    message("CUDA Toolkit not found")
endif()

target_sources(libggml
  PUBLIC
    FILE_SET primary_interface
    TYPE CXX_MODULES
    FILES
      ggml.cppm
      alloc.cppm
      ds.cppm
      func.cppm
      op.cppm
      op_back.cppm
      quants.cppm
      tensor.cppm
      types.cppm
      traits.cppm
      utility.cppm
  PRIVATE
      allocator.cpp
      cgraph.cpp
      context.cpp
      op.cpp
      op_back.cpp
      tensor.cpp
)

target_compile_features(libggml
  PUBLIC
    cxx_std_23
)

target_include_directories(libggml
  PRIVATE
    ./
)

target_link_libraries(libggml
  PRIVATE
    STDEXEC::stdexec
)

add_subdirectory(examples)
add_subdirectory(tests)