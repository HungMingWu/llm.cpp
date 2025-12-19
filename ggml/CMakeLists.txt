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
add_subdirectory(rpc)
add_subdirectory(stdexec)
find_package(CUDAToolkit)

if (GGML_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, build CUDA Backend")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            if (CUDAToolkit_VERSION VERSION_LESS "13")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
            endif ()

            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
            endif()
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(libggml PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
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
      alloc.cppm
      buffer.cppm
      buffer_type.cppm
      ds.cppm
      func.cppm
      ggml.cppm
      gguf.cppm
      log.cppm
      op.cppm
      opt.cppm
      op_back.cppm
      stopwatch.cppm
      tensor.cppm
      types.cppm
      traits.cppm
      utility.cppm
      os/os.cppm
  PRIVATE
    FILE_SET private_interface
    TYPE CXX_MODULES
    FILES
      host_buffer.cppm
      quants.cppm
  PRIVATE
      allocator.cpp
      backend.cpp
      backend_registry.cpp
      buffer.cpp
      buffer_type.cpp
      cgraph.cpp
      context.cpp
      func.cpp
      gguf.cpp
      log.cpp
      op.cpp
      opt.cpp
      op_back.cpp
      quants.cpp
      sched.cpp
      tensor.cpp
      types.cpp
)

if (CMAKE_HOST_WIN32)
target_sources(libggml
  PRIVATE
      os/win32.cpp
)
elseif (CMAKE_HOST_APPLE)
elseif (CMAKE_HOST_UNIX)
target_sources(libggml
  PRIVATE
      os/unix.cpp
)
endif()

target_compile_features(libggml
  PUBLIC
    cxx_std_23
)

target_include_directories(libggml
  PRIVATE
    ./
    ${CMAKE_CURRENT_BINARY_DIR}/
)

target_link_libraries(libggml
  PRIVATE
    STDEXEC::stdexec
)

if (WIN32)
target_link_libraries(libggml
  PRIVATE
    Ws2_32
)
elseif (LINUX)
target_link_libraries(libggml
  PRIVATE
    m
)
endif()

add_subdirectory(examples)
add_subdirectory(tests)
