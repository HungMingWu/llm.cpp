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
        # 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
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

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                # The CUDA architecture 120f-virtual would in principle work for Blackwell support
                #     but the newly added "f" suffix conflicted with a preexising regex for validating CUDA architectures in CMake.
                # So either a recent CMake version or one with the backported fix is needed.
                # The following versions should work:
                #   - CMake >= v3.31.8 && CMake < v4.0.0
                #   - CMake >= v4.0.2
                # This is NOT documented in the CMake release notes,
                #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)

    # Replace any plain 12X CUDA architectures with their "architecture-specific" equivalents 12Xa.
    # 12X is forwards-compatible, 12Xa is not.
    # Notably the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa.
    # But while 12X vs. 12Xa can be checked in device code there is (to my knowledge) no easy way to do the same check in host code.
    # So for now just replace all instances of 12X with 12Xa, this should be fine until Rubin is released.
    foreach(ARCHS IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(FIXED_ARCHS "")
        foreach(ARCH IN LISTS ${ARCHS})
            if (ARCH MATCHES "^12[0-9](-real|-virtual)?$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
                message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
                list(APPEND FIXED_ARCHS "${FIXED_ARCH}")
            else()
                list(APPEND FIXED_ARCHS "${ARCH}")
            endif()
        endforeach()
        set(${ARCHS} ${FIXED_ARCHS})
    endforeach()

    # If we try to compile a "native" build it will use the 12X architectures and fail.
    # So we should instead use the native architectures as determined by CMake after replacing 12X with 12Xa.
    # But if at the time of the build no GPUs are connected at all CMAKE_CUDA_ARCHITECTURES will contain garbage that we should not use.
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

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
