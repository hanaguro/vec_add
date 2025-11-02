cmake_minimum_required(VERSION 3.18)
project(VectorAddBenchmark LANGUAGES CXX)

# C++11を使用
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイルコマンドをエクスポート（Language Server用）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ソースファイル
set(SRC_CPP vector_add_benchmark.cpp)
set(SRC_CU vector_add_benchmark.cu)

# オプション: ビルドタイプの選択
option(BUILD_CUDA "Build CUDA version" OFF)
option(BUILD_HIP "Build ROCm/HIP version" OFF)
option(USE_OPENMP "Enable OpenMP" OFF)

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# CPU版
add_executable(vec_add_cpu ${SRC_CPP})
target_compile_definitions(vec_add_cpu PRIVATE CPU_ONLY)
if(USE_OPENMP)
    target_link_libraries(vec_add_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# OpenMP定義確認用テストプログラム（オプション）
# 注: CUDAとOpenMPの組み合わせでomp.hのインクルードに問題があるため無効化
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_openmp_defines.cpp)
#     add_executable(test_openmp_cpu test_openmp_defines.cpp)
#     target_compile_definitions(test_openmp_cpu PRIVATE CPU_ONLY)
#     if(USE_OPENMP)
#         target_link_libraries(test_openmp_cpu PRIVATE OpenMP::OpenMP_CXX)
#     endif()
# endif()

# CUDA版
if(BUILD_CUDA)
    enable_language(CUDA)
    
    # CUDAバージョンを確認
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected: Removing sm_60 support")
        # CUDA 13.0以降はsm_60をサポートしない
        set(DEFAULT_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected: Including sm_60 support")
        set(DEFAULT_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        set(DEFAULT_CUDA_ARCHITECTURES "60;70;75;80;86")
    else()
        set(DEFAULT_CUDA_ARCHITECTURES "60;70;75")
    endif()
    
    # 内容が同じ場合はコピーされず、内容が異なればコピーされる
    configure_file(${SRC_CPP} ${SRC_CU} COPYONLY)
    
    add_executable(vec_add_cuda ${SRC_CU})
    target_compile_definitions(vec_add_cuda PRIVATE USE_CUDA)

    target_compile_options(vec_add_cuda PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11 -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++11 -O3>
    )
   
    # OpenMPサポート
    if(USE_OPENMP)
        # CUDAコンパイラにOpenMPフラグを渡す（これで_OPENMPが自動定義される）
        target_compile_options(vec_add_cuda PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
        )
        target_link_libraries(vec_add_cuda PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # CUDA アーキテクチャ設定
    # CUDA 13.0以降ではcompute_60/sm_60が削除されているため、70以降のみサポート
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # デフォルト: CUDAバージョンに応じた適切なアーキテクチャ
        set(CMAKE_CUDA_ARCHITECTURES "${DEFAULT_CUDA_ARCHITECTURES}")
    endif()
    
    set_target_properties(vec_add_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # 注: 特定のアーキテクチャのみビルドする場合は以下のように指定:
    # cmake .. -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
endif()

# ROCm/HIP版
if(BUILD_HIP)
    # HIP_PLATFORMを設定（通常はamd）
    if(NOT DEFINED HIP_PLATFORM)
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform")
    endif()
    set(ENV{HIP_PLATFORM} ${HIP_PLATFORM})
    
    # hipccコンパイラを探す
    find_program(HIP_HIPCC_EXECUTABLE
        NAMES hipcc
        PATHS /opt/rocm/bin /opt/rocm/hip/bin
        DOC "HIP compiler driver"
    )
    
    if(NOT HIP_HIPCC_EXECUTABLE)
        message(FATAL_ERROR "Could not find hipcc. Please install ROCm.")
    endif()
    
    message(STATUS "Found hipcc: ${HIP_HIPCC_EXECUTABLE}")
    
    # HIPアーキテクチャの設定
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(HIP_ARCH_FLAGS "--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
        message(STATUS "HIP Architecture: ${CMAKE_HIP_ARCHITECTURES}")
    else()
        # rocminfoで検出するか、デフォルト値を使用
        execute_process(
            COMMAND bash -c "/opt/rocm/bin/rocm_agent_enumerator | grep -v gfx000"
            OUTPUT_VARIABLE DETECTED_GPU
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(DETECTED_GPU)
            string(REPLACE "\n" ";" GPU_LIST ${DETECTED_GPU})
            list(GET GPU_LIST 0 DEFAULT_GPU)
            set(HIP_ARCH_FLAGS "--offload-arch=${DEFAULT_GPU}")
            message(STATUS "Auto-detected HIP Architecture: ${DEFAULT_GPU}")
        else()
            # 検出できない場合はビルドしない（ランタイムに決定）
            set(HIP_ARCH_FLAGS "")
            message(STATUS "HIP Architecture: Not specified (will use runtime detection)")
        endif()
    endif()
    
    # 出力ファイルのパス
    set(VEC_ADD_HIP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vec_add_hip_exe)
    
    # カスタムコマンドでhipccを使用してビルド
    add_custom_command(
        OUTPUT ${VEC_ADD_HIP_OUTPUT}
        COMMAND ${HIP_HIPCC_EXECUTABLE}
            -std=c++11 -O3 -DUSE_HIP
            ${HIP_ARCH_FLAGS}
            $<$<BOOL:${USE_OPENMP}>:-fopenmp>
            ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_CPP}
            -o ${VEC_ADD_HIP_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_CPP}
        COMMENT "Building HIP version with hipcc"
        VERBATIM
    )
    
    # カスタムターゲット
    add_custom_target(vec_add_hip_build ALL
        DEPENDS ${VEC_ADD_HIP_OUTPUT}
    )
    
    # 実行ファイルを適切な名前にコピー
    add_custom_command(
        TARGET vec_add_hip_build POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${VEC_ADD_HIP_OUTPUT}
            ${CMAKE_CURRENT_BINARY_DIR}/vec_add_hip
        COMMENT "Copying vec_add_hip to final location"
    )
    
    # 注: 特定のアーキテクチャを指定する場合:
    # cmake .. -DBUILD_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1103
endif()

# ビルド情報の表示
message(STATUS "================================================")
message(STATUS "  Vector Addition Benchmark - CMake Configuration")
message(STATUS "================================================")
message(STATUS "Build CUDA version: ${BUILD_CUDA}")
message(STATUS "Build HIP version: ${BUILD_HIP}")
message(STATUS "Use OpenMP: ${USE_OPENMP}")
if(BUILD_CUDA)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "================================================")
