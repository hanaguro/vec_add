/********************************************************************
 *  ベクトル加算ベンチマーク（CPU vs GPU）                           *
 *  - ROCm (HIP) と CUDA の両方に対応                                *
 *  - GPU の戻り値を必ずチェック                                     *
 ********************************************************************/

#include <iostream>		// std::cout, std::cerr, std::endl
#include <vector>		// std::vector<float>
#include <chrono>		// std::chrono::high_resolution_clock::now(), std::chrono::duration<double, std::milli>
#include <random>		// std::mt19937, std::uniform_real_distribution<float>
#include <iomanip>		// std::fixed, std::setprecision()
#include <cstdlib>		// std::abort

// GPUバックエンドの選択
#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_PREFIX cuda
    #define GPU_ERROR_T cudaError_t
    #define GPU_SUCCESS cudaSuccess
    #define GPU_GET_ERROR_STRING cudaGetErrorString
    #define GPU_MALLOC cudaMalloc
    #define GPU_FREE cudaFree
    #define GPU_MEMCPY cudaMemcpy
    #define GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
    #define GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
    #define GPU_EVENT_T cudaEvent_t
    #define GPU_EVENT_CREATE cudaEventCreate
    #define GPU_EVENT_RECORD cudaEventRecord
    #define GPU_EVENT_SYNCHRONIZE cudaEventSynchronize
    #define GPU_EVENT_ELAPSED_TIME cudaEventElapsedTime
    #define GPU_EVENT_DESTROY cudaEventDestroy
    #define GPU_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
    #define GPU_GET_DEVICE_PROPERTIES cudaGetDeviceProperties
    #define GPU_DEVICE_PROP cudaDeviceProp
    #define GPU_GET_DEVICE_COUNT cudaGetDeviceCount
    #define GPU_SET_DEVICE cudaSetDevice
    #define GPU_KERNEL_LAUNCH(kernel, blocks, threads, smem, stream, ...) \
        kernel<<<blocks, threads, smem, stream>>>(__VA_ARGS__)
#elif defined(USE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_PREFIX hip
    #define GPU_ERROR_T hipError_t
    #define GPU_SUCCESS hipSuccess
    #define GPU_GET_ERROR_STRING hipGetErrorString
    #define GPU_MALLOC hipMalloc
    #define GPU_FREE hipFree
    #define GPU_MEMCPY hipMemcpy
    #define GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
    #define GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
    #define GPU_EVENT_T hipEvent_t
    #define GPU_EVENT_CREATE hipEventCreate
    #define GPU_EVENT_RECORD hipEventRecord
    #define GPU_EVENT_SYNCHRONIZE hipEventSynchronize
    #define GPU_EVENT_ELAPSED_TIME hipEventElapsedTime
    #define GPU_EVENT_DESTROY hipEventDestroy
    #define GPU_DEVICE_SYNCHRONIZE hipDeviceSynchronize
    #define GPU_GET_DEVICE_PROPERTIES hipGetDeviceProperties
    #define GPU_DEVICE_PROP hipDeviceProp_t
    #define GPU_GET_DEVICE_COUNT hipGetDeviceCount
    #define GPU_SET_DEVICE hipSetDevice
    #define GPU_KERNEL_LAUNCH(kernel, blocks, threads, smem, stream, ...) \
        hipLaunchKernelGGL(kernel, blocks, threads, smem, stream, __VA_ARGS__)
#endif

/* -------------------------------------------------------------
 *  エラーチェックマクロ
 * ------------------------------------------------------------- */
#ifndef CPU_ONLY
#define GPU_CHECK(expr)                                            \
    do {                                                            \
        GPU_ERROR_T _err = (expr);                                  \
        if (_err != GPU_SUCCESS) {                                  \
            std::cerr << "GPU error: " << GPU_GET_ERROR_STRING(_err)\
                      << " (" << #expr << ") at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;              \
            std::abort();                                           \
        }                                                           \
    } while (0)
#endif

// -------------------------------------------------------------
// カーネル（GPU 用）
// -------------------------------------------------------------
#ifndef CPU_ONLY
#if defined(USE_CUDA)
__global__
#elif defined(USE_HIP)
__global__
#endif
void vecAddKernel(const float* a,
                  const float* b,
                  float*       c,
                  size_t       n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
#endif   // CPU_ONLY

// -------------------------------------------------------------
// 乱数でベクトル初期化
// -------------------------------------------------------------
void initVector(std::vector<float>& v)
{
    std::mt19937 rng(0xdeadbeef);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// -------------------------------------------------------------
// CPU 版（シングルスレッド）
// -------------------------------------------------------------
void vecAddCPU(const std::vector<float>& a,
               const std::vector<float>& b,
               std::vector<float>&       c)
{
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// -------------------------------------------------------------
// CPU 版（OpenMP 並列化） ※ -fopenmp で有効化
// -------------------------------------------------------------
#ifdef _OPENMP
#include <omp.h>
void vecAddCPU_omp(const std::vector<float>& a,
                   const std::vector<float>& b,
                   std::vector<float>&       c)
{
    const size_t n = a.size();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
#endif

// -------------------------------------------------------------
// GPU デバイス情報表示
// -------------------------------------------------------------
#ifndef CPU_ONLY
void printGPUInfo()
{
    int deviceCount = 0;
    GPU_CHECK( GPU_GET_DEVICE_COUNT(&deviceCount) );
    
    if (deviceCount == 0) {
        std::cout << "No GPU devices found!\n";
        return;
    }
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        GPU_DEVICE_PROP prop;
        GPU_CHECK( GPU_GET_DEVICE_PROPERTIES(&prop, dev) );
        
        std::cout << "\n=== GPU Device " << dev << " ===\n";
        std::cout << "  Name               : " << prop.name << "\n";
        std::cout << "  Compute Capability : " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Mem   : " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        std::cout << "  Multiprocessors    : " << prop.multiProcessorCount << "\n";
#if defined(USE_HIP)
        std::cout << "  Clock Rate         : " << (prop.clockRate / 1000) << " MHz\n";
        std::cout << "  Memory Clock Rate  : " << (prop.memoryClockRate / 1000) << " MHz\n";
        std::cout << "  Memory Bus Width   : " << prop.memoryBusWidth << " bits\n";
#endif
    }
    std::cout << std::endl;
}
#endif

// -------------------------------------------------------------
// GPU 版
// -------------------------------------------------------------
#ifndef CPU_ONLY
void vecAddGPU(const std::vector<float>& h_a,
               const std::vector<float>& h_b,
               std::vector<float>&       h_c)
{
    const size_t N = h_a.size();
    const size_t bytes = N * sizeof(float);

    // デバイスメモリ確保
    float *d_a, *d_b, *d_c;
    GPU_CHECK( GPU_MALLOC(&d_a, bytes) );
    GPU_CHECK( GPU_MALLOC(&d_b, bytes) );
    GPU_CHECK( GPU_MALLOC(&d_c, bytes) );

    // Host → Device 転送
    GPU_CHECK( GPU_MEMCPY(d_a, h_a.data(), bytes, GPU_MEMCPY_HOST_TO_DEVICE) );
    GPU_CHECK( GPU_MEMCPY(d_b, h_b.data(), bytes, GPU_MEMCPY_HOST_TO_DEVICE) );

    // カーネル起動設定
    const int threadsPerBlock = 256;
    const int blocks = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

    // GPU タイマー（Event）で計測
    GPU_EVENT_T start, stop;
    GPU_CHECK( GPU_EVENT_CREATE(&start) );
    GPU_CHECK( GPU_EVENT_CREATE(&stop) );

    GPU_CHECK( GPU_EVENT_RECORD(start, nullptr) );

    // カーネル起動（統一マクロを使用）
    GPU_KERNEL_LAUNCH(vecAddKernel, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                      d_a, d_b, d_c, N);

    GPU_CHECK( GPU_EVENT_RECORD(stop, nullptr) );
    GPU_CHECK( GPU_EVENT_SYNCHRONIZE(stop) );

    float ms = 0.0f;
    GPU_CHECK( GPU_EVENT_ELAPSED_TIME(&ms, start, stop) );   // ミリ秒

    // Device → Host 転送
    GPU_CHECK( GPU_MEMCPY(h_c.data(), d_c, bytes, GPU_MEMCPY_DEVICE_TO_HOST) );

    // 後始末
    GPU_CHECK( GPU_FREE(d_a) );
    GPU_CHECK( GPU_FREE(d_b) );
    GPU_CHECK( GPU_FREE(d_c) );
    GPU_CHECK( GPU_EVENT_DESTROY(start) );
    GPU_CHECK( GPU_EVENT_DESTROY(stop) );

#if defined(USE_CUDA)
    std::cout << "GPU (CUDA) kernel time : " << std::fixed << std::setprecision(3)
              << ms << " ms\n";
#elif defined(USE_HIP)
    std::cout << "GPU (ROCm/HIP) kernel time : " << std::fixed << std::setprecision(3)
              << ms << " ms\n";
#endif
}
#endif   // CPU_ONLY

// -------------------------------------------------------------
// 正しさチェック（簡易）
// -------------------------------------------------------------
void verify(const std::vector<float>& ref,
            const std::vector<float>& got)
{
    const float eps = 1e-5f;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - got[i]) > eps) {
            std::cerr << "Mismatch at [" << i << "]: "
                      << ref[i] << " vs " << got[i] << std::endl;
            std::abort();
        }
    }
}

// -------------------------------------------------------------
// main()
// -------------------------------------------------------------
int main()
{
//    std::cout << "=================================================\n";
//    std::cout << "  Vector Addition Benchmark\n";
//    std::cout << "=================================================\n";

#if defined(USE_CUDA)
    std::cout << "GPU Backend: CUDA\n";
#elif defined(USE_HIP)
    std::cout << "GPU Backend: ROCm (HIP)\n";
#elif defined(CPU_ONLY)
    std::cout << "Mode: CPU Only\n";
#else
    std::cout << "Warning: No GPU backend specified!\n";
#endif

#ifndef CPU_ONLY
    printGPUInfo();
#endif

    // ベクトルサイズ選択（コメントアウトを切り替え）
    // const size_t N = 1 << 24;                 // 16M 要素程度（約64MB）
    const size_t N = 1 << 28;                 // 256M 要素程度（約1GB）
    // const size_t N = 1 << 30;                    // 1G 要素程度（約4GB）
    
    std::cout << "Vector size: " << N << " elements (" 
              << (N * sizeof(float) / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "-------------------------------------------------\n";

    std::vector<float> a(N), b(N), c_gpu(N), c_cpu(N);

//    std::cout << "Initializing vectors...\n";
    initVector(a);
    initVector(b);

    // ------------------- CPU 計測 -------------------
//    std::cout << "\nRunning CPU version...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    vecAddCPU(a, b, c_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU time        : " << std::fixed << std::setprecision(3) 
              << cpu_ms << " ms\n";

#ifdef _OPENMP
    // ------------------- OpenMP 計測（オプション） -------------------
//   std::cout << "\nRunning CPU (OpenMP) version...\n";
    std::fill(c_cpu.begin(), c_cpu.end(), 0.0f);
    auto t2 = std::chrono::high_resolution_clock::now();
    vecAddCPU_omp(a, b, c_cpu);
    auto t3 = std::chrono::high_resolution_clock::now();
    double omp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "CPU (OpenMP)    : " << std::fixed << std::setprecision(3) 
              << omp_ms << " ms\n";
    std::cout << "OpenMP Speedup  : " << std::fixed << std::setprecision(2) 
              << (cpu_ms / omp_ms) << "x\n";
#endif

#ifndef CPU_ONLY
    // ------------------- GPU 計測 -------------------
//   std::cout << "\nRunning GPU version...\n";
    vecAddGPU(a, b, c_gpu);
    
    // 正解との比較
//   std::cout << "Verifying results...\n";
    verify(c_cpu, c_gpu);
//   std::cout << "✓ Verification passed!\n";
    
    // スピードアップ計算（カーネル実行時間は別途表示済み）
//    std::cout << "\n=================================================\n";
//    std::cout << "Performance Summary:\n";
//    std::cout << "  CPU time        : " << std::fixed << std::setprecision(3) 
//              << cpu_ms << " ms\n";
#ifdef _OPENMP
//    std::cout << "  CPU (OpenMP)    : " << std::fixed << std::setprecision(3) 
//              << omp_ms << " ms\n";
#endif
//    std::cout << "=================================================\n";
#else
//    std::cout << "\n✓ CPU computation completed!\n";
#endif

    return 0;
}
