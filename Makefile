# Makefile for Vector Addition Benchmark
# Supports: CPU-only, CUDA, and ROCm (HIP)

# デフォルトのコンパイラとフラグ
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

# 実行ファイル名
TARGET_CPU = vec_add_cpu
TARGET_CUDA = vec_add_cuda
TARGET_HIP = vec_add_hip

# ソースファイル
SRC_CPP = vector_add_benchmark.cpp
SRC_CU = vector_add_benchmark.cu

# CUDA設定
NVCC = nvcc
CUDA_FLAGS = -std=c++11 -O3 -DUSE_CUDA
CUDA_ARCH = -arch=sm_86  # 必要に応じて変更（sm_60, sm_75, sm_80, sm_86, sm_89, sm_90など）

# ROCm (HIP) 設定
HIPCC = hipcc
HIP_FLAGS = -std=c++11 -O3 -DUSE_HIP
# ROCm アーキテクチャ（必要に応じて変更）
# gfx900: Vega (MI25, etc.)
# gfx906: Vega 7nm (MI50, MI60)
# gfx908: CDNA (MI100)
# gfx90a: CDNA2 (MI210, MI250X)
# gfx940, gfx941, gfx942: CDNA3 (MI300)
# gfx1030: RDNA2 (RX 6000 series)
# gfx1100: RDNA3 (RX 7000 series)
HIP_ARCH = --offload-arch=gfx1103  # 必要に応じて変更

# OpenMP設定（オプション）
OPENMP_FLAGS = -fopenmp

# 実際のファイルを生成しないターゲット
.PHONY: all cpu cuda hip cpu_omp cuda_omp hip_omp clean help

# デフォルトターゲット
all: help

# CPU版（シングルスレッド）
cpu:
	$(CXX) $(CXXFLAGS) -DCPU_ONLY $(SRC_CPP) -o $(TARGET_CPU)
	@echo "Built: $(TARGET_CPU) (CPU single-threaded)"

# CPU版（OpenMP並列）
cpu_omp:
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -DCPU_ONLY $(SRC_CPP) -o $(TARGET_CPU)
	@echo "Built: $(TARGET_CPU) (CPU with OpenMP)"

# CUDA版
cuda: $(SRC_CU)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) $(SRC_CU) -o $(TARGET_CUDA)
	@echo "Built: $(TARGET_CUDA) (CUDA)"

# CUDA版（OpenMP付き）
cuda_omp: $(SRC_CU)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -Xcompiler "$(OPENMP_FLAGS)" $(SRC_CU) -o $(TARGET_CUDA)
	@echo "Built: $(TARGET_CUDA) (CUDA with OpenMP)"

# ROCm (HIP) 版
hip:
	$(HIPCC) $(HIP_FLAGS) $(HIP_ARCH) $(SRC_CPP) -o $(TARGET_HIP)
	@echo "Built: $(TARGET_HIP) (ROCm/HIP)"

# ROCm (HIP) 版（OpenMP付き）
hip_omp:
	$(HIPCC) $(HIP_FLAGS) $(HIP_ARCH) $(OPENMP_FLAGS) $(SRC_CPP) -o $(TARGET_HIP)
	@echo "Built: $(TARGET_HIP) (ROCm/HIP with OpenMP)"

# クリーンアップ
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) $(TARGET_HIP)
	@echo "Cleaned all executables"

# ヘルプ
help:
	@echo "=================================================="
	@echo "  Vector Addition Benchmark - Build Options"
	@echo "=================================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  cpu         - Build CPU-only version (single-threaded)"
	@echo "  cpu_omp     - Build CPU-only version with OpenMP"
	@echo "  cuda        - Build CUDA version (requires .cu file)"
	@echo "  cuda_omp    - Build CUDA version with OpenMP for CPU"
	@echo "  hip         - Build ROCm (HIP) version"
	@echo "  hip_omp     - Build ROCm (HIP) version with OpenMP for CPU"
	@echo "  clean       - Remove all executables"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Note: CUDA build requires vector_add_benchmark.cu file"
	@echo "      HIP and CPU builds use vector_add_benchmark.cpp file"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_ARCH   - Set CUDA architecture (default: sm_70)"
	@echo "              Example: make cuda CUDA_ARCH=-arch=sm_80"
	@echo "  HIP_ARCH    - Set ROCm architecture (default: gfx906)"
	@echo "              Example: make hip HIP_ARCH=--offload-arch=gfx90a"
	@echo ""
	@echo "Examples:"
	@echo "  make cpu              # CPU single-threaded"
	@echo "  make cpu_omp          # CPU with OpenMP"
	@echo "  make cuda             # CUDA (default architecture)"
	@echo "  make cuda CUDA_ARCH=-arch=sm_86  # CUDA for RTX 3090"
	@echo "  make hip              # ROCm/HIP (default architecture)"
	@echo "  make hip HIP_ARCH=--offload-arch=gfx90a  # MI210/MI250X"
	@echo "=================================================="

# .cuファイルが存在しない場合、または.cppが.cuより新しい場合は.cppから作成
$(SRC_CU): $(SRC_CPP)
	@if [ ! -f $(SRC_CU) ]; then \
		echo "Creating $(SRC_CU) from $(SRC_CPP)..."; \
		cp $(SRC_CPP) $(SRC_CU); \
	elif [ $(SRC_CPP) -nt $(SRC_CU) ]; then \
		echo "$(SRC_CPP) is newer than $(SRC_CU). Updating..."; \
		cp $(SRC_CPP) $(SRC_CU); \
	else \
		echo "$(SRC_CU) is up to date."; \
	fi
