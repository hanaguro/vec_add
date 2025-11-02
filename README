# ベクトル加算ベンチマーク（CPU vs GPU）

このプログラムは、CPU、NVIDIA CUDA、AMD ROCm (HIP) での大規模ベクトル加算の性能を比較するベンチマークツールです。

## 特徴

- **マルチプラットフォーム対応**: CPU専用、CUDA、ROCm (HIP) のいずれかでビルド可能
- **統一されたコードベース**: プリプロセッサマクロで各プラットフォームに対応
- **エラーチェック**: すべてのGPU API呼び出しでエラーチェックを実施
- **OpenMP対応**: CPU計算をマルチスレッド並列化可能
- **GPU情報表示**: 利用可能なGPUデバイスの詳細情報を表示
- **結果検証**: GPU計算結果の正確性を自動検証

## 必要な環境

### CPU版
- C++11対応コンパイラ（g++、clang++など）
- OpenMP対応コンパイラ（並列化を有効にする場合）

### CUDA版
- NVIDIA GPU（Compute Capability 3.5以上推奨）
- CUDA Toolkit 10.0以上
- NVIDIA ドライバ

### ROCm版
- AMD GPU（GCN 3rd Gen以降、RDNA、CDNA アーキテクチャ）
- ROCm 4.0以上（5.x以降推奨）

## ビルド方法

### 重要: CUDAビルドについて

**CUDAでビルドする場合は、`.cu`拡張子のファイルが必要です。**

最初に以下のコマンドで`.cu`ファイルを作成してください：

```bash
cp vector_add_benchmark.cpp vector_add_benchmark.cu
```

Makefileは自動的に`.cu`ファイルを作成しますが、手動で作成することもできます。

### 1. CPU版（シングルスレッド）
```bash
make cpu
./vec_add_cpu
```

### 2. CPU版（OpenMP並列）
```bash
make cpu_omp
./vec_add_cpu
```

### 3. CUDA版
```bash
# デフォルトアーキテクチャ（sm_70）でビルド
make cuda
./vec_add_cuda

# 特定のアーキテクチャを指定してビルド
make cuda CUDA_ARCH=-arch=sm_86  # RTX 3090/A6000など
./vec_add_cuda
```

#### 主なCUDAアーキテクチャ

**注意**: CUDA 13.0以降では`sm_60` (Pascal) のサポートが削除されました。

| アーキテクチャ | GPU例 | CUDA 11.x | CUDA 12.x | CUDA 13.x+ |
|-------------|------|----------|----------|-----------|
| `sm_60` | Pascal (P100, GTX 1080) | ✓ | ✓ | ✗ |
| `sm_70` | Volta (V100, Titan V) | ✓ | ✓ | ✓ |
| `sm_75` | Turing (RTX 2080, T4) | ✓ | ✓ | ✓ |
| `sm_80` | Ampere (A100) | ✓ | ✓ | ✓ |
| `sm_86` | Ampere (RTX 3060/3090, A40) | ✓ | ✓ | ✓ |
| `sm_89` | Ada Lovelace (RTX 4090, L40) | ✗ | ✓ | ✓ |
| `sm_90` | Hopper (H100) | ✗ | ✓ | ✓ |

**使用中のCUDAバージョン確認:**
```bash
nvcc --version
```

### 4. ROCm (HIP) 版
```bash
# デフォルトアーキテクチャ（gfx906）でビルド
make hip
./vec_add_hip

# 特定のアーキテクチャを指定してビルド
make hip HIP_ARCH=--offload-arch=gfx90a  # MI210/MI250Xなど
./vec_add_hip
```

#### 主なROCmアーキテクチャ
- `gfx900`: Vega (MI25)
- `gfx906`: Vega 7nm (MI50, MI60, Radeon VII)
- `gfx908`: CDNA (MI100)
- `gfx90a`: CDNA2 (MI210, MI250, MI250X)
- `gfx940`, `gfx941`, `gfx942`: CDNA3 (MI300 シリーズ)
- `gfx1030`: RDNA2 (RX 6000 シリーズ)
- `gfx1100`: RDNA3 (RX 7000 シリーズ)

### 5. OpenMP付きGPU版
CPU部分をOpenMPで並列化したい場合：
```bash
make cuda_omp  # CUDA + OpenMP
make hip_omp   # ROCm + OpenMP
```

## CMakeを使用したビルド（オプション）

CMakeを使用することで、より柔軟なビルド設定とLanguage Serverサポートが得られます。

### CPU版
```bash
mkdir build && cd build
cmake ..
make
./vec_add_cpu
```

### CUDA版（単一アーキテクチャ）
```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
make
./vec_add_cuda
```

### CUDA版（複数アーキテクチャ - デフォルト）
```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA=ON
# CUDA 13.0+: 70,75,80,86,89,90をサポート
# CUDA 12.x: 60,70,75,80,86,89,90をサポート
# CUDA 11.x: 60,70,75,80,86をサポート
# (CUDAバージョンに応じて自動選択されます)
make
./vec_add_cuda
```

### ROCm版
```bash
# GPUアーキテクチャを自動検出してビルド
mkdir build && cd build
cmake .. -DBUILD_HIP=ON
make
./vec_add_hip

# または特定のアーキテクチャを指定
cmake .. -DBUILD_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1103
make
./vec_add_hip
```

### OpenMP有効化
```bash
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON
make
```

### Language Server用のcompile_commands.json生成
```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA=ON
ln -s build/compile_commands.json .
```

これにより、clangdやVS Code IntelliSenseがCUDAコードを正しく解析できるようになります。

## ベクトルサイズの変更

ソースコード内の以下の行を編集してベクトルサイズを変更できます：

```cpp
// const size_t N = 1 << 24;  // 16M 要素（約64MB）
// const size_t N = 1 << 28;  // 256M 要素（約1GB）
const size_t N = 1 << 30;     // 1G 要素（約4GB） ← デフォルト
```

## 実行例

### CPU版の出力例
```
==================================================
  Vector Addition Benchmark
==================================================
Mode: CPU Only
Vector size: 1073741824 elements (4096.00 MB)
-------------------------------------------------
Initializing vectors...

Running CPU version...
CPU time        : 1234.567 ms

Running CPU (OpenMP) version...
CPU (OpenMP)    : 234.567 ms
OpenMP Speedup  : 5.26x

✓ CPU computation completed!
```

### CUDA版の出力例
```
==================================================
  Vector Addition Benchmark
==================================================
GPU Backend: CUDA

=== GPU Device 0 ===
  Name               : NVIDIA GeForce RTX 3090
  Compute Capability : 8.6
  Total Global Mem   : 24268 MB
  Multiprocessors    : 82
  Clock Rate         : 1695 MHz
  Memory Clock Rate  : 9751 MHz
  Memory Bus Width   : 384 bits

Vector size: 1073741824 elements (4096.00 MB)
-------------------------------------------------
Initializing vectors...

Running CPU version...
CPU time        : 1234.567 ms

Running GPU version...
GPU (CUDA) kernel time : 12.345 ms
Verifying results...
✓ Verification passed!

==================================================
Performance Summary:
  CPU time        : 1234.567 ms
==================================================
```

### ROCm版の出力例
```
==================================================
  Vector Addition Benchmark
==================================================
GPU Backend: ROCm (HIP)

=== GPU Device 0 ===
  Name               : AMD Radeon RX 7900 XTX
  Compute Capability : 11.0
  Total Global Mem   : 24564 MB
  Multiprocessors    : 96
  Clock Rate         : 2500 MHz
  Memory Clock Rate  : 10000 MHz
  Memory Bus Width   : 384 bits

Vector size: 1073741824 elements (4096.00 MB)
-------------------------------------------------
Initializing vectors...

Running CPU version...
CPU time        : 1234.567 ms

Running GPU version...
GPU (ROCm/HIP) kernel time : 13.456 ms
Verifying results...
✓ Verification passed!

==================================================
Performance Summary:
  CPU time        : 1234.567 ms
==================================================
```

## GPUアーキテクチャの確認方法

### CUDAの場合
```bash
# デバイス情報を表示
nvidia-smi

# Compute Capabilityを確認
nvidia-smi --query-gpu=compute_cap --format=csv
```

### ROCmの場合
```bash
# デバイス情報を表示
rocm-smi

# アーキテクチャを確認（方法1）
rocminfo | grep "Name:" -A 5

# アーキテクチャを確認（方法2 - 推奨）
/opt/rocm/bin/rocm_agent_enumerator

# 例: gfx1103 が表示される場合
# cmake .. -DBUILD_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1103
```

## トラブルシューティング

### Language Serverのエラー（`threadIdx` is undeclared）

`.cu`ファイルをエディタで開くと、`threadIdx`、`blockIdx`などが「未宣言」とエラー表示されることがあります。これは、Language ServerがCUDA固有の組み込み変数を認識できないためです。

#### 解決方法

**方法1: `.clangd`設定ファイル（推奨）**

プロジェクトルートに`.clangd`ファイルを配置（付属しています）。

**方法2: VS Code設定**

`.vscode/c_cpp_properties.json`を使用（付属しています）。

**方法3: CMakeでビルド**

```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
make
# compile_commands.jsonをルートにリンク
ln -s build/compile_commands.json ..
```

詳細は付属の`CMakeLists.txt`を参照してください。

### CUDA版でエラーが出る場合
1. CUDA Toolkitがインストールされているか確認
   ```bash
   nvcc --version
   ```

2. 適切なCompute Capabilityを指定
   ```bash
   make cuda CUDA_ARCH=-arch=sm_XX
   ```

### ROCm版でエラーが出る場合
1. ROCmがインストールされているか確認
   ```bash
   hipcc --version
   ```

2. 環境変数を設定
   ```bash
   export PATH=/opt/rocm/bin:$PATH
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
   ```

3. 適切なGFXアーキテクチャを指定
   ```bash
   make hip HIP_ARCH=--offload-arch=gfxXXX
   ```

### メモリ不足エラー
ベクトルサイズが大きすぎる場合、以下を試してください：
- ソースコード内のNを小さくする（例：`1 << 28` または `1 << 24`）
- システムメモリやGPUメモリを確認

## パフォーマンス比較のヒント

1. **複数回実行して平均を取る**: 初回実行はキャッシュウォーミング等で遅くなることがあります

2. **異なるベクトルサイズでテスト**: メモリバウンド特性を観察

3. **システムの負荷を最小化**: 他のプロセスを停止してベンチマーク実行

4. **GPU周波数を確認**: 省電力モードになっていないか確認
   - NVIDIA: `nvidia-smi -q -d CLOCK`
   - AMD: `rocm-smi --showclocks`

## ライセンス

このコードは教育・研究目的で自由に使用できます。

## 参考リンク

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
