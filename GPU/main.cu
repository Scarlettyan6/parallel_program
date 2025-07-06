#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip> 
#include "cuda_runtime.h"
using namespace std;
// 宏定义，用于统一的错误检查
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ <<endl; \
        exit(EXIT_FAILURE); \
    } \
}

// -----------------------------------------------------
// GPU上执行的代码
// -----------------------------------------------------

__global__ void division_kernel(float* m, int n, int k) {
    // 任务：并行计算第k行的除法
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从 k+1 列开始计算
    j += (k + 1);

    if (j < n) {
        m[k * n + j] /= m[k * n + k];
    }
}

__global__ void eliminate_kernel(float* m, int n, int k) {
    // 任务：一个Block负责一行，并行计算消去
    // 每个块负责一行
    int i = k + 1 + blockIdx.x;

    if (i < n) {
        // 块内线程负责不同列
        int j = k + 1 + threadIdx.x;
        
        // 使用grid-stride loop处理列数大于BLOCK_SIZE的情况
        while (j < n) {
            m[i * n + j] -= m[i * n + k] * m[k * n + j];
            j += blockDim.x;
        }
    }
}


// -----------------------------------------------------
//CPU上执行的代码
// -----------------------------------------------------

// CPU版本的高斯消元，用于对比
void gaussian_elimination_cpu(float* m, int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j) {
            m[k * n + j] = m[k * n + j] / m[k * n + k];
        }
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                m[i * n + j] = m[i * n + j] - m[i * n + k] * m[k * n + j];
            }
        }
    }
}



// 验证结果
bool verify_results(const float* cpu_res, const float* gpu_res, int n) {
    const float epsilon = 1e-4; // 允许的误差
    for (int i = 0; i < n * n; ++i) {
        if (abs(cpu_res[i] - gpu_res[i]) > epsilon) {
            cerr << "Verification failed at index " << i << "! "
                      << "CPU: " << cpu_res[i] << ", GPU: " << gpu_res[i] << endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    int N=1024;
    N = std::stoi(argv[1]); //通过第一个参数设置矩阵规模
    const int matrix_size = N * N * sizeof(float);

    // 1. Host内存分配与初始化
    float* h_matrix_cpu = new float[N * N];
    float* h_matrix_gpu = new float[N * N];
    srand(time(0));
    for (int i = 0; i < N * N; ++i) {
        h_matrix_cpu[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    // 拷贝一份用于GPU计算
    memcpy(h_matrix_gpu, h_matrix_cpu, matrix_size);

    // --- CPU-based calculation ---
    cout << "Starting CPU Gaussian Elimination..." << endl;
    auto start_cpu = chrono::high_resolution_clock::now();
    gaussian_elimination_cpu(h_matrix_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    cout << "CPU time: " << cpu_duration.count() << " ms" <<endl;
    
    // --- GPU-based calculation ---
    cout << "\nStarting GPU Gaussian Elimination..." << endl;

    // 2. Device内存分配
    float* d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size));

    // 3. 数据从Host传输到Device
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix_gpu, matrix_size, cudaMemcpyHostToDevice));

    // 创建CUDA事件用于计时
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // 4. 在CPU循环中启动Kernel
    const int BLOCK_SIZE = 1024; // 还可以选择128, 256, 512
    for (int k = 0; k < N; ++k) {
        // --- division kernel ---
        int grid_size_div = (N - (k + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid_size_div > 0) {
            division_kernel<<<grid_size_div, BLOCK_SIZE>>>(d_matrix, N, k);
            CUDA_CHECK(cudaGetLastError()); // 每次kernel后检查错误
        }

        // --- elimination kernel ---
        int grid_size_elim = (N - (k + 1)); // 一个块负责一行
        if (grid_size_elim > 0) {
            eliminate_kernel<<<grid_size_elim, BLOCK_SIZE>>>(d_matrix, N, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu)); // 等待GPU完成

    float gpu_duration_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu));
    std::cout << "GPU time: " << gpu_duration_ms << " ms" << endl;

    // 5. 数据从Device传输回Host
    CUDA_CHECK(cudaMemcpy(h_matrix_gpu, d_matrix, matrix_size, cudaMemcpyDeviceToHost));
    
    // --- 资源释放 ---
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    // --- 验证和性能对比 ---
    cout << "\nVerifying results..." << std::endl;
    if (verify_results(h_matrix_cpu, h_matrix_gpu, N)) {
        cout << "Results are consistent!" << std::endl;
        cout << "Speedup: " << cpu_duration.count() / gpu_duration_ms << "x" << endl;
    }

    delete[] h_matrix_cpu;
    delete[] h_matrix_gpu;

    return 0;
}