#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// GPU上的除法核函数，按行计算
__global__ void division_kernel(float* m, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (j < n) {
        m[k * n + j] /= m[k * n + k];  // 行主序
    }
}

// GPU上的消元核函数，按行计算
__global__ void eliminate_kernel(float* m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    __shared__ float shared_matrix[256]; // 使用共享内存

    if (i < n) {
        // 将数据加载到共享内存中
        int j = threadIdx.x + k + 1;
        while (j < n) {
            shared_matrix[threadIdx.x] = m[i * n + j];  // 行主序
            __syncthreads();
            m[i * n + j] -= m[i * n + k] * shared_matrix[threadIdx.x];
            j += blockDim.x;
        }
    }
}

// CPU实现的高斯消元，用于对比
void gaussian_elimination_cpu(float* m, int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j) {
            m[k * n + j] /= m[k * n + k];
        }
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                m[i * n + j] -= m[i * n + k] * m[k * n + j];
            }
        }
    }
}

// 验证CPU和GPU计算结果的函数
bool verify_results(const float* cpu_res, const float* gpu_res, int n) {
    const float epsilon = 1e-4; // 允许的误差
    for (int i = 0; i < n * n; ++i) {
        if (abs(cpu_res[i] - gpu_res[i]) > epsilon) {
            std::cerr << "Verification failed at index " << i << "! "
                      << "CPU: " << cpu_res[i] << ", GPU: " << gpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int N = 1024;  // 默认矩阵规模
    if (argc > 1) {
        N = std::stoi(argv[1]);  // 从命令行获取矩阵规模
    }
    
    const int matrix_size = N * N * sizeof(float);

    // 分配主机内存
    float* h_matrix_cpu = new float[N * N];
    float* h_matrix_gpu = new float[N * N];

    // 随机初始化矩阵
    srand(time(0));
    for (int i = 0; i < N * N; ++i) {
        h_matrix_cpu[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    memcpy(h_matrix_gpu, h_matrix_cpu, matrix_size);

    // --- CPU版本计算 ---
    std::cout << "Starting CPU Gaussian Elimination..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gaussian_elimination_cpu(h_matrix_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    
    // --- GPU版本计算 ---
    std::cout << "\nStarting GPU Gaussian Elimination..." << std::endl;

    // 1. 分配设备内存
    float* d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size));

    // 2. 将数据从Host复制到Device
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_matrix, h_matrix_gpu, matrix_size, cudaMemcpyHostToDevice, stream1));

    // 创建CUDA事件，用于计时
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // 3. 启动Kernel进行计算
    const int BLOCK_SIZE = 256;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 对每个k执行除法与消元核函数
    for (int k = 0; k < N; ++k) {
        division_kernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_matrix, N, k);
        eliminate_kernel<<<grid_size, BLOCK_SIZE, 0, stream2>>>(d_matrix, N, k);
    }

    // 等待所有流完成
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // 获取GPU计算时间
    float gpu_duration_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu));
    std::cout << "GPU time: " << gpu_duration_ms << " ms" << std::endl;

    // 4. 将数据从Device复制回Host
    CUDA_CHECK(cudaMemcpyAsync(h_matrix_gpu, d_matrix, matrix_size, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // 5. 释放资源
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    // --- 验证结果与性能对比 ---
    std::cout << "\nVerifying results..." << std::endl;
    if (verify_results(h_matrix_cpu, h_matrix_gpu, N)) {
        std::cout << "Results are consistent!" << std::endl;
        std::cout << "Speedup: " << cpu_duration.count() / gpu_duration_ms << "x" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // 清理内存
    delete[] h_matrix_cpu;
    delete[] h_matrix_gpu;

    return 0;
}