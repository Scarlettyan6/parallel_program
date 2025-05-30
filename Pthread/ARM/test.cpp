#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <arm_neon.h>

#define MAX_N 2048
#define STEP 256
#define NUM_THREADS 4

float A[MAX_N][MAX_N];         // 原始矩阵（用于行并行）
float A_row[MAX_N][MAX_N];     // 行划分版本
float A_serial[MAX_N][MAX_N];  // 串行版本

pthread_barrier_t barrier; // 用于线程同步

// 线程数据结构
typedef struct {
    int start_row;
    int end_row;
    int n;
} thread_data_t;

void init_matrix(int n) {
    srand(0);  // 固定随机种子
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = (i == j) ? 1.0f : (rand() % 10 + 1);

    // 拷贝给其它版本
    memcpy(A_row, A, sizeof(float) * n * n);
    memcpy(A_serial, A, sizeof(float) * n * n);
}

void gaussian_serial(int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A_serial[k][k];
        for (int j = k + 1; j < n; j++)
            A_serial[k][j] /= pivot;
        A_serial[k][k] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = A_serial[i][k];
            for (int j = k + 1; j < n; j++)
                A_serial[i][j] -= factor * A_serial[k][j];
            A_serial[i][k] = 0.0f;
        }
    }
}

void* gaussian_parallel_row_pthread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    int n = data->n;

    int i, j, k;
    float tmp, temp;

    for (k = 0; k < n; k++) {
        tmp = A_row[k][k];
        for (j = k + 1; j < n; j++)
            A_row[k][j] /= tmp;
        A_row[k][k] = 1.0f;

        // 每个线程处理不同的行
        for (i = start_row; i < end_row; i++) {
            temp = A_row[i][k];
            for (j = k + 1; j < n; j++)
                A_row[i][j] -= temp * A_row[k][j];
            A_row[i][k] = 0.0f;
        }

        // 同步所有线程
        pthread_barrier_wait(&barrier);
    }

    return NULL;
}

// 使用 NEON 加速的行并行版本
void* gaussian_neon_parallel_pthread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    int n = data->n;

    int i, j, k;
    float tmp;

    for (k = 0; k < n; k++) {
        if (k % 2 == 0) {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++)
                A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }

        // 使用 Neon 优化：对每行进行 SIMD 加速
        for (i = start_row; i < end_row; i++) {
            float factor = A[i][k];
            // 使用 Neon SIMD 向量化操作
            for (j = k + 1; j + 4 <= n; j += 4) {
                float32x4_t row = vld1q_f32(&A[i][j]);
                float32x4_t pivot_row = vld1q_f32(&A[k][j]);
                row = vsubq_f32(row, vmulq_n_f32(pivot_row, factor));
                vst1q_f32(&A[i][j], row);
            }
            // 处理剩余的元素
            for (j = (n / 4) * 4; j < n; j++)
                A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }

        // 同步所有线程
        pthread_barrier_wait(&barrier);
    }

    return NULL;
}

int main() {
    printf("Matrix Size\tSerial (ms)\tRow-Parallel (ms)\tNeon-Parallel (ms)\n");

    for (int n = STEP; n <= 2048; n += STEP) {
        init_matrix(n);

        // 记录串行计算时间
        double t1 = omp_get_wtime();
        gaussian_serial(n);
        double t2 = omp_get_wtime();

        // 记录并行计算时间
        printf("%4d\t\t", n);
        
        // 用 pthread 实现行并行
        pthread_t* thread_ids = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
        thread_data_t* thread_data = (thread_data_t*)malloc(NUM_THREADS * sizeof(thread_data_t));

        pthread_barrier_init(&barrier, NULL, NUM_THREADS);  // 初始化 barrier

        double t3 = omp_get_wtime();
        
        // 为每个线程分配任务
        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start_row = (n / NUM_THREADS) * i;
            thread_data[i].end_row = (i == NUM_THREADS - 1) ? n : (n / NUM_THREADS) * (i + 1);
            thread_data[i].n = n;
            pthread_create(&thread_ids[i], NULL, gaussian_parallel_row_pthread, (void*)&thread_data[i]);
        }

        // 等待所有线程完成
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(thread_ids[i], NULL);
        }

        double t4 = omp_get_wtime();
        printf("%lld\t", (long long)((t4 - t3) * 1000));  // 转换为毫秒并输出

        // 用 pthread 实现 NEON 加速的行并行
        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start_row = (n / NUM_THREADS) * i;
            thread_data[i].end_row = (i == NUM_THREADS - 1) ? n : (n / NUM_THREADS) * (i + 1);
            thread_data[i].n = n;
            pthread_create(&thread_ids[i], NULL, gaussian_neon_parallel_pthread, (void*)&thread_data[i]);
        }

        // 等待所有线程完成
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(thread_ids[i], NULL);
        }

        double t5 = omp_get_wtime();
        printf("%lld\n", (long long)((t5 - t4) * 1000));  // 转换为毫秒并输出

        // 清理资源
        free(thread_ids);
        free(thread_data);
        pthread_barrier_destroy(&barrier);  // 销毁 barrier
    }

    return 0;
}
