#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <xmmintrin.h>  // 包含 SSE 的头文件

#define MAX_N 2048
#define STEP 256
#define NUM_THREADS 4

float A[MAX_N][MAX_N];         // 原始矩阵（用于列并行）
float A_row[MAX_N][MAX_N];     // 行划分版本
float A_serial[MAX_N][MAX_N];  // 串行版本

pthread_barrier_t barrier; // 用于线程同步

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
        if (k % NUM_THREADS == 0) {
            tmp = A_row[k][k];
            for (j = k + 1; j < n; j++)
                A_row[k][j] /= tmp;
            A_row[k][k] = 1.0f;
        }

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

void* gaussian_sse_parallel_pthread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    int n = data->n;

    int i, j, k;
    float tmp;

    for (k = 0; k < n; k++) {
        if (k % NUM_THREADS == 0) {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++)
                A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }

        // 使用 SSE 优化：对每行进行 SIMD 加速
        for (i = start_row; i < end_row; i++) {
            float factor = A[i][k];

            // 使用 SSE 加速，处理 4 个元素
            for (j = k + 1; j + 4 <= n; j += 4) {
                // 加载 A[i][j] 和 A[k][j] 到 SSE 寄存器
                __m128 row = _mm_loadu_ps(&A[i][j]);
                __m128 pivot_row = _mm_loadu_ps(&A[k][j]);

                // 执行 A[i][j] -= factor * A[k][j] 操作
                __m128 factor_vec = _mm_set1_ps(factor);
                pivot_row = _mm_mul_ps(pivot_row, factor_vec);
                row = _mm_sub_ps(row, pivot_row);

                // 存储结果回 A[i][j]
                _mm_storeu_ps(&A[i][j], row);
            }

            // 处理剩余的元素（如果 n 不能被 4 整除）
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
    printf("Matrix Size\tSerial (ms)\tRow-Parallel (ms)\tSSE-Parallel (ms)\n");

    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    for (int n = STEP; n <= 2048; n += STEP) {
        init_matrix(n);
        pthread_barrier_init(&barrier, NULL, NUM_THREADS);  // 初始化 barrier

        // 串行版本
        double t1 = omp_get_wtime();
        gaussian_serial(n);
        double t2 = omp_get_wtime();

        // 行并行版本（pthread）
        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start_row = (n / NUM_THREADS) * i;
            thread_data[i].end_row = (i == NUM_THREADS - 1) ? n : (n / NUM_THREADS) * (i + 1);
            thread_data[i].n = n;
            pthread_create(&threads[i], NULL, gaussian_parallel_row_pthread, (void*)&thread_data[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        double t3 = omp_get_wtime();

        // SSE 并行版本（pthread）
        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start_row = (n / NUM_THREADS) * i;
            thread_data[i].end_row = (i == NUM_THREADS - 1) ? n : (n / NUM_THREADS) * (i + 1);
            thread_data[i].n = n;
            pthread_create(&threads[i], NULL, gaussian_sse_parallel_pthread, (void*)&thread_data[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        double t4 = omp_get_wtime();

        printf("%4d\t\t%10.3f\t%16.3f\t%16.3f\n",
               n,
               (t2 - t1) * 1000,
               (t3 - t2) * 1000,
               (t4 - t3) * 1000);

        pthread_barrier_destroy(&barrier);  // 销毁 barrier
    }

    return 0;
}
