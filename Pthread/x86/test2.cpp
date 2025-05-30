#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_N 2048
#define STEP 256

float A[MAX_N][MAX_N];         // 原始矩阵（用于行并行）
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

    // 拷贝给串行版本
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
        if (k % 2 == 0) {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++)
                A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }

        // 每个线程处理不同的行
        for (i = start_row; i < end_row; i++) {
            temp = A[i][k];
            for (j = k + 1; j < n; j++)
                A[i][j] -= temp * A[k][j];
            A[i][k] = 0.0f;
        }
        
        // 同步所有线程
        pthread_barrier_wait(&barrier);
    }

    return NULL;
}

int main() {
    printf("N\t\t2\t4\t6\t8\t10\n");  // 输出列标题

    // 设定不同的矩阵规模
    for (int n = 256; n <= 2024; n += 256) {
        init_matrix(n);  // 初始化矩阵

        // 记录串行计算时间
        double t1 = omp_get_wtime();
        gaussian_serial(n);
        double t2 = omp_get_wtime();

        // 记录并行计算时间
        printf("%d\t\t", n);
        for (int threads = 2; threads <= 10; threads += 2) {
            init_matrix(n);  // 重置矩阵

            pthread_t* thread_ids = (pthread_t*)malloc(threads * sizeof(pthread_t));
            thread_data_t* thread_data = (thread_data_t*)malloc(threads * sizeof(thread_data_t));

            pthread_barrier_init(&barrier, NULL, threads);  // 初始化 barrier

            double t3 = omp_get_wtime();

            // 为每个线程分配任务
            for (int i = 0; i < threads; i++) {
                thread_data[i].start_row = (n / threads) * i;
                thread_data[i].end_row = (i == threads - 1) ? n : (n / threads) * (i + 1);
                thread_data[i].n = n;
                pthread_create(&thread_ids[i], NULL, gaussian_parallel_row_pthread, (void*)&thread_data[i]);
            }

            // 等待所有线程完成
            for (int i = 0; i < threads; i++) {
                pthread_join(thread_ids[i], NULL);
            }

            double t4 = omp_get_wtime();
            printf("%lld\t", (long long)((t4 - t3) * 1000));  // 转换为毫秒并输出

            // 清理资源
            free(thread_ids);
            free(thread_data);
            pthread_barrier_destroy(&barrier);  // 销毁 barrier
        }
        printf("\n");
    }

    return 0;
}
