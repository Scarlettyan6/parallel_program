#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <arm_neon.h>

#define MAX_N 2048
#define STEP 256
#define NUM_THREADS 4

float A[MAX_N][MAX_N];         // 原始矩阵（用于列并行）
float A_row[MAX_N][MAX_N];     // 行划分版本
float A_serial[MAX_N][MAX_N];  // 串行版本

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

void gaussian_parallel_row(int n) {
    int i, j, k;
    float tmp, temp;

    #pragma omp parallel num_threads(NUM_THREADS) private(i, j, k, tmp, temp)
    {
        for (k = 0; k < n; k++) {
            #pragma omp single
            {
                tmp = A_row[k][k];
                for (j = k + 1; j < n; j++)
                    A_row[k][j] /= tmp;
                A_row[k][k] = 1.0f;
            }

            #pragma omp for
            for (i = k + 1; i < n; i++) {
                temp = A_row[i][k];
                for (j = k + 1; j < n; j++)
                    A_row[i][j] -= temp * A_row[k][j];
                A_row[i][k] = 0.0f;
            }
        }
    }
}

void gaussian_neon_parallel(int n) {
    int i, j, k;
    float tmp;

    #pragma omp parallel num_threads(NUM_THREADS) private(i, j, k, tmp)
    {
        for (k = 0; k < n; k++) {
            #pragma omp single
            {
                tmp = A[k][k];
                for (j = k + 1; j < n; j++)
                    A[k][j] /= tmp;
                A[k][k] = 1.0f;
            }

            // 使用 Neon 优化：对每行进行 SIMD 加速
            #pragma omp for
            for (i = k + 1; i < n; i++) {
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
        }
    }
}

int main() {
    printf("Matrix Size\tSerial (ms)\tRow-Parallel (ms)\tNeon-Parallel (ms)\n");

    for (int n = STEP; n <= 2048; n += STEP) {
        init_matrix(n);

        double t1 = omp_get_wtime();
        gaussian_serial(n);
        double t2 = omp_get_wtime();
        gaussian_parallel_row(n); // uses A_row
        double t3 = omp_get_wtime();
        gaussian_neon_parallel(n); // uses A
        double t4 = omp_get_wtime();

        printf("%4d\t\t%10.3f\t%16.3f\t%16.3f\n",
               n,
               (t2 - t1) * 1000,
               (t3 - t2) * 1000,
               (t4 - t3) * 1000);
    }

    return 0;
}
