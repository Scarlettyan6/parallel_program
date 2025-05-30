#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_N 2048
#define STEP 256

float A[MAX_N][MAX_N];         // 原始矩阵（用于行并行）
float A_serial[MAX_N][MAX_N];  // 串行版本

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

void gaussian_parallel_row(int n, int num_threads) {
    int i, j, k;
    float tmp, temp;

    #pragma omp parallel num_threads(num_threads) private(i, j, k, tmp, temp)
    {
        for (k = 0; k < n; k++) {
            #pragma omp single
            {
                tmp = A[k][k];
                for (j = k + 1; j < n; j++)
                    A[k][j] /= tmp;
                A[k][k] = 1.0f;
            }

            #pragma omp for
            for (i = k + 1; i < n; i++) {
                temp = A[i][k];
                for (j = k + 1; j < n; j++)
                    A[i][j] -= temp * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }
}

int main() {
    printf("N\t\t2\t4\t6\t8\t10\n");  // 输出列标题

    // 设定不同的矩阵规模
    for (int n =256; n <=2024; n += 256) {
        init_matrix(n);  // 初始化矩阵

        // 记录串行计算时间
        double t1 = omp_get_wtime();
        gaussian_serial(n);
        double t2 = omp_get_wtime();

        // 记录并行计算时间
        printf("%d\t\t", n);
        for (int threads = 2; threads <= 10; threads += 2) {
            init_matrix(n);  // 重置矩阵

            double t3 = omp_get_wtime();
            gaussian_parallel_row(n, threads); // 使用行并行
            double t4 = omp_get_wtime();

            // 输出每个线程数对应的运行时间（单位：毫秒）
            printf("%lld\t", (long long)((t4 - t3) * 1000));  // 转换为毫秒并输出
        }
        printf("\n");
    }

    return 0;
}

