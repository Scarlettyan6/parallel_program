#include <iostream>
#include <vector>
#include <iomanip> // 用于设置输出精度
#include <xmmintrin.h>
#include <immintrin.h>
#include<chrono>
using namespace std;
const int N = 2024;
float A[N][N]; // 系数矩阵
float b[N]; // 常数项向量

int NUMBER[] = { 200,400,600,800,1000,1200 };
void a_reset(int n) {
    for (int i = 0; i < i; i++) {
        for (int j = i + 1; j < n; j++)
        {
            A[i][j] = 0.0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            A[i][j] = rand();
        }
        b[i] = rand();
    }
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i][j] += A[k][j];
            }
}
vector<float> gaussianElimination3(int n) {
    vector<float> x(n, 0.0f);
    // 消去过程
    for (int k = 0; k < n; ++k) {
        float divisor1 = A[k][k];
        __m256 divisor = _mm256_set1_ps(divisor1);

        // 行归一化（8路并行）
        int j = k;
        for (; j <= n - 8; j += 8) {
            __m256 row_elements = _mm256_loadu_ps(&A[k][j]);
            row_elements = _mm256_div_ps(row_elements, divisor);
            _mm256_storeu_ps(&A[k][j], row_elements);
        }

        // 处理尾部剩余元素
        for (; j < n; ++j) {
            A[k][j] /= divisor1;
        }
        b[k] /= divisor1;

        // 消去下方行（8路并行）
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(-factor);

            j = k;
            for (; j <= n - 8; j += 8) {
                __m256 A_i_j = _mm256_loadu_ps(&A[i][j]);
                __m256 A_k_j = _mm256_loadu_ps(&A[k][j]);
                __m256 temp = _mm256_mul_ps(A_k_j, factor_vec);
                A_i_j = _mm256_add_ps(A_i_j, temp);
                _mm256_storeu_ps(&A[i][j], A_i_j);
            }

            // 处理尾部剩余元素
            for (; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代过程
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        float sum = 0.0f;
        __m256 sum_vec = _mm256_setzero_ps();

        // 计算从i+1到n-1范围内能被8整除的最大区间
        int start = i + 1;
        int end = n - ((n - start) % 8); // 最后一个可被8整除的位置

        // 8路累加
        for (int j = start; j < end; j += 8) {
            __m256 A_i_j = _mm256_loadu_ps(&A[i][j]);
            __m256 x_j = _mm256_loadu_ps(&x[j]);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(A_i_j, x_j));
        }

        // 水平求和，将8个浮点数结果合并
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        for (int k = 0; k < 8; ++k) {
            sum += sum_array[k];
        }

        // 处理尾部剩余元素
        for (int j = end; j < n; ++j) {
            sum += A[i][j] * x[j];
        }

        // 计算当前解
        x[i] = b[i] - sum;
    }

    return x;
}

int main() {

    a_reset(200);

    vector<float> result2 = gaussianElimination3(200);
}