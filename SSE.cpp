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
vector<float> gaussianElimination2(int n) {
    vector<float> x(n, 0.0f);

    for (int k = 0; k < n; k++) {
        float divisor1 = A[k][k];
        __m128 divisor = _mm_set1_ps(divisor1);

        // 行归一化
        for (int j = k; j < n - 3; j += 4) {
            __m128 row_elements = _mm_loadu_ps(&A[k][j]);
            row_elements = _mm_div_ps(row_elements, divisor);
            _mm_storeu_ps(&A[k][j], row_elements);
        }

        // 处理尾部剩余元素
        for (int j = n - (n - k) % 4; j < n; j++) {
            A[k][j] /= divisor1;
        }
        b[k] /= divisor1;

        // 消去下方行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m128 factor_vector = _mm_set1_ps(-factor);

            for (int j = k; j < n - 3; j += 4) {
                __m128 row_elements = _mm_loadu_ps(&A[i][j]);
                __m128 k_elements = _mm_loadu_ps(&A[k][j]);
                __m128 temp = _mm_mul_ps(k_elements, factor_vector);
                row_elements = _mm_add_ps(row_elements, temp);
                _mm_storeu_ps(&A[i][j], row_elements);
            }

            // 处理尾部剩余元素
            for (int j = n - ((n - k) % 4); j < n; j++) {
                A[i][j] -= A[k][j] * factor;
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代过程
    x[n - 1] = b[n - 1];

    for (int i = n - 2; i >= 0; i--) {
        __m128 sum_vector = _mm_setzero_ps();

        // 从 i + 1 开始，使用 4 路处理
        for (int j = i + 1; j <= n - 4; j += 4) {
            __m128 a_vector = _mm_loadu_ps(&A[i][j]);
            __m128 x_vector = _mm_loadu_ps(&x[j]);
            __m128 product = _mm_mul_ps(a_vector, x_vector);
            sum_vector = _mm_add_ps(sum_vector, product);
        }

        // 处理最后的累加结果
        float sum_array[4];
        _mm_storeu_ps(sum_array, sum_vector);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // 处理剩余的元素
        for (int j = n - (n - (i + 1)) % 4; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        x[i] = b[i] - sum;
    }

    return x;
}
int main() {

    a_reset(200);

    vector<float> result2 = gaussianElimination2(200);
}