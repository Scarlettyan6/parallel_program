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
vector<float> gaussianElimination1(int n) {
    vector<float> x(n, 0.0f); // 存储解的向量

    // 消去过程
    for (int k = 0; k < n; k++) {
        // 行归一化
#pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            A[k][i] = A[k][i] / A[k][k];
        }

        b[k] /= A[k][k];
        A[k][k] = 1.0;


        // 下面的行消去

        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k]; // 计算消去因子

            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j]; // 更新当前行
            }
            A[i][k] = 0.0;
            b[i] -= factor * b[k]; // 更新常数向量
        }
    }

    // 回代过程
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum;
    }

    return x;
}

int main() {

    a_reset(200);

    vector<float> result1 = gaussianElimination1(200);
}
