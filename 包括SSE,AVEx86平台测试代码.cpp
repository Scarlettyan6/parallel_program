#include <iostream>
#include <vector>
#include <iomanip> // 用于设置输出精度
#include <xmmintrin.h>
#include <immintrin.h>
#include<chrono>
using namespace std;
const int N=2024;
float A[N][N]; // 系数矩阵
float b[N]; // 常数项向量

int NUMBER[]={200,400,600,800,1000,1200};
void a_reset(int n){
  for(int i=0;i<i;i++){
    for(int j=i+1;j<n;j++)
    {
        A[i][j]=0.0;
    }
    A[i][i]=1.0;
    for(int j=i+1;j<n;j++){
       A[i][j]=rand();
    }
    b[i]=rand();
  }
  for(int k=0;k<n;k++)
    for(int i=k+1;i<N;i++)
      for(int j=0;j<N;j++)
        {  A[i][j]+=A[k][j];
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
    for(int i:NUMBER){
        a_reset(i);
        auto Start1=chrono::high_resolution_clock::now();
        vector<float> result1 = gaussianElimination1(i);
        auto End1=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed1 = End1 - Start1;
        auto Start2=chrono::high_resolution_clock::now();
        vector<float> result2 = gaussianElimination2(i);
        auto End2=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed2 = End2 - Start2;
        auto Start3=chrono::high_resolution_clock::now();
        vector<float> result3 = gaussianElimination3(i);
        auto End3=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed3 = End3 - Start3;
        cout<<"问题规模为"<<i<<" ";
        cout<<elapsed1.count()<<" ";
        cout<<elapsed2.count()<<" ";
        cout<<elapsed3.count()<<endl;
    }
}