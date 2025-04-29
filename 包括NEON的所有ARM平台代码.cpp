#include<iostream>
#include<vector>
#include <arm_neon.h>  
#include<chrono>
#include <omp.h> 
#include <algorithm>  
using namespace std;

const int BLOCK_SIZE = 64;
const int N=2024;
float A[N][N]; // 系数矩阵  
float b[N]; // 常数项向量  
// 全局变量声明  
alignas(64) float B[N][N];
alignas(64) float c[N];

int NUMBER[]={200,400,600,800,1000,1200};
void a_reset(int n){
  for(int i=0;i<i;i++){
    for(int j=i+1;j<n;j++)
    {
        A[i][j]=0.0;
        B[i][j]=0.0;
    }
    A[i][i]=1.0;
    B[i][i]=1.0;
    for(int j=i+1;j<n;j++){
       A[i][j]=rand();
       B[i][j]=A[i][j];
    }
    b[i]=rand();
    c[i]=rand();
  }
  for(int k=0;k<n;k++)
    for(int i=k+1;i<N;i++)	
      for(int j=0;j<N;j++)
        {  A[i][j]+=A[k][j];
        B[i][j]+=B[k][j];
        }
}

vector<float> gaussianEliminationCacheOpt(int n) {
    vector<float> x(n, 0.0f);
    for (int k = 0; k < n; k += BLOCK_SIZE) {
        int k_end = min(k + BLOCK_SIZE, n);
        // 行归一化  
        for (int i = k; i < k_end; ++i) {
            float divisor = A[i][i];
            for (int j = i; j < n; ++j) {
                A[i][j] /= divisor;
            }
            b[i] /= divisor;
            // 消去  
            for (int i2 = i + 1; i2 < n; ++i2) {
                float factor = A[i2][i];
                for (int j = i; j < n; ++j) {
                    A[i2][j] -= factor * A[i][j];
                }
                b[i2] -= factor * b[i];
            }
        }
    }
    // 逆向回代  
    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum;
    }
    return x;
}

vector<float> gaussianElimination1(int n) {  
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {
        // 行归一化  
        float divisor = A[k][k];
        for (int j = k; j < n; j++) {
            A[k][j] /= divisor; // 归一化当前行  
        }
        b[k] /= divisor; // 更新常数向量  

        // 下面的行消去  
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k]; // 计算消去因子  

            for (int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j]; // 更新当前行  
            }
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
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {
        // 行归一化（处理过程中进行4路加速）
        float divisor1=A[k][k];
        float32x4_t divisor = vdupq_n_f32(divisor1);
        for (int j = k; j < n-3; j += 4) {
            float32x4_t row_elements = vld1q_f32(&A[k][j]);
            row_elements = vdivq_f32(row_elements, divisor);
            vst1q_f32(&A[k][j], row_elements);
        }
        // 处理尾部剩余元素  
        for (int j = n - (n-k) % 4; j < n; j++) {
            A[k][j] /=divisor1;
        }
        b[k] /=divisor1;

        // 消去下方行(处理时进行4路加速)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j=k; j<n-3; j+=4) {
                float32x4_t row_elements = vld1q_f32(&A[i][j]);
                float32x4_t k_elements = vld1q_f32(&A[k][j]);
                // 将 -factor 转换为向量  
                float32x4_t factor_vector = vdupq_n_f32(-factor);
                // 执行逐位乘法，计算 k_elements 和 -factor 的乘积     
                float32x4_t temp = vmulq_f32(k_elements, factor_vector);
                //将结果加到 row_elements 上  
                row_elements = vaddq_f32(row_elements, temp);
                vst1q_f32(&A[i][j], row_elements);
            }
            // 处理尾部剩余元素  
            for (int j=n-((n-k) % 4); j < n; j++) {
                A[i][j] -= A[k][j] * factor;
            }
            b[i]-= factor * b[k];
        }
    }
    // 回代过程  
    x[n - 1] = b[n - 1];  // 最后一行的回代  
    for (int i = n - 2; i >= 0; i--) {  
        float32x4_t sum_vector = vdupq_n_f32(0.0f);  // 累加初始值  
        // 从 i + 1 开始，使用 4 路处理  
        for (int j = i + 1; j <=n-4; j += 4) {  
            float32x4_t a_vector = vld1q_f32(&A[i][j]);  // 加载 A[i][j] 的四个元素  
            float32x4_t x_vector = vld1q_f32(&x[j]);      // 加载 x[j] 的四个元素  
            float32x4_t product = vmulq_f32(a_vector, x_vector);  // 逐位相乘  
            sum_vector = vaddq_f32(sum_vector, product);  // 累加到 sum_vector  
        }  

        // 处理最后的累加结果  
        float sum_array[4];  
        vst1q_f32(sum_array, sum_vector);  // 存储在数组中  
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];  

        // 处理剩余的元素  
        for (int j = n-(n-(i + 1)) % 4; j < n; j++) {  
            sum += A[i][j] * x[j];  // 处理剩余元素  
        }  

        x[i] = b[i] - sum;  // 计算当前未知数  
    }  
    return x;
}
//只对第一部分并行
vector<float> gaussianElimination3(int n) {  
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {  
        // 行归一化（处理过程中进行4路加速）
        float32x4_t divisor = vdupq_n_f32(A[k][k]);  
        for (int j = k + 1; j < n; j += 4) {  
            float32x4_t row_elements = vld1q_f32(&A[k][j]);   
            row_elements = vdivq_f32(row_elements, divisor); 
            vst1q_f32(&A[k][j], row_elements); 
        }  
       // 处理尾部剩余元素  
        for (int j = n - (n - (k + 1)) % 4; j < n; j++) {  
            A[k][j] /= A[k][k];  
        }  
        b[k] /= A[k][k]; 
        A[k][k] = 1.0;   // 设定主元 

        //下面的行消去
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
    x[n-1] = b[n-1];  
    for (int i = n - 2; i >= 0; i--) {  
        float sum = b[i];  
        for (int j = i + 1; j < n; j++) {  
            sum -= A[i][j] * x[j];  
        }  
        x[i] = sum;  
    }  
    return x;
}
//只对第二部分并行
vector<float> gaussianElimination4(int n) {  
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {  
        //行归一化
        for(int i=k+1;i<n;i++){
            A[k][i]=A[k][i]/A[k][k];
        }
        b[k]/=A[k][k];
        A[k][k]=1.0;

        // 消去下方行(处理时进行4路加速)
        for (int i = k + 1; i < n; i++) {  
            float factor = A[i][k];  
            for (int j = k; j<n-3; j += 4) {  
                float32x4_t row_elements = vld1q_f32(&A[i][j]);  
                float32x4_t k_elements = vld1q_f32(&A[k][j]);  
                // 将 -factor 转换为向量  
                float32x4_t factor_vector = vdupq_n_f32(-factor); 
                // 执行逐位乘法，计算 k_elements 和 -factor 的乘积     
                float32x4_t temp = vmulq_f32(k_elements, factor_vector); 
                //将结果加到 row_elements 上  
                row_elements = vaddq_f32(row_elements, temp);  
                vst1q_f32(&A[i][j], row_elements);  
            } 
            // 处理尾部剩余元素  
            for (int j = n-(n-k) % 4; j < n; j++) {  
                A[i][j] -= A[k][j] * factor;   
            }  
            A[i][k] = 0.0; 
            b[i] -= factor * b[k]; 
        }  
    }  

    // 回代过程  
    x[n-1] = b[n-1];  
    for (int i = n - 2; i >= 0; i--) {  
        float sum = b[i];  
        for (int j = i + 1; j < n; j++) {  
            sum -= A[i][j] * x[j];  
        }  
        x[i] = sum;  
    }  
    return x;
}

//只对第三部分并行
vector<float> gaussianElimination5(int n) {  
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {  
        //行归一化
        for(int i=k+1;i<n;i++){
            A[k][i]=A[k][i]/A[k][k];
        }
        b[k]/=A[k][k];
        A[k][k]=1.0;
       // 处理尾部剩余元素  
        for (int j = n - (n - (k + 1)) % 4; j < n; j++) {  
            A[k][j] /= A[k][k];  
        }  
        b[k] /= A[k][k]; 
        A[k][k] = 1.0;   // 设定主元 

        //下面的行消去
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
    x[n - 1] = b[n - 1];  // 最后一行的回代  
    for (int i = n - 2; i >= 0; i--) {  
        float32x4_t sum_vector = vdupq_n_f32(0.0f);  // 累加初始值  
        // 从 i + 1 开始，使用 4 路处理  
        for (int j = i + 1; j <=n-4; j += 4) {  
            float32x4_t a_vector = vld1q_f32(&A[i][j]);  // 加载 A[i][j] 的四个元素  
            float32x4_t x_vector = vld1q_f32(&x[j]);      // 加载 x[j] 的四个元素  
            float32x4_t product = vmulq_f32(a_vector, x_vector);  // 逐位相乘  
            sum_vector = vaddq_f32(sum_vector, product);  // 累加到 sum_vector  
        }  

        // 处理最后的累加结果  
        float sum_array[4];  
        vst1q_f32(sum_array, sum_vector);  // 存储在数组中  
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];  

        // 处理剩余的元素  
        for (int j = n-(n-(i + 1)) % 4; j < n; j++) {  
            sum += A[i][j] * x[j];  // 处理剩余元素  
        }  
        x[i] = b[i] - sum;  // 计算当前未知数  
    }  
    return x;
}

vector<float> gaussianElimination6(int n) {  
    vector<float> x(n, 0.0f); // 存储解的向量
    // 消去过程  
    for (int k = 0; k < n; k++) {
        // 行归一化（处理过程中进行4路加速）
        float divisor1=B[k][k];
        float32x4_t divisor = vdupq_n_f32(divisor1);
        for (int j = k; j < n-3; j += 4) {
            float32x4_t row_elements = vld1q_f32(&B[k][j]);
            row_elements = vdivq_f32(row_elements, divisor);
            vst1q_f32(&B[k][j], row_elements);
        }
        // 处理尾部剩余元素  
        for (int j = n - (n-k) % 4; j < n; j++) {
            B[k][j] /=divisor1;
        }
        c[k] /=divisor1;

        // 消去下方行(处理时进行4路加速)
        for (int i = k + 1; i < n; i++) {
            float factor = B[i][k];
            for (int j=k; j<n-3; j+=4) {
                float32x4_t row_elements = vld1q_f32(&B[i][j]);
                float32x4_t k_elements = vld1q_f32(&B[k][j]);
                // 将 -factor 转换为向量  
                float32x4_t factor_vector = vdupq_n_f32(-factor);
                // 执行逐位乘法，计算 k_elements 和 -factor 的乘积     
                float32x4_t temp = vmulq_f32(k_elements, factor_vector);
                //将结果加到 row_elements 上  
                row_elements = vaddq_f32(row_elements, temp);
                vst1q_f32(&B[i][j], row_elements);
            }
            // 处理尾部剩余元素  
            for (int j=n-((n-k) % 4); j < n; j++) {
                B[i][j] -= B[k][j] * factor;
            }
            c[i]-= factor * c[k];
        }
    }
    // 回代过程  
    x[n - 1] = c[n - 1];  // 最后一行的回代  
    for (int i = n - 2; i >= 0; i--) {  
        float32x4_t sum_vector = vdupq_n_f32(0.0f);  // 累加初始值  
        // 从 i + 1 开始，使用 4 路处理  
        for (int j = i + 1; j <=n-4; j += 4) {  
            float32x4_t b_vector = vld1q_f32(&B[i][j]);  // 加载 A[i][j] 的四个元素  
            float32x4_t x_vector = vld1q_f32(&x[j]);      // 加载 x[j] 的四个元素  
            float32x4_t product = vmulq_f32(b_vector, x_vector);  // 逐位相乘  
            sum_vector = vaddq_f32(sum_vector, product);  // 累加到 sum_vector  
        }  

        // 处理最后的累加结果  
        float sum_array[4];  
        vst1q_f32(sum_array, sum_vector);  // 存储在数组中  
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];  

        // 处理剩余的元素  
        for (int j = n-(n-(i + 1)) % 4; j < n; j++) {  
            sum += B[i][j] * x[j];  // 处理剩余元素  
        }  

        x[i] = c[i] - sum;  // 计算当前未知数  
    }  
    return x;
}


int main(){
    // 定义一个线性方程组的增广矩阵 A 和 常数向量 b  
    for(int i:NUMBER){
        a_reset(i);
        auto Start=chrono::high_resolution_clock::now();
        vector<float> result= gaussianEliminationCacheOpt(i);
        auto End=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
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
        auto Start4=chrono::high_resolution_clock::now();
        vector<float> result4 = gaussianElimination4(i);
        auto End4=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed4 = End4 - Start4;
        auto Start5=chrono::high_resolution_clock::now();
        vector<float> result5 = gaussianElimination5(i);
        auto End5=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed5 = End5 - Start5;
        auto Start6=chrono::high_resolution_clock::now();
        vector<float> result6 = gaussianElimination6(i);
        auto End6=chrono::high_resolution_clock::now();
        chrono::duration<double,std::ratio<1,1000>>elapsed6 = End6 - Start6;
        cout<<"问题规模为"<<i<<",";
        cout<<elapsed1.count()<<",";
        cout<<elapsed.count()<<",";
        cout<<elapsed2.count()<<",";
        cout<<elapsed3.count()<<",";
        cout<<elapsed4.count()<<",";
        cout<<elapsed5.count()<<",";
        cout<<elapsed6.count()<<",";
    }
}