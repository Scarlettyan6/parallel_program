#include<iostream>
#include <windows.h>  
#include <fstream>
#include <stdlib.h>
using namespace std;
//平凡算法
void innerp1(int** b, int* a, int n, int* res) {
	for (int i = 0; i < n; i++) {
		res[i] = 0;
		for (int j = 0; j < n; j++) {
			res[i] += b[j][i] * a[j];
		}
	}
}

//cache优化算法
void innerp2(int** b, int* a, int n, int* res) {
	for (int i = 0; i < n; i++) {
		res[i] = 0;
	}
	for (int j= 0; j<n; j++) {
		for (int i= 0; i< n; i++) {
			res[i] += b[j][i] * a[j];
		}
	}
}

//进一步unroll优化算法，降低循环次数
void innerp3(int** b, int* a, int n, int* res) {
		// 初始化结果数组  
		for (int i = 0; i < n; i++) {
			res[i] = 0;
		}

		// 使用较大的展开因子，并避免越界检查  
		for (int j = 0; j < n; j++) {
			int i = 0;
			// 主循环，处理4个元素  
			for (; i <= n - 4; i += 4) {
				res[i] += b[j][i] * a[j];
				res[i + 1] += b[j][i + 1] * a[j];
				res[i + 2] += b[j][i + 2] * a[j];
				res[i + 3] += b[j][i + 3] * a[j];
			}
			// 处理剩余的元素  
			for (; i < n; i++) {
				res[i] += b[j][i] * a[j];
			}
		}
}

// 测试函数  
void test(int size) {
	// 生成对应数组
	int** matrix = new int* [size];
	for (int i = 0; i < size; ++i) {
		matrix[i] = new int[size];
		for (int t = 0; t < size; t++) {
			matrix[i][t] = 1;
		}
	}
	int* vec = new int[size];
	for (int i = 0; i < size; i++)
	{
		vec[i] = 1;
	};

	int* res1 = new int[size];
	int* res2 = new int[size];
	int* res3 = new int[size];

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	// 测试平凡算法  
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	innerp1(matrix, vec, size, res1);
	QueryPerformanceCounter(&end);
	double time1 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

	// 测试cache优化算法  
	QueryPerformanceCounter(&start);
	innerp2(matrix, vec, size, res2);
	QueryPerformanceCounter(&end);
	double time2 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

	// 测试unroll优化算法  
	QueryPerformanceCounter(&start);
	innerp3(matrix, vec, size, res2);
	QueryPerformanceCounter(&end);
	double time3 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	// 输出结果  
	//ofstream outputFile("1.2times1.csv", ios::app);
	//outputFile << size << "," << time1 << "," << time2 <<"," <<time3<< "\n";
	//outputFile.close();

	for (int i = 0; i < size; ++i) {
		delete[] matrix[i]; // 释放每一行  
	}
	delete[] matrix;
	delete[] vec;
	delete[] res1;
	delete[] res2;
	delete[] res3;
}
int main() {  
	//ofstream outputFile("1.2times1.csv");
	//outputFile << "n,time1,time2,time3\n";
	//outputFile.close();
	//为方便递归，元素规模取2的幂
	for (int size =100; size <=900; size+=100) {
		test((size));
	}
	return 0;
}