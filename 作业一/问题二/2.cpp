#include<iostream>
#include <windows.h>  
#include <fstream>
#include <stdlib.h>
using namespace std;

#define SIZE 1<<16
int* a;
int sum=0;
//平凡算法
void add1(int n) {
	for (int i = 0; i < n; i++) {
		sum += a[i];
	}
}
//多链路式
void add2(int n) {
	int sum1 = 0;
	int sum2 = 0;
	for (int i = 0; i <n; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];
	}
}
//递归之递归函数方法
//最终结果放在a[0]中
void add3(int n) {
	//n等于1表示已经得到最终结果，在a[0]处
	if (n <= 1) {
		return;
	}
	else
	{	for (int i = 0; i < n / 2; i++)
		{
			a[i] += a[n - i - 1];
		}
	n = n / 2;
	add3(n);
     }
}

// 测试函数  
void test(int size) {
	// 生成随机数组  
	a = new int[size];
	for (int i = 0; i < size; ++i) {
		a[i] = 1;
	}
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	// 测试平凡算法  
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	add1(size);
	QueryPerformanceCounter(&end);
	double time1 = (end.QuadPart - start.QuadPart) * 1000.0/freq.QuadPart;

	// 测试多链路式优化算法  
	QueryPerformanceCounter(&start);
	add2(size);
	QueryPerformanceCounter(&end);
	double time2 = (end.QuadPart - start.QuadPart) * 1000.0/freq.QuadPart;

	// 测试递归算法一  
	QueryPerformanceCounter(&start);
	add3(size);
	QueryPerformanceCounter(&end);
	double time3 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;




	// 输出结果  
	//ofstream outputFile("1.1times.csv", ios::app);
	//outputFile << size << "," << time1 << "," << time2 << "," << time3 <<"," << "\n";
	//outputFile.close();

	delete[]a;

}

int main() {
	
	//为方便递归，元素规模取2的幂(8-14)
	test((1<<14));
	
}