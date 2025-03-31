#include<iostream>
#include <windows.h>  
#include <fstream>
#include <stdlib.h>
using namespace std;
//ƽ���㷨
void innerp1(int** b, int* a, int n, int* res) {
	for (int i = 0; i < n; i++) {
		res[i] = 0;
		for (int j = 0; j < n; j++) {
			res[i] += b[j][i] * a[j];
		}
	}
}

//cache�Ż��㷨
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

//��һ��unroll�Ż��㷨������ѭ������
void innerp3(int** b, int* a, int n, int* res) {
		// ��ʼ���������  
		for (int i = 0; i < n; i++) {
			res[i] = 0;
		}

		// ʹ�ýϴ��չ�����ӣ�������Խ����  
		for (int j = 0; j < n; j++) {
			int i = 0;
			// ��ѭ��������4��Ԫ��  
			for (; i <= n - 4; i += 4) {
				res[i] += b[j][i] * a[j];
				res[i + 1] += b[j][i + 1] * a[j];
				res[i + 2] += b[j][i + 2] * a[j];
				res[i + 3] += b[j][i + 3] * a[j];
			}
			// ����ʣ���Ԫ��  
			for (; i < n; i++) {
				res[i] += b[j][i] * a[j];
			}
		}
}

// ���Ժ���  
void test(int size) {
	// ���ɶ�Ӧ����
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

	// ����ƽ���㷨  
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	innerp1(matrix, vec, size, res1);
	QueryPerformanceCounter(&end);
	double time1 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

	// ����cache�Ż��㷨  
	QueryPerformanceCounter(&start);
	innerp2(matrix, vec, size, res2);
	QueryPerformanceCounter(&end);
	double time2 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

	// ����unroll�Ż��㷨  
	QueryPerformanceCounter(&start);
	innerp3(matrix, vec, size, res2);
	QueryPerformanceCounter(&end);
	double time3 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	// ������  
	//ofstream outputFile("1.2times1.csv", ios::app);
	//outputFile << size << "," << time1 << "," << time2 <<"," <<time3<< "\n";
	//outputFile.close();

	for (int i = 0; i < size; ++i) {
		delete[] matrix[i]; // �ͷ�ÿһ��  
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
	//Ϊ����ݹ飬Ԫ�ع�ģȡ2����
	for (int size =100; size <=900; size+=100) {
		test((size));
	}
	return 0;
}