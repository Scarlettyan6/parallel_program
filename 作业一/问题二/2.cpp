#include<iostream>
#include <windows.h>  
#include <fstream>
#include <stdlib.h>
using namespace std;

#define SIZE 1<<16
int* a;
int sum=0;
//ƽ���㷨
void add1(int n) {
	for (int i = 0; i < n; i++) {
		sum += a[i];
	}
}
//����·ʽ
void add2(int n) {
	int sum1 = 0;
	int sum2 = 0;
	for (int i = 0; i <n; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];
	}
}
//�ݹ�֮�ݹ麯������
//���ս������a[0]��
void add3(int n) {
	//n����1��ʾ�Ѿ��õ����ս������a[0]��
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

// ���Ժ���  
void test(int size) {
	// �����������  
	a = new int[size];
	for (int i = 0; i < size; ++i) {
		a[i] = 1;
	}
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	// ����ƽ���㷨  
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	add1(size);
	QueryPerformanceCounter(&end);
	double time1 = (end.QuadPart - start.QuadPart) * 1000.0/freq.QuadPart;

	// ���Զ���·ʽ�Ż��㷨  
	QueryPerformanceCounter(&start);
	add2(size);
	QueryPerformanceCounter(&end);
	double time2 = (end.QuadPart - start.QuadPart) * 1000.0/freq.QuadPart;

	// ���Եݹ��㷨һ  
	QueryPerformanceCounter(&start);
	add3(size);
	QueryPerformanceCounter(&end);
	double time3 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;




	// ������  
	//ofstream outputFile("1.1times.csv", ios::app);
	//outputFile << size << "," << time1 << "," << time2 << "," << time3 <<"," << "\n";
	//outputFile.close();

	delete[]a;

}

int main() {
	
	//Ϊ����ݹ飬Ԫ�ع�ģȡ2����(8-14)
	test((1<<14));
	
}