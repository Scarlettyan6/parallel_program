#include <iostream>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define MAX_N 2048
#define STEP 256

float A[MAX_N][MAX_N];

void init_matrix(int n) {
    srand(0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = (i == j) ? 1.0f : (rand() % 10 + 1);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int n = STEP; n <= MAX_N; n += STEP) {
        if (rank == 0) init_matrix(n);
        MPI_Bcast(A, MAX_N * MAX_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        for (int k = 0; k < n; ++k) {
            if (rank == 0) {
                float pivot = A[k][k];
                __m256 pivot_vec = _mm256_set1_ps(pivot);

                int j = k + 1;
                for (; j + 8 <= n; j += 8) {
                    __m256 row = _mm256_loadu_ps(&A[k][j]);
                    row = _mm256_div_ps(row, pivot_vec);
                    _mm256_storeu_ps(&A[k][j], row);
                }
                for (; j < n; ++j)
                    A[k][j] /= pivot;

                A[k][k] = 1.0f;
            }

            // 广播主元行
            MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // 并行处理每个进程负责的行
            #pragma omp parallel for schedule(static)
            for (int i = rank; i < n; i += size) {
                if (i > k) {
                    float factor = A[i][k];
                    __m256 factor_vec = _mm256_set1_ps(factor);

                    int j = k + 1;
                    for (; j + 8 <= n; j += 8) {
                        __m256 a_ij = _mm256_loadu_ps(&A[i][j]);
                        __m256 a_kj = _mm256_loadu_ps(&A[k][j]);
                        __m256 mul = _mm256_mul_ps(factor_vec, a_kj);
                        __m256 res = _mm256_sub_ps(a_ij, mul);
                        _mm256_storeu_ps(&A[i][j], res);
                    }
                    for (; j < n; ++j)
                        A[i][j] -= factor * A[k][j];

                    A[i][k] = 0.0f;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        double t2 = MPI_Wtime();

        if (rank == 0)
            cout << "Matrix Size: " << n << ", MPI+OpenMP+AVX Time (" << size << " procs): "
                 << (long long)((t2 - t1) * 1000) << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}