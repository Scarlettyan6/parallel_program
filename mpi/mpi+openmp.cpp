#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>

#define MAX_N 2048
float A[MAX_N][MAX_N];

void init_matrix(int n) {
    srand(0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = (i == j) ? 1.0f : (rand() % 10 + 1);
}

// 任务划分方案 A: MPI 按行划分，OpenMP 并行行内消元（每行内列循环）
void gauss_scheme_A(int n, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            #pragma omp parallel for
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
        }
        MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(static)
        for (int i = rank; i < n; i += size) {
            if (i > k) {
                float factor = A[i][k];
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }
}

// 任务划分方案 B: MPI 按行划分，OpenMP 并行列循环
void gauss_scheme_B(int n, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            #pragma omp parallel for
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
        }
        MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (int i = rank; i < n; i += size) {
            if (i > k) {
                float factor = A[i][k];
                #pragma omp parallel for
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }
}

// 任务划分方案 C: MPI 负责二维块划分，OpenMP 对块内行并行
void gauss_scheme_C(int n, int rank, int size) {
    int rows_per_proc = n / size;
    int row_start = rank * rows_per_proc;
    int row_end = (rank + 1) * rows_per_proc;

    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
        }
        MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = row_start; i < row_end; ++i) {
            if (i > k) {
                float factor = A[i][k];
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }
}

// 任务划分方案 D: MPI 负责二维块划分，OpenMP 对块内列并行
void gauss_scheme_D(int n, int rank, int size) {
    int rows_per_proc = n / size;
    int row_start = rank * rows_per_proc;
    int row_end = (rank + 1) * rows_per_proc;

    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
        }
        MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (int i = row_start; i < row_end; ++i) {
            if (i > k) {
                float factor = A[i][k];
                #pragma omp parallel for
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "用法: ./program [A|B|C|D]" << std::endl;
        MPI_Finalize(); return 1;
    }
    char mode = argv[1][0];

    for (int n = 256; n <= 2048; n += 256) {
        if (rank == 0) init_matrix(n);
        MPI_Bcast(A, MAX_N * MAX_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        switch (mode) {
            case 'A': gauss_scheme_A(n, rank, size); break;
            case 'B': gauss_scheme_B(n, rank, size); break;
            case 'C': gauss_scheme_C(n, rank, size); break;
            case 'D': gauss_scheme_D(n, rank, size); break;
            default:
                if (rank == 0) std::cerr << "未知模式: " << mode << std::endl;
                MPI_Finalize(); return 1;
        }
        double t2 = MPI_Wtime();

        if (rank == 0) {
            std::cout << "Matrix Size: " << n
                      << ", Mode: " << mode
                      << ", Time: " << (int)((t2 - t1) * 1000) << " ms" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
