#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#define MAX_N 2048
#define STEP 256

float A[MAX_N][MAX_N];
float row_buffer[MAX_N];

void init_matrix(int n) {
    srand(0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = (i == j) ? 1.0f : (rand() % 10 + 1);
}

void gaussian_twosided(int n, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
            memcpy(row_buffer, A[k], sizeof(float) * n);
        }
        MPI_Bcast(row_buffer, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (int i = rank; i < n; i += size) {
            if (i > k) {
                float factor = A[i][k];
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * row_buffer[j];
                A[i][k] = 0.0f;
            }
        }
    }
}

void gaussian_onesided(int n, int rank, int size, MPI_Win win) {
    for (int k = 0; k < n; ++k) {
        if (rank == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0f;
        }
        MPI_Win_fence(0, win);

        if (rank != 0) {
            MPI_Get(row_buffer, n, MPI_FLOAT, 0, k * MAX_N, n, MPI_FLOAT, win);
        } else {
            memcpy(row_buffer, A[k], sizeof(float) * n);
        }
        MPI_Win_fence(0, win);

        for (int i = rank; i < n; i += size) {
            if (i > k) {
                float factor = A[i][k];
                for (int j = k + 1; j < n; ++j)
                    A[i][j] -= factor * row_buffer[j];
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
        if (rank == 0)
            std::cerr << "Usage: ./gauss_rma_compare [twosided|onesided]" << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Win win;
    MPI_Win_create(A, MAX_N * MAX_N * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    for (int n = STEP; n <= MAX_N; n += STEP) {
        if (rank == 0) init_matrix(n);
        MPI_Bcast(A, MAX_N * MAX_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        if (strcmp(argv[1], "twosided") == 0) {
            gaussian_twosided(n, rank, size);
        } else if (strcmp(argv[1], "onesided") == 0) {
            gaussian_onesided(n, rank, size, win);
        }

        double t2 = MPI_Wtime();
        if (rank == 0) {
            std::cout << "Matrix Size: " << n << ", Mode: " << argv[1] << ", Time: " << (long long)((t2 - t1) * 1000) << " ms" << std::endl;
        }
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
