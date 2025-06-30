#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

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
                for (int j = k + 1; j < n; ++j)
                    A[k][j] /= pivot;
                A[k][k] = 1.0f;
            }

            MPI_Bcast(&A[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

            for (int i = rank; i < n; i += size) {
                if (i > k) {
                    float factor = A[i][k];
                    for (int j = k + 1; j < n; ++j)
                        A[i][j] -= factor * A[k][j];
                    A[i][k] = 0.0f;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        double t2 = MPI_Wtime();

        if (rank == 0) {
            cout << "Matrix Size: " << n << ", MPI Time (" << size << " procs): " << (long long)((t2 - t1) * 1000) << " ms" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
