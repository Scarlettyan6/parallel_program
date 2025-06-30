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

void gaussian_serial(int n) {
    for (int k = 0; k < n; ++k) {
        float pivot = A[k][k];
        for (int j = k + 1; j < n; ++j)
            A[k][j] /= pivot;
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        cout << "Matrix Size\tSerial Time (ms)" << endl;
        for (int n = STEP; n <= MAX_N; n += STEP) {
            init_matrix(n);
            double t1 = MPI_Wtime();
            gaussian_serial(n);
            double t2 = MPI_Wtime();
            cout << n << "\t\t" << (long long)((t2 - t1) * 1000) << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
