/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double*)malloc(N * N * sizeof(double));
	double *D = (double*)malloc(N * N * sizeof(double));

	// Multiply A^T with B and store the result in C
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k <= i; k++) {
				C[i * N + j] += A[k * N + i] * B[k * N + j];
			}
		}
	}

	// Multiply B with A and store the result in D
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			D[i * N + j] = 0;
			for (int k = 0; k <= j; k++) {
				D[i * N + j] += B[i * N + k] * A[k * N + j];
			}
		}
	}

	// Add C and D and store the result in C
	for (int i = 0; i < N * N; i++) {
		C[i] += D[i];
	}

	// Multiply C with B^T and store the result in D
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			D[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				D[i * N + j] += C[i * N + k] * B[j * N + k];
			}
		}
	}

	free(C);

	return D;
}
