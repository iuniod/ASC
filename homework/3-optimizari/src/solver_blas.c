/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>
#include <memory.h>

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	double *C = (double*)malloc(N * N * sizeof(double));
	double *D = (double*)malloc(N * N * sizeof(double));

	// Multiply A^T with B and store the result in C
	memcpy(C, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, C, N);

	// Multiply B with A and store the result in D
	memcpy(D, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, D, N);

	// Add C and D and store the result in C
	cblas_daxpy(N * N, 1, D, 1, C, 1);

	// Multiply C with B^T and store the result in D
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, C, N, B, N, 0, D, N);

	free(C);

	return D;
}
