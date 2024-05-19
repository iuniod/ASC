/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	register int N2 = N * N;
	int i, j, k;
	double *C = (double*)malloc(N2 * sizeof(double));
	double *D = (double*)malloc(N2 * sizeof(double));

	// Multiply A^T with B and store the result in C
    for (register double *original_b = B; original_b < B + N; original_b ++) {
        for (i = 0; i < N; i++) {
            register double suma = 0;
            register double *pa = A + i;
            register double *pb = original_b;
            for (k = 0; k <= i; k++, pa += N, pb += N) {
                suma += *pa * *pb;
            }
            C[i * N + (original_b - B)] = suma;
        }
    }

	// Multiply B with A and store the result in D
	for (register double *original_b = B; original_b < B + N2; original_b+=N) {
		for (i = 0; i < N; i++) {
			register double suma = 0;
			register double *pb = original_b;
			register double *pa = A + i;
			for (k = 0; k <= i; k++, pb++, pa += N) {
				suma += *pb * *pa;
			}
			D[i + (original_b - B)] = suma;
		}
	}

	// Add C and D and store the result in C
	double *pc = C;
	register double *pd = D;
	for (i = 0; i < N2; i++, pc++, pd++) {
		*pc = *pc + *pd;
	}

	// Multiply C with B^T and store the result in D
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            register double suma = 0;
            register double *pc = C + i * N;
            register double *pb = B + j * N;
            for (k = 0; k < N; k++, pc++, pb++) {
                suma += *pc * *pb;
            }
            D[i * N + j] = suma;
        }
    }

	free(C);

	return D;
}
