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
	int i, k;
	double *C = (double*)malloc(N2 * sizeof(double));
	double *D = (double*)malloc(N2 * sizeof(double));

	// Multiply A^T with B and store the result in C
	register double *ptr_original_c = C;
	for (register double *original_a = A; original_a < A + N; original_a++) {
		register double *ptr_c = ptr_original_c;
    	for (i = 0; i < N; i++) {
			register double *pa = original_a;
			register double *pb = B + i;
            register double suma = 0;
            for (k = 0; k <= (original_a - A); k++, pa += N, pb += N) {
                suma += *pa * *pb;
            }
            *ptr_c = suma;
			ptr_c++;
        }

		ptr_original_c += N;
    }

	// Multiply B with A and store the result in D
	register double *ptr_original_d = D;
	for (register double *original_b = B; original_b < B + N2; original_b+=N) {
		register double *ptr_d = ptr_original_d;
		for (i = 0; i < N; i++) {
			register double suma = 0;
			register double *pb = original_b;
			register double *pa = A + i;
			for (k = 0; k <= i; k++, pb++, pa += N) {
				suma += *pb * *pa;
			}
			*ptr_d = suma;
			ptr_d++;
		}

		ptr_original_d += N;
	}

	// Add C and D and store the result in C
	register double *pd = D;
	for (double *pc = C; (pc - C) < N2; pc++, pd++) {
		*pc = *pc + *pd;
	}

	// Multiply C with B^T and store the result in D
	ptr_original_d = D;
    for (register double *original_c = C; original_c < C + N2; original_c += N) {
		register double *ptr_d = ptr_original_d;
        for (i = 0; i < N2; i += N) {
            register double suma = 0;
            register double *pc = original_c;
            register double *pb = B + i;
            for (; (pc - original_c) < N; pc++, pb++) {
                suma += *pc * *pb;
            }
			*ptr_d = suma;
			ptr_d++;
        }

		ptr_original_d += N;
    }

	free(C);

	return D;
}
