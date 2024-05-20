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
	double *C = (double*)malloc(N2 * sizeof(double));
	double *D = (double*)malloc(N2 * sizeof(double));

	// Multiply A^T with B and store the result in C
	register double *ptr_original_c = C;
	for (register double *original_a = A; original_a < A + N; original_a++) {
		register double *ptr_c = ptr_original_c;
		register int limit = (original_a - A) * N;

    	for (register double *original_b = B; original_b < B + N; original_b++) {
            register double suma = 0;

            for (register double *pa = original_a, *pb = original_b;
				 (pa - original_a) <= limit; pa += N, pb += N) {
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

		for (register double *original_a = A; original_a < A + N; original_a++) {
			register double suma = 0;
			register int limit = original_a - A;

			for (register double *pa = original_a, *pb = original_b;
				 (pb - original_b) <= limit; pb++, pa += N) {
				suma += *pb * *pa;
			}

			*ptr_d = suma;
			ptr_d++;
		}

		ptr_original_d += N;
	}

	// Add C and D and store the result in C
	for (register double *pc = C, *pd = D; (pc - C) < N2; pc++, pd++) {
		*pc = *pc + *pd;
	}

	// Multiply C with B^T and store the result in D
	ptr_original_d = D;
    for (register double *original_c = C; original_c < C + N2; original_c += N) {
		register double *ptr_d = ptr_original_d;

        for (register double *original_b = B; original_b < B + N2; original_b += N) {
            register double suma = 0;

            for (register double *pb = original_b, *pc = original_c;
				 (pc - original_c) < N; pc++, pb++) {
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
