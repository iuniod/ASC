#include <stdio.h>
#include <math.h>
#include "utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;
    int num_bytes = N * sizeof(float);

    float *host_array_a = 0;
    float *host_array_b = 0;
    float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    // TODO 1: Allocate the host's arrays
    host_array_a = (float *)malloc(num_bytes);
    host_array_b = (float *)malloc(num_bytes);
    host_array_c = (float *)malloc(num_bytes);

    // TODO 2: Allocate the device's arrays
    cudaMalloc((void **)&device_array_a, num_bytes);
    cudaMalloc((void **)&device_array_b, num_bytes);
    cudaMalloc((void **)&device_array_c, num_bytes);

    // TODO 3: Check for allocation errors
    if (host_array_a == 0 || host_array_b == 0 || host_array_c == 0 ||
        device_array_a == 0 || device_array_b == 0 || device_array_c == 0) {
        printf("[*] Error in memory allocation\n");
        return 1;
    }

    // TODO 4: Fill array with values; use fill_array_float to fill
    // host_array_a and fill_array_random to fill host_array_b. Each
    // function has the signature (float *a, int n), where n = the size.
    fill_array_float(host_array_a, N);
    fill_array_random(host_array_b, N);

    // TODO 5: Copy the host's arrays to device
    cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_b, host_array_b, num_bytes, cudaMemcpyHostToDevice);

    // TODO 6: Execute the kernel, calculating first the grid size
    // and the amount of threads in each block from the grid
    const size_t block_size = 255;
    size_t blocks_no = N / block_size;

    if (N % block_size) 
		++blocks_no;

    add_arrays<<<blocks_no, block_size>>>(device_array_a, device_array_b, device_array_c, N);

    // TODO 7: Copy back the results and then uncomment the checking function
    cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);

    check_task_3(host_array_a, host_array_b, host_array_c, N);

    // TODO 8: Free the memory
    free(host_array_a);
    free(host_array_b);
    free(host_array_c);
    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);

    return 0;
}