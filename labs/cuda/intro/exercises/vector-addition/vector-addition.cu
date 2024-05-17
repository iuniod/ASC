#include <stdio.h>
#include <math.h>
#include "utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;

    float *host_array_a = 0;
    float *host_array_b = 0;
    float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    // TODO 1: Allocate the host's arrays
    cudaMallocManaged(&host_array_a, N * sizeof(float));
    cudaMallocManaged(&host_array_b, N * sizeof(float));
    cudaMallocManaged(&host_array_c, N * sizeof(float));

    // TODO 2: Allocate the device's arrays
    cudaMalloc(&device_array_a, N * sizeof(float));
    cudaMalloc(&device_array_b, N * sizeof(float));
    cudaMalloc(&device_array_c, N * sizeof(float));

    // TODO 3: Check for allocation errors
    if (host_array_a == 0 || host_array_b == 0 || host_array_c == 0 ||
        device_array_a == 0 || device_array_b == 0 || device_array_c == 0) {
        printf("Error allocating memory\n");
        return 1;
    }

    // TODO 4: Fill array with values; use fill_array_float to fill
    // host_array_a and fill_array_random to fill host_array_b. Each
    // function has the signature (float *a, int n), where n = the size.
    fill_array_float(host_array_a, N);
    fill_array_random(host_array_b, N);

    // TODO 5: Copy the host's arrays to device
    cudaMemcpy(device_array_a, host_array_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_b, host_array_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // TODO 6: Execute the kernel, calculating first the grid size
    // and the amount of threads in each block from the grid
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    add_arrays<<<grid_size, block_size>>>(device_array_a, device_array_b, device_array_c, N);

    // TODO 7: Copy back the results and then uncomment the checking function
    cudaMemcpy(host_array_c, device_array_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_task_3(host_array_a, host_array_b, host_array_c, N);

    // TODO 8: Free the memory
    cudaFree(host_array_a);
    cudaFree(host_array_b);
    cudaFree(host_array_c);
    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);

    return 0;
}