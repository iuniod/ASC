#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

__device__ bool globalFound;

// TODO: Implement function to search for all nonces from 1 through MAX_NONCE (inclusive) using CUDA Threads
__global__ void findNonce(BYTE *d_block_content, size_t current_length, BYTE *d_DIFFICULTY, uint64_t* d_nonce) {
	if (globalFound) {
		return;
	}

	char nonce_str[NONCE_SIZE];
	uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
	BYTE block_hash[SHA256_HASH_SIZE];
	BYTE block_content[BLOCK_SIZE];
	d_strcpy((char*)block_content, (const char*)d_block_content);

	if (index > MAX_NONCE || index < 1) {
		return;
	}

	intToString(index, nonce_str);
	d_strcpy((char*)block_content + current_length, nonce_str);
	apply_sha256(block_content, d_strlen((const char*)block_content), block_hash, 1);

	if (compare_hashes(block_hash, d_DIFFICULTY) <= 0) {
		if (globalFound) {
			return;
		}
		atomicOr((unsigned int*)&globalFound, true);
		atomicExch(reinterpret_cast<unsigned long long*>(d_nonce), index);
	}
}

int main(int argc, char **argv) {
	BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
			tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
			tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
	BYTE block_hash[SHA256_HASH_SIZE] = "0000000000000000000000000000000000000000000000000000000000000000";
	uint64_t nonce = 0;
	size_t current_length;

	// Top hash
	apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
	apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
	apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
	apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
	strcpy((char *)tx12, (const char *)hashed_tx1);
	strcat((char *)tx12, (const char *)hashed_tx2);
	apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
	strcpy((char *)tx34, (const char *)hashed_tx3);
	strcat((char *)tx34, (const char *)hashed_tx4);
	apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
	strcpy((char *)tx1234, (const char *)hashed_tx12);
	strcat((char *)tx1234, (const char *)hashed_tx34);
	apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

	// prev_block_hash + top_hash
	strcpy((char*)block_content, (const char*)prev_block_hash);
	strcat((char*)block_content, (const char*)top_hash);
	current_length = strlen((char*) block_content);

	cudaEvent_t start, stop;
	startTiming(&start, &stop);

	// copy from host to device
	BYTE *d_DIFFICULTY;
	cudaMalloc(&d_DIFFICULTY, SHA256_HASH_SIZE);
	cudaMemcpy(d_DIFFICULTY, DIFFICULTY, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);

	BYTE *d_block_content;
	cudaMalloc(&d_block_content, BLOCK_SIZE);
	cudaMemcpy(d_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

	uint64_t* d_nonce;
	cudaMalloc(&d_nonce, sizeof(uint64_t));
	cudaMemcpy(&d_nonce, &nonce, sizeof(uint64_t), cudaMemcpyHostToDevice);

	// initialize global variable
	cudaMemset(&globalFound, false, sizeof(bool));

	int threadsPerBlock = 256;
    int numBlocks = (MAX_NONCE + threadsPerBlock - 1) / threadsPerBlock;
	findNonce<<<numBlocks, threadsPerBlock>>>(d_block_content, current_length, d_DIFFICULTY, d_nonce);

	// wait for kernel to finish
	cudaDeviceSynchronize();

	// copy nonce back to host
	cudaMemcpy(&nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	printf("Nonce: %lu\n", nonce);


	// update block_hash
	sprintf((char*)block_content + current_length, "%lu", nonce);
	strcpy((char*)block_content + current_length + strlen((char*)block_content + current_length), "\0");
	apply_sha256(block_content, strlen((const char*)block_content), block_hash, 1);


	float seconds = stopTiming(&start, &stop);
	printResult(block_hash, nonce, seconds);

	return 0;
}
