#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

cudaError_t getNumBlocksThreads(int *numBlocks, int *numThreads, int nr) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	*numThreads = deviceProp.maxThreadsPerBlock;
	*numBlocks = nr / (*numThreads);
	if (*numBlocks * (*numThreads) != nr) {
		(*numBlocks)++;
	}
	return cudaSuccess;
}

static __device__ size_t calculateHash(size_t key) {
	key -= (key << 6);
    key ^= (key >> 17);
    key -= (key << 9);
    key ^= (key << 4);
    key -= (key << 3);
    key ^= (key << 10);
    key ^= (key >> 15);
    return key;
}

static __global__ void insert_entry(HashElement *hashTable, int *keys,
	int *values, int *nrUpdates, int maxElements)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > maxElements) {
		return;
	}
	bool added = false;
	size_t computedHash = calculateHash(keys[idx]) % maxElements;
	while (!added) {
		int currentKey = atomicCAS(&hashTable[computedHash].key, 0, keys[idx]);
		if (currentKey == 0 || keys[idx] == currentKey) {
			if (currentKey == keys[idx]) {
				atomicAdd(nrUpdates, 1);
			}
			hashTable[computedHash].value = values[idx];
			added = true;
		}
		computedHash = (computedHash + 1) % maxElements;
	}
}
/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
float GpuHashTable::GpuHashTable(int size) {
	maxElements = size;
	nrElements = 0;
	cudaMallocManaged(&hashTable, maxElements * sizeof(*hashTable));
	cudaMemset(hashTable, 0, maxElements * sizeof(*hashTable));
}

GpuHashTable::loadFactor() {
	return nrElements / float(maxElements);
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashTable);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	return;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *keysCopy;
	int *valuesCopy;
	int *updates;
	int nrBlocks, nrThreads;
	cudaError_t err;
	err = cudaMallocManaged(&keysCopy, numKeys * sizeof(int));
	if (err) {
		fprintf(stderr, "cudaMalloc");
		return false;
	}
	err = cudaMemcpy(keysCopy, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err) {
		fprintf(stderr, "cudaMemcpy");
		return false;
	}
	err = cudaMallocManaged(&valuesCopy, numKeys * sizeof(int));
	if (err) {
		fprintf(stderr, "cudaMalloc");
		return false;
	}
	err = cudaMemcpy(valuesCopy, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err) {
		fprintf(stderr, "cudaMemcpy");
		return false;
	}
	err = cudaMallocManaged(&updates, sizeof(int));
	if (err) {
		fprintf(stderr, "cudaMalloc");
		return false;
	}
	if ((nrElements + numKeys) / float(maxElements) >= .9f) {
		reshape((nrElements + numKeys) / .85f);
	}
	err = getNumBlocksThreads(&nrBlocks, &nrThreads, numKeys);
	if (err) {
		fprintf(stderr, "getNumBlocksThreads");
		return false;
	}
	// insert part
	insert_entry<<<nrBlocks, nrThreads>>>(hashTable, keysCopy, valuesCopy, updates, maxElements);
	err = cudaDeviceSynchronize();
	if (err) {
		fprintf(stderr, "cudaDeviceSynchronize");
		return false;
	}
	nrElements += numKeys - *updates;
	err = cudaFree(keysCopy);
	if (err) {
		fprintf(stderr, "cudaFree");
		return false;
	}
	err = cudaFree(valuesCopy);
	if (err) {
		fprintf(stderr, "cudaFree");
		return false;
	}
	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	return 0;
}
