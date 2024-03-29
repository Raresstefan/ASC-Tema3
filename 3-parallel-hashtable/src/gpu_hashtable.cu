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
#define LOAD_FACTOR_MIN 0.8f
#define LOAD_FACTOR_MAX 0.9f

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

// Calculates hash for a key using the algorithm described here:
//https://burtleburtle.net/bob/hash/integer.html
static __device__ size_t calculateHash(int key) {
	size_t keyHash = (size_t) key;
    keyHash -= (keyHash << 6);
    keyHash ^= (keyHash >> 17);
    keyHash -= (keyHash << 9);
    keyHash ^= (keyHash << 4);
    keyHash -= (keyHash << 3);
    keyHash ^= (keyHash << 10);
    keyHash ^= (keyHash >> 15);
    return keyHash;
}

static __global__ void insert_entry(HashElement *hashTable, int *keys,
    int nrKeys, int *values, int *nrUpdates, int maxElements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrKeys) {
        return;
    }
    size_t computedHash = calculateHash(keys[idx]) % maxElements;
	int currentKey = atomicCAS(&hashTable[computedHash].key, 0, keys[idx]);
    while (currentKey != 0 && keys[idx] != currentKey) {
        computedHash = (computedHash + 1) % maxElements;
		currentKey = atomicCAS(&hashTable[computedHash].key, 0, keys[idx]);
    }
	if (currentKey == keys[idx]) {
		atomicAdd(nrUpdates, 1);
	}
	hashTable[computedHash].value = values[idx];
}

static __global__ void reshape_table(HashElement *oldTable, HashElement *newTable,
	int oldSize, int newSize)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > oldSize || oldTable[idx].key == 0) {
		return;
	}
	size_t computedHash = calculateHash(oldTable[idx].key) % newSize;
	bool readded = false;
	while (!readded) {
		if (atomicCAS(&newTable[computedHash].key, 0, oldTable[idx].key) == 0) {
			newTable[computedHash].value = oldTable[idx].value;
			readded = true;
		}
		computedHash = (computedHash + 1) % newSize;
	}
}

static __global__ void get_entry(HashElement *hashTable, int *keys,
	int nrKeys, int *values, int maxElements)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > nrKeys) {
		return;
	}
	size_t computedHash = calculateHash(keys[idx]) % maxElements;
	while (hashTable[computedHash].key != keys[idx]) {
		computedHash = (computedHash + 1) % maxElements;
	}
	values[idx] = hashTable[computedHash].value;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
    maxElements = size;
    nrElements = 0;
    glbGpuAllocator->_cudaMalloc((void **)&hashTable, maxElements * sizeof(*hashTable));
    cudaMemset(hashTable, 0, maxElements * sizeof(*hashTable));
}

float GpuHashTable::loadFactor() {
    return nrElements / float(maxElements);
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
    glbGpuAllocator->_cudaFree(hashTable);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	HashElement *reshaped;
	int nrBlocks, nrThreads;
	glbGpuAllocator->_cudaMallocManaged((void **)&reshaped, numBucketsReshape * sizeof(*reshaped));
	cudaMemset(reshaped, 0, numBucketsReshape * sizeof(*reshaped));
	getNumBlocksThreads(&nrBlocks, &nrThreads, maxElements);
	reshape_table<<<nrBlocks, nrThreads>>>(hashTable, reshaped, maxElements, numBucketsReshape);
	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(hashTable);
	hashTable = reshaped;
	maxElements = numBucketsReshape;
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
    glbGpuAllocator->_cudaMallocManaged((void **)&keysCopy, numKeys * sizeof(int));
    cudaMemcpy(keysCopy, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    glbGpuAllocator->_cudaMallocManaged((void **)&valuesCopy, numKeys * sizeof(int));
    cudaMemcpy(valuesCopy, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    glbGpuAllocator->_cudaMallocManaged((void **)&updates, sizeof(int));
    if ((nrElements + numKeys) / float(maxElements) >= LOAD_FACTOR_MAX) {
        reshape((nrElements + numKeys) / LOAD_FACTOR_MIN);
    }
    getNumBlocksThreads(&nrBlocks, &nrThreads, numKeys);
    insert_entry<<<nrBlocks, nrThreads>>>(hashTable, keysCopy, numKeys, valuesCopy, updates, maxElements);
    cudaDeviceSynchronize();
    nrElements += numKeys - *updates;
	glbGpuAllocator->_cudaFree(keysCopy);
    glbGpuAllocator->_cudaFree(valuesCopy);
	
    return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *values;
	int *keysCopy;
	int nrBlocks, nrThreads;
	glbGpuAllocator->_cudaMallocManaged((void **)&keysCopy, numKeys * sizeof(int));
	cudaMemcpy(keysCopy, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	glbGpuAllocator->_cudaMallocManaged((void **)&values, numKeys * sizeof(int));
	getNumBlocksThreads(&nrBlocks, &nrThreads, numKeys);
	get_entry<<<nrBlocks, nrThreads>>>(hashTable, keysCopy, numKeys, values, maxElements);
	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(keysCopy);
	return values;
}
