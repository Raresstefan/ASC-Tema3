#ifndef _HASHCPU_
#define _HASHCPU_

struct HashElement {
	int key, value;
};

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		HashElement *hashTable;
		int maxElements;
		int nrElements;
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		float loadFactor();
		~GpuHashTable();
};

#endif
