#ifndef _HASHCPU_
#define _HASHCPU_

struct hashElement {
	int key, value;
};

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		hashElement *hashTable;
		int maxElements;
		int nrElements;
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();
};

#endif
