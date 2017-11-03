/*This is the device memory kvbuffer, it is used
for storing kv data from shared memory and used for
job scheduling*/

#include "configuration.h"

#ifndef KVBUFFER
#define KVBUFFER

struct element 
{
	unsigned int k;
	unsigned int v;
	unsigned short k_size;
	unsigned short v_size;
};

typedef struct element Index;

struct kvbuffer
{
	Index index[KVINDEX_NUM];
	int memory_pool[KV_POOL_SIZE];
};

typedef struct kvbuffer Kvbuffer;


#endif
