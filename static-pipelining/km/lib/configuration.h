#ifndef CONFIGURATION
#define CONFIGURATION

#define NUM_BUCKETS_C 101			//number of buckets in a CPU ro

#define NUM_BUCKETS_G 101
#define NUM_BUCKETS_S 101
#define MAX_POOL_S 960
#define MAX_POOL_G 960
#define MAX_POOL_C 960				//pool size in GPU ro
#define ALIGN_SIZE 4
#define KVINDEX_NUM 20000			//number of key_value pairs in the buffer
#define KV_POOL_SIZE  200000		//memory pool size
#define NUM_BUFFERS_IN_BLOCK 4		//each block has 2 buffers//number of int space in device memory

#define TASK_BLOCK_SIZE 1000 //each time schedule one block to an idle thread block for mapping

#define CPU_LOCAL_THREADS 1
#define CPU_GLOBAL_THREADS 4

#define GPU_LOCAL_THREADS 256
#define GPU_GLOBAL_THREADS 1280
#define NUM_GROUPS 1
#define WAVEFRONT_SIZE 64
#define USE_LOCAL

#define TYPE_GPU 0
#define TYPE_CPU 1

//#define CMGR               //cpu does map and gpu does reduce

#endif
