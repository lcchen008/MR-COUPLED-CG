#ifndef CONFIGURATION
#define CONFIGURATION

#define NUM_BUCKETS_C 101    //number of buckets in a CPU ro

#define NUM_BUCKETS_G 101
#define NUM_BUCKETS_S 101
#define MAX_POOL_S 960
#define MAX_POOL_G 960
#define MAX_POOL_C 960       //pool size in GPU ro
#define ALIGN_SIZE 4
#define KVINDEX_NUM 4096 
#define KV_POOL_SIZE  524288 
#define KVBUFFER_SIZE 32768			  //number of int space in device memory

#define SMALL_SIZE 2000
#define TASK_BLOCK_SIZE 200000  //each time schedule one block to an idle thread block

#define KVBUFFERS_SIZE 4096           //4096 ints space in local memory

#define CPU_LOCAL_THREADS 256
#define CPU_GLOBAL_THREADS 768

#define GPU_LOCAL_THREADS 256
#define GPU_GLOBAL_THREADS 1280
#define NUM_GROUPS 1
#define WAVEFRONT_SIZE 64
#define USE_LOCAL

#define TYPE_GPU 0
#define TYPE_CPU 1
#endif
