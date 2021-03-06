#ifndef CONFIGURATION
#define CONFIGURATION

#define NUM_BUCKETS_G 128
#define NUM_BUCKETS_S 128
#define MAX_POOL_S 2048
#define MAX_POOL_G 4096
#define ALIGN_SIZE 4

#define LOCAL_THREADS 256
#define GLOBAL_THREADS 1280
#define NUM_GROUPS 1 
#define WAVEFRONT_SIZE 64
#define USE_LOCAL

#define USE_GPU
#define SORT_REMAIN 20

#ifdef USE_CPU
#define LOCAL_THREADS 1
#define GLOBAL_THREADS 4
#endif

#endif