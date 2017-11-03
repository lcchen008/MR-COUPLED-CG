#ifndef ROGF
#define ROGF
#include "rog.h"
#include "util_device.h"
#include "hash.h"
#include "map.h"
#include "reduce.h"
#include "atomic.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

/*initialize the memory offsets for the global memory reduction object*/
void ginit(__local unsigned int * global_object_offset)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_group_id = local_id/WAVEFRONT_SIZE;
	if(local_id%WAVEFRONT_SIZE==0)
	global_object_offset[local_group_id] = ((float)global_id/(float)GLOBAL_THREADS)*MAX_POOL_G;
	//printf("thread %d: global_object_offset[%d]: %d\n", get_global_id(0), local_group_id, global_object_offset[local_group_id]);
}

int gmalloc(__global Reduction_Object_G *object, unsigned int size, local unsigned int *global_object_offset)
{
	size = align(size)/ALIGN_SIZE;
	unsigned int tid = get_local_id(0);
	unsigned int gid = tid/WAVEFRONT_SIZE;   //group id
	unsigned int offset = atomic_add(&global_object_offset[gid], size);
	return offset;
	//unsigned int offset = atomic_add(&(object->memory_offset), size);
	//return offset;
}

__global void * gget_address(__global Reduction_Object_G *object, int memory_index)
{
	return object->memory_pool + memory_index;
}

__global void * gget_key_address(__global Reduction_Object_G *object, int bucket_index)
{
	unsigned int key_index = (object->buckets)[bucket_index].x;
	//printf("***key_index is: %d\n", key_index);
	__global void *key_address = gget_address(object, key_index);
	//printf("***The key is: %d\n", *(__global int *)key_address);
	return key_address;
}

unsigned int gget_key_size(__global Reduction_Object_G *object, int bucket_index)
{
	return (object->key_size_per_bucket)[bucket_index];
}

__global void * gget_value_address(__global Reduction_Object_G *object, int bucket_index)
{
	unsigned int value_index = (object->buckets)[bucket_index].y;
	return gget_address(object, value_index);
}

unsigned int gget_value_size(__global Reduction_Object_G *object, int bucket_index)
{
	return (object->value_size_per_bucket)[bucket_index];
}

/*insert from local to global*/
bool ginsert_from_local(__global Reduction_Object_G *object, __local void *key, 
unsigned short key_size, __local void *value, unsigned short value_size, 
__local unsigned int *global_object_offset)
{
	unsigned int h = hash_local(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;

	int DoWork = 1;
	bool ret = true;
	int stride = 2;


	//printf("key is %d and index is %d\n", *(__local int *)key, index);

	//printf("k-v pair come...\n");

	//int key_test;
	//copyVal_local_to_private(&key_test, key, key_size);
	//printf("from local, key is: %d\n", key_test);

	while(!finish)
	{
		//printf("not finish...\n");
		
		DoWork = 1;

		while(DoWork)
		{
			if(global_getlock(&(object->locks)[index]))
			{
				if((object->pairs_per_bucket)[index]==0)
				{
					int k = gmalloc(object, key_size, global_object_offset);
					if(k == -1)
						ret = false;

					int v = gmalloc(object, value_size, global_object_offset);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						global_releaselock(&(object->locks)[index]);
						finish = 1;
						ret = false;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

					(object->key_size_per_bucket)[index] = key_size;
					(object->value_size_per_bucket)[index] = value_size;
					(object->pairs_per_bucket)[index] = 1;

					global char *key_data_start = (global char *)gget_address(object, k);

					//store the key data and value data into the ro
					//global void *key_data_start = key_size_address;
					global void *value_data_start = gget_address(object, v);

					copyVal_local_to_global(key_data_start, key, key_size);
					copyVal_local_to_global(value_data_start, value, value_size);

					global_releaselock(&(object->locks)[index]);
					finish = 1;
					DoWork = 0;
					//printf("success insert...\n");
				}

				/*compare them*/
				else
				{
					unsigned short size = gget_key_size(object, index);
					__global void *key_data = gget_key_address(object, index);

					size = ((object->key_size_per_bucket)[index]);

					/*if equal, conduct reduce*/
					if(equal_local_and_global(key, key_size, key_data, size))
					{
						unsigned short value_ro_size = gget_value_size(object, index);
						__global void *value_ro_data = gget_value_address(object, index);
						reduce_local_to_global(value_ro_data, value_ro_size, value, value_size);
						DoWork = 0;
						finish = 1;
						ret = true;
						global_releaselock(&(object->locks)[index]);
						//printf("success reduce...\n");
						
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = 0;
						//finish = 1;
						global_releaselock(&(object->locks)[index]);
						index = (index + stride)%NUM_BUCKETS_G;
						//printf("********new index...\n");
					}
				}
			}
		}
	}
	return ret;
}


/*insert from private to global directly*/
bool ginsert_from_private(__global Reduction_Object_G *object, void *key, 
unsigned short key_size, void *value, unsigned short value_size, 
__local unsigned int *global_object_offset)
{
	unsigned int h = hash(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;
	//volatile LONG_INT kvn;
	//kvn.first = kvn.second = 0;

	bool DoWork = true;
	bool ret = true;
	int stride = 1;

	while(!finish)
	{
		DoWork = 1;

		while(DoWork)
		{
			if(global_getlock(&((object->locks)[index])))
			{
				//printf("The pair per bucket: %d\n", (object->pairs_per_bucket)[index]);
				//if(*(long long *)&((object->buckets)[index])==0)
				if((object->pairs_per_bucket)[index]==0)
				{	
					//printf("index is: %d\n", index);

					int k = gmalloc(object, key_size, global_object_offset);
					if(k == -1)
						ret = false;

					int v = gmalloc(object, value_size, global_object_offset);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						global_releaselock(&(object->locks)[index]);
						finish = 1;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

				
					(object->key_size_per_bucket)[index] = key_size;
					(object->value_size_per_bucket)[index] = value_size;
					(object->pairs_per_bucket)[index] = 1;

					global char *key_size_address = (global char *)gget_address(object, k);

					//store the key data and value data into the ro
					global void *key_data_start = key_size_address;
					global void *value_data_start = gget_address(object, v);

					copyVal_private_to_global(key_data_start, key, key_size);
					copyVal_private_to_global(value_data_start, value, value_size);

					global_releaselock(&((object->locks)[index]));

					finish = 1;
					DoWork = false;
				}

				/*compare them*/
				else
				{
					//printf("******not zero**********\n");
					unsigned int size = gget_key_size(object, index);
					global void *key_data = gget_key_address(object, index);

					//size = ((object->key_size_per_bucket)[index]);
					//printf("index is: %d key1 is: %d and size is: %d\n", index, key1, size);

					/*if equal, conduct reduce*/
					if(equal_private_and_global(key, key_size, key_data, size))
					{
						//printf("I am thread %d and I am reducing at bucket %d\n", get_global_id(0), index);
						unsigned int value_ro_size = gget_value_size(object, index);
						global void *value_ro_data = gget_value_address(object, index);

						reduce_private_to_global(value_ro_data, value_ro_size, value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						global_releaselock(&((object->locks)[index]));
						//printf("I am thread %d and I release the lock at bucket %d\n", get_global_id(0), index);
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = false;
						//finish = true;
						global_releaselock(&((object->locks)[index]));
						index = (index + stride)%NUM_BUCKETS_G;
					}
				}
			}
		}
		if(finish)
			return ret;
	}
	return ret;
}



/*insert from global (value in private) to global directly*/
bool ginsert_from_gp(__global Reduction_Object_G *object, __global void *key, 
unsigned short key_size, void *value, unsigned short value_size, 
__local unsigned int *global_object_offset)
{
	unsigned int h = hash_global(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;
	//volatile LONG_INT kvn;
	//kvn.first = kvn.second = 0;

	bool DoWork = true;
	bool ret = true;
	int stride = 1;

	while(!finish)
	{
		DoWork = 1;
		
		while(DoWork)
		{
			if(global_getlock(&((object->locks)[index])))
			{
				//printf("The pair per bucket: %d\n", (object->pairs_per_bucket)[index]);
				//if(*(long long *)&((object->buckets)[index])==0)
				if((object->pairs_per_bucket)[index]==0)
				{	
					//printf("index is: %d\n", index);

					int k = gmalloc(object, key_size, global_object_offset);
					if(k == -1)
						ret = false;

					int v = gmalloc(object, value_size, global_object_offset);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						global_releaselock(&(object->locks)[index]);
						finish = 1;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

				
					(object->key_size_per_bucket)[index] = key_size;
					(object->value_size_per_bucket)[index] = value_size;
					(object->pairs_per_bucket)[index] = 1;

					global char *key_size_address = (global char *)gget_address(object, k);

					//store the key data and value data into the ro
					global void *key_data_start = key_size_address;
					global void *value_data_start = gget_address(object, v);

					copyVal_global_to_global(key_data_start, key, key_size);
					copyVal_private_to_global(value_data_start, value, value_size);

					global_releaselock(&((object->locks)[index]));

					finish = 1;
					DoWork = false;

					//printf("success map...\n");
				}

				/*compare them*/
				else
				{
					unsigned int size = gget_key_size(object, index);
					global void *key_data = gget_key_address(object, index);

					//size = ((object->key_size_per_bucket)[index]);

					/*if equal, conduct reduce*/
					if(equal_global_and_global(key, key_size, key_data, size))
					{
						unsigned short value_ro_size = gget_value_size(object, index);
						global void *value_ro_data = gget_value_address(object, index);

						reduce_private_to_global(value_ro_data, value_ro_size, value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						global_releaselock(&((object->locks)[index]));
						//printf("success reduce...\n");
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = false;
						global_releaselock(&((object->locks)[index]));
						index = (index + stride)%NUM_BUCKETS_G;
						//finish = true;
						//printf("new index...\n");
					}
				}
			}
		}
		if(finish)
			return ret;
	}
	return ret;
}


int gget_compare_value(__global Reduction_Object_G *object, unsigned int bucket_index1, unsigned int bucket_index2)
{
	LONG_INT bucket1 = (object->buckets)[bucket_index1];
	LONG_INT bucket2 = (object->buckets)[bucket_index2];

	int compare_value = 0;

	if(bucket1.y==0&&bucket2.y==0)
		compare_value = 0;

	else if(bucket2.y==0)
		compare_value = -1;

	else if(bucket1.y==0)
		compare_value = 1;

	else
	{
		unsigned int key_size1 = gget_key_size(object, bucket_index1);
		//unsigned int value_size1 = lget_value_size(object, bucket_index1);
		unsigned int key_size2 = gget_key_size(object, bucket_index2);
		//unsigned int value_size2 = lget_value_size(object, bucket_index2);
		__global void *key_addr1 = gget_key_address(object, bucket_index1);
		__global void *key_addr2 = gget_key_address(object, bucket_index2);

		unsigned int value_size1 = gget_value_size(object, bucket_index1);
		//unsigned int value_size1 = lget_value_size(object, bucket_index1);
		unsigned int value_size2 = gget_value_size(object, bucket_index2);
		//unsigned int value_size2 = lget_value_size(object, bucket_index2);
		__global void *value_addr1 = lget_value_address(object, bucket_index1);
		__global void *value_addr2 = lget_value_address(object, bucket_index2);

		compare_value = compare_global(key_addr1, key_size1, key_addr2, key_size2, value_addr1, value_size1, value_addr2, value_size2);
	}

	return compare_value;
}

void gswap_bucket(__local Reduction_Object_S *object, unsigned int a, unsigned int b)
{
	int xtmp = (object->buckets)[a].x;
	int ytmp = (object->buckets)[a].y;

	(object->buckets)[a].x = (object->buckets)[b].x;
	(object->buckets)[a].y = (object->buckets)[b].y;

	(object->buckets)[b].x = xtmp;
	(object->buckets)[b].y = ytmp;
}

/*bitonic sort in the global reduction object*/
void gbitonic_merge(__global Reduction_Object_G *object, unsigned int k, unsigned int j)
{
	const unsigned int id = get_global_id(0);
	const unsigned int ixj = id ^ j;

	int compare = 0;

	if(ixj < NUM_BUCKETS_G && id < NUM_BUCKETS_G)
	if(ixj > id)
	{
		compare = get_compare_value(object, id, ixj);
		if((id&k)==0)
		{
			if(compare > 0)
			{
				swap(object, id, ixj);
			}
		}
		else
		{
			if(compare < 0)
			{
				swap(object, id, ixj);
			}
		}
	}
}

void remove(__local Reduction_Object_S *object, unsigned int bucket_index)
{
	(object->buckets)[bucket_index].x = 0;
	(object->buckets)[bucket_index].y = 0;
}


#endif