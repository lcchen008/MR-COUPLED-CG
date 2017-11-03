#ifndef ROLF
#define ROLF

#include "rol.h"
#include "util_device.h"
#include "hash.h"
//#include "map.h"
#include "atomic.h"
#include "reduce.h"
#include "ds.h"


#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

void linit(__local Reduction_Object_S *object)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = LOCAL_THREADS/NUM_GROUPS;
	for(int index = local_id%group_size; index < NUM_BUCKETS_S; index += group_size)
	{
		(object->buckets)[index].x = 0;
		(object->buckets)[index].y = 0;
		(object->locks)[index] = 0;
	}

	if(local_id%group_size == 0)
	{
		object->num_buckets = NUM_BUCKETS_S;
		object->memory_offset = 0;
	}
}

int lmalloc(__local Reduction_Object_S *object, unsigned int size)
{
	size = align(size)/ALIGN_SIZE;
	unsigned int offset = atomic_add(&(object->memory_offset), size);
	if(offset + size > MAX_POOL_S)
		return -1;
	else
		return offset;
}

__local void * lget_address(__local Reduction_Object_S *object, unsigned memory_index)
{
	return object->memory_pool + memory_index;
}

__local void * lget_key_address(__local Reduction_Object_S *object, unsigned bucket_index)
{
	unsigned short key_index = ((object->buckets)[bucket_index]).x;
	__local char *size_address = lget_address(object, key_index);
	return size_address+2;   
}

unsigned short lget_key_size(__local Reduction_Object_S *object, unsigned bucket_index)
{
	unsigned short key_index = ((object->buckets)[bucket_index]).x;
	__local char *size_address = lget_address(object, key_index);
	return *size_address;    
}

__local void * lget_value_address(__local Reduction_Object_S *object, unsigned bucket_index)
{
	unsigned short value_index =  ((object->buckets)[bucket_index]).y;;
	return lget_address(object, value_index);
}

unsigned short lget_value_size(__local Reduction_Object_S *object, unsigned bucket_index)
{
	unsigned short key_index = ((object->buckets)[bucket_index]).x;
	__local char *size_address = lget_address(object, key_index);
	return *(size_address + 1);   
}

bool linsert(__local Reduction_Object_S *object, void *key, unsigned short key_size, void *value, unsigned short value_size)
{
	unsigned int h = hash(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;

	//printf("key is %d and index is %d\n", *(int *)key, index);

	unsigned int finish = 0;
	//unsigned int kvn = 0;

	int DoWork = 1;
	bool ret = true;
	int stride = 1;

	while(!finish)
	{
		if(object->num_buckets==0)
			return false;

		DoWork = 1;

		while(DoWork)
		{
			if(local_getlock(&((object->locks)[index])))
			{
				if(((object->buckets)[index]).x==0 && ((object->buckets)[index]).y==0)
				{
					int k = lmalloc(object, 2 + key_size);
					if(k == -1)
						ret = false;

					int v = lmalloc(object, value_size);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						local_releaselock(&(object->locks)[index]);
						finish  = 1;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

					__local char *key_size_address = (__local char *)lget_address(object, k);
					__local char *value_size_address = key_size_address + 1;
					*key_size_address = key_size;
					*value_size_address = value_size;

					//store the key data and value data into the ro
					__local void *key_data_start = key_size_address + 2;
					__local void *value_data_start = lget_address(object, v);

					copyVal_private_to_local(key_data_start, key, key_size);
					copyVal_private_to_local(value_data_start, value, value_size);
					//(object->buckets)[index] = kvn;
					atomic_add(&(object->num_buckets), -1);

					local_releaselock(&((object->locks)[index]));
					finish = 1;
					DoWork = false;
				}

				/*compare them*/
				else
				{
					//printf("no **********************\n");
					unsigned short size = lget_key_size(object, index);
					__local void *key_data = lget_key_address(object, index);

					/*if equal, conduct reduce*/
					if(equal_private_and_local(key, key_size, key_data, size))
					{
						unsigned short value_ro_size = lget_value_size(object, index);
						__local void *value_ro_data = lget_value_address(object, index);
						reduce_private_to_local(value_ro_data, value_ro_size, value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						local_releaselock(&((object->locks)[index]));
						
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = false;
						//finish = true;
						local_releaselock(&((object->locks)[index]));
						index = (index + stride)%NUM_BUCKETS_S;
						//printf("new index...\n");
					}
				}
			}
		}
	}
	return ret;
}


bool linsert_from_local(__local Reduction_Object_S *object, __local void *key, unsigned short key_size, __local void *value, unsigned short value_size)
{
	unsigned int h = hash_local(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;

	//printf("key is %d and index is %d\n", *(int *)key, index);

	unsigned int finish = 0;
	//unsigned int kvn = 0;

	int DoWork = 1;
	bool ret = true;
	int stride = 1;

	while(!finish)
	{
		if(object->num_buckets==0)
			return false;

		DoWork = 1;

		while(DoWork)
		{
			if(local_getlock(&((object->locks)[index])))
			{
				if(((object->buckets)[index]).x==0 && ((object->buckets)[index]).y==0)
				{
					int k = lmalloc(object, 2 + key_size);
					if(k == -1)
						ret = false;

					int v = lmalloc(object, value_size);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						local_releaselock(&(object->locks)[index]);
						finish  = 1;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

					__local char *key_size_address = (__local char *)lget_address(object, k);
					__local char *value_size_address = key_size_address + 1;
					*key_size_address = key_size;
					*value_size_address = value_size;

					//store the key data and value data into the ro
					__local void *key_data_start = key_size_address + 2;
					__local void *value_data_start = lget_address(object, v);

					copyVal_local_to_local(key_data_start, key, key_size);
					copyVal_local_to_local(value_data_start, value, value_size);
					//(object->buckets)[index] = kvn;
					atomic_add(&(object->num_buckets), -1);

					local_releaselock(&((object->locks)[index]));
					finish = 1;
					DoWork = false;
				}

				/*compare them*/
				else
				{
					//printf("no **********************\n");
					unsigned short size = lget_key_size(object, index);
					__local void *key_data = lget_key_address(object, index);

					/*if equal, conduct reduce*/
					if(equal_local_and_local(key, key_size, key_data, size))
					{
						unsigned short value_ro_size = lget_value_size(object, index);
						__local void *value_ro_data = lget_value_address(object, index);
						reduce_private_to_local(value_ro_data, value_ro_size, value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						local_releaselock(&((object->locks)[index]));
						
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = false;
						//finish = true;
						local_releaselock(&((object->locks)[index]));
						index = (index + stride)%NUM_BUCKETS_S;
						//printf("new index...\n");
					}
				}
			}
		}
	}
	return ret;
}


bool linsert_from_gp(__local Reduction_Object_S *object, __global void *key, unsigned short key_size, void *value, unsigned short value_size)
{
	unsigned int h = hash_global(key, key_size);
	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;
	//unsigned int kvn = 0;

	bool DoWork = true;
	bool ret = true;
	int stride = 1;

	while(!finish)
	{
		if(object->num_buckets==0)
			return false;

		DoWork = 1;

		while(DoWork)
		{
			if(local_getlock(&((object->locks)[index])))
			{
				if(((object->buckets)[index]).x==0 && ((object->buckets)[index]).y==0)
				{
					int k = lmalloc(object, 2 + key_size);
					if(k == -1)
						ret = false;

					int v = lmalloc(object, value_size);
					if(v == -1)
						ret = false;

					if(ret == false)
					{
						local_releaselock(&(object->locks)[index]);
						finish  = 1;
						break;
					}

					((object->buckets)[index]).x = k;
					((object->buckets)[index]).y = v;

					__local char *key_size_address = (__local char *)lget_address(object, k);
					__local char *value_size_address = key_size_address + 1;
					*key_size_address = key_size;
					*value_size_address = value_size;

					//store the key data and value data into the ro
					__local void *key_data_start = key_size_address + 2;
					__local void *value_data_start = lget_address(object, v);

					copyVal_global_to_local(key_data_start, key, key_size);
					copyVal_private_to_local(value_data_start, value, value_size);
					//(object->buckets)[index] = kvn;
					atomic_add(&(object->num_buckets), -1);

					local_releaselock(&((object->locks)[index]));
					finish = 1;
					DoWork = false;
					//printf("inserting...\n");
				}

				/*compare them*/
				else
				{
					//printf("no **********************\n");
					unsigned short size = lget_key_size(object, index);
					__local void *key_data = lget_key_address(object, index);

					/*if equal, conduct reduce*/
					if(equal_global_and_local(key, key_size, key_data, size))
					{
						unsigned short value_ro_size = lget_value_size(object, index);
						__local void *value_ro_data = lget_value_address(object, index);
						//printf("value size is: %d\n", value_size);
						reduce_private_to_local(value_ro_data, value_ro_size, value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						local_releaselock(&((object->locks)[index]));
						//printf("reducing...\n");
					}

					/*else, compute a new index and continue*/
					else
					{
						DoWork = false;
						//finish = true;
						local_releaselock(&((object->locks)[index]));
						index = (index + stride)%NUM_BUCKETS_S;
						//printf("computing new index...\n");
					}
				}
			}
		}
	}
	return ret;
}


int get_compare_value(__local Reduction_Object_S *object, unsigned int bucket_index1, unsigned int bucket_index2)
{
	Ints bucket1 = (object->buckets)[bucket_index1];
	Ints bucket2 = (object->buckets)[bucket_index2];

	int compare_value = 0;

	if(bucket1.y==0&&bucket2.y==0)
		compare_value = 0;

	else if(bucket2.y==0)
		compare_value = -1;

	else if(bucket1.y==0)
		compare_value = 1;

	else
	{
		unsigned int key_size1 = lget_key_size(object, bucket_index1);
		//unsigned int value_size1 = lget_value_size(object, bucket_index1);
		unsigned int key_size2 = lget_key_size(object, bucket_index2);
		//unsigned int value_size2 = lget_value_size(object, bucket_index2);
		__local void *key_addr1 = lget_key_address(object, bucket_index1);
		__local void *key_addr2 = lget_key_address(object, bucket_index2);

		unsigned int value_size1 = lget_value_size(object, bucket_index1);
		//unsigned int value_size1 = lget_value_size(object, bucket_index1);
		unsigned int value_size2 = lget_value_size(object, bucket_index2);
		//unsigned int value_size2 = lget_value_size(object, bucket_index2);
		__local void *value_addr1 = lget_value_address(object, bucket_index1);
		__local void *value_addr2 = lget_value_address(object, bucket_index2);

		compare_value = compare_local(key_addr1, key_size1, key_addr2, key_size2, value_addr1, value_size1, value_addr2, value_size2);
	}

	return compare_value;
}

void swap_bucket(__local Reduction_Object_S *object, unsigned int a, unsigned int b)
{
	short xtmp = (object->buckets)[a].x;
	short ytmp = (object->buckets)[a].y;

	(object->buckets)[a].x = (object->buckets)[b].x;
	(object->buckets)[a].y = (object->buckets)[b].y;

	(object->buckets)[b].x = xtmp;
	(object->buckets)[b].y = ytmp;
}

/*bitonic sort in the local reduction object*/
void lbitonic_sort(__local Reduction_Object_S *object)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = get_local_size(0)/NUM_GROUPS;
	const unsigned int id = local_id%group_size;

	for(int k = 2; k <= NUM_BUCKETS_S; k = k << 1)
		for(int j = k/2; j > 0; j = j >>1)
		{
			unsigned int ixj = id ^ j;
			if(id < NUM_BUCKETS_S && ixj < NUM_BUCKETS_S)
			if(ixj > id)
			{
				if((id & k)==0)
				{
					if(get_compare_value(object, id, ixj) > 0)
						swap_bucket(object, id, ixj);
				}
				else
				{
					if(get_compare_value(object, id, ixj) < 0)
						swap_bucket(object, id, ixj);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
}

void remove(__local Reduction_Object_S *object, unsigned int bucket_index)
{
	(object->buckets)[bucket_index].x = 0;
	(object->buckets)[bucket_index].y = 0;
}

#endif