#ifndef KVBUFFERF
#define KVBUFFERF

#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#include "kvbuffer.h"
#include "util_device.h"

//int kalloc_index(__local Kvbuffer *buffer)
//{
//	if(buffer->index_offset >= KVINDEX_NUM)
//		return -1;
//	unsigned int offset = atomic_add(&(buffer->index_offset), 1);
//	if(offset > KVINDEX_NUM - 1)
//		return -1;
//	else 
//		return offset;
//}

//int kalloc_pool(__local Kvbuffer *buffer, int size)
//{
//	if(buffer->memory_offset >= KV_POOL_SIZE)
//		return -1;
//	size = align(size)/ALIGN_SIZE;
//	unsigned int offset = atomic_add(&(buffer->memory_offset), size);
//	if(offset + size > KV_POOL_SIZE)
//		return -1;
//	else
//		return offset;
//}

int kalloc_index_g(__global Kvbuffer *buffer, __local int *index_offset)
{
	if(*index_offset >= KVINDEX_NUM)
		return -1;
	unsigned int offset = atomic_add(index_offset, 1);
	if(offset > KVINDEX_NUM - 1)
	{
		atomic_add(index_offset, -1);
		return -1;
	}
	else 
		return offset;
}

int kalloc_pool_g(__global Kvbuffer *buffer, int size, __local int *pool_offset)
{
	if(*pool_offset >= KV_POOL_SIZE)
		return -1;
	size = align(size)/ALIGN_SIZE;
	unsigned int offset = atomic_add(pool_offset, size);
	if(offset + size > KV_POOL_SIZE)
		return -1;
	else
		return offset;
}

__local void * kget_address(__local Kvbuffer *buffer, unsigned memory_index)
{
	return buffer->memory_pool + memory_index;
}

__global void * kget_address_g(__global Kvbuffer *buffer, unsigned memory_index)
{
	return buffer->memory_pool + memory_index;
}

__global void * kget_key_address(__global Kvbuffer *buffer, unsigned bucket_index)
{
	unsigned int key_index = ((buffer->index)[bucket_index]).k;
	return kget_address_g(buffer, key_index);
}

unsigned short kget_key_size(__global Kvbuffer *buffer, unsigned bucket_index)
{
	return (buffer->index)[bucket_index].k_size;    
}

__global void * kget_value_address(__global Kvbuffer *buffer, unsigned bucket_index)
{
	unsigned int value_index =  ((buffer->index)[bucket_index]).v;;
	return kget_address_g(buffer, value_index);
}

unsigned short kget_value_size(__global Kvbuffer *buffer, unsigned bucket_index)
{
	return (buffer->index)[bucket_index].v_size;  
}

//bool kinsert(__local Kvbuffer *buffer, void *key, unsigned short key_size, void *value, unsigned short value_size)
//{
//	int index;
//	int k;
//	int v;
//
//	k = kalloc_pool(buffer, key_size);
//	if(k==-1)
//		return false;
//	v = kalloc_pool(buffer, value_size);
//	if(v==-1)
//		return false;
//	index = kalloc_index(buffer);
//	if(index==-1)
//		return false;
//
//	/*copy key data and value data*/
//	__local char *key_data_start = (__local char *)kget_address(buffer, k);
//	__local char *value_data_start = (__local char *)kget_address(buffer, v);
//
//	copyVal_private_to_local(key_data_start, key, key_size);
//	copyVal_private_to_local(value_data_start, value, value_size);
//
//	(buffer->index)[index].k = k;
//	(buffer->index)[index].v = v;
//	(buffer->index)[index].k_size = key_size;
//	(buffer->index)[index].v_size = value_size;
//
//	return true;
//}

bool kinsert_g(__global Kvbuffer *buffer, 
	void *key, 
	unsigned short key_size, 
	void *value, 
	unsigned short value_size, 
	__local unsigned int *index_offset, 
	__local unsigned int *pool_offset)
{
	int index;
	int k;
	int v;

	k = kalloc_pool_g(buffer, key_size, pool_offset);
	if(k==-1)
		return false;
	v = kalloc_pool_g(buffer, value_size, pool_offset);
	if(v==-1)
		return false;
	index = kalloc_index_g(buffer, index_offset);
	if(index==-1)
		return false;

	/*copy key data and value data*/
	__global char *key_data_start = (__global char *)kget_address_g(buffer, k);
	__global char *value_data_start = (__global char *)kget_address_g(buffer, v);

	copyVal_private_to_global(key_data_start, key, key_size);
	copyVal_private_to_global(value_data_start, value, value_size);

	(buffer->index)[index].k = k;
	(buffer->index)[index].v = v;
	(buffer->index)[index].k_size = key_size;
	(buffer->index)[index].v_size = value_size;	

	return true;
}

void dump(__global Kvbuffer *buffer)
{
	int i = 0;
	int count = 0;
	for(; i < KVINDEX_NUM; i++)
	{
		if(!((buffer->index)[i].k==0&&(buffer->index)[i].v==0))
		{
			count++;
		}
	}
	printf("****Total: %d\n", count);
}
#endif