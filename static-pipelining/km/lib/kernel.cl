#include "lib\kvbuffer.h"
#include "lib\kvbuffer_f.h"
#include "map.h"
#include "lib\configuration.h"
#include "lib\worker_info.h"
#include "lib\rog.h"
#include "lib\roc.h"
#include "lib\roc_f.h"
#include "lib\buffer_info.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable

void merge(__global Reduction_Object_G *object_g, __local Reduction_Object_S *object_s, 
__local unsigned int *global_object_offset)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = GPU_LOCAL_THREADS/NUM_GROUPS;
	
	for(int index = local_id%group_size; index < NUM_BUCKETS_S; index += group_size)
	{
		if(!((object_s->buckets)[index].x==0&&(object_s->buckets)[index].y==0))
		{
			
			int key_size = lget_key_size(object_s,index);
			int value_size = lget_value_size(object_s, index);
			__local void *key = lget_key_address(object_s, index);
			__local void *value = lget_value_address(object_s, index);
			ginsert_from_local(object_g, key, key_size, value, value_size, global_object_offset);
		}
	}
}

void cmerge(__global Reduction_Object_C *object_c, __local Reduction_Object_S *object_s)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = CPU_LOCAL_THREADS/NUM_GROUPS;
	
	for(int index = local_id%group_size; index < NUM_BUCKETS_S; index += group_size)
	{
		//printf("x: %d y: %d\n", (object_s->buckets)[index].x, (object_s->buckets)[index].y);
		if(!((object_s->buckets)[index].x==0&&(object_s->buckets)[index].y==0))
		{
			
			int key_size = lget_key_size(object_s,index);
			int value_size = lget_value_size(object_s, index);
			__local void *key = lget_key_address(object_s, index);
			__local void *value = lget_value_address(object_s, index);

			//printf("key is: %d\n", *(__local int *)key);

			cinsert_from_local(object_c, key, key_size, value, value_size);
		}
	}
}

__kernel void mergecg(__global Reduction_Object_C *object_c, __global Reduction_Object_G *object_g)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int global_id = get_global_id(0);
	
	for(int index = global_id; index < NUM_BUCKETS_G; index += CPU_GLOBAL_THREADS)
	{
		if(!((object_g->buckets)[index].x==0&&(object_g->buckets)[index].y==0))
		{
			int key_size = gget_key_size(object_g,index);
			int value_size = gget_value_size(object_g, index);
			__global void *key = gget_key_address(object_g, index);
			__global void *value = gget_value_address(object_g, index);
			cinsert_from_global(object_c, key, key_size, value, value_size);
		}
	}
}

__kernel void map_worker(
			__global const void *global_offset, 
			unsigned int offset_number, 
			unsigned int unit_size,
			__global char *global_data,

			__global Kvbuffer *buffers,
			__global Reduction_Object_G *object_g,
			__global Reduction_Object_C *object_c,
			__global Bufferinfo *bufferinfos,
			//__global int *testdata,
			int use_gc
	)
{
		int global_id = get_global_id(0);
		int local_id = get_local_id(0);
		int global_size = get_global_size(0);
		int local_size = get_local_size(0); 
		int block_id = global_id/local_size;

		__local unsigned int index_offset;
		__local unsigned int pool_offset;
		__local unsigned int current_buffer;
		__local int did;
		__local int full;

		if(local_id==0)
		{
			index_offset = 0;
			pool_offset = 0;
			current_buffer = 0;
			did = 0;
			full = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		bool first = 1;

		int i = global_id;

		while(did!=local_size)
		{
			if(local_id==0)
			while(bufferinfos[block_id*NUM_BUFFERS_IN_BLOCK + current_buffer].full==1)
			{
				current_buffer = (current_buffer + 1)%NUM_BUFFERS_IN_BLOCK;
			}

			barrier(CLK_LOCAL_MEM_FENCE);
					
			for(; i<offset_number; i += global_size)
			{
				if(full)
					break;
						
				bool success = map_buffer(&buffers[block_id*NUM_BUFFERS_IN_BLOCK + current_buffer], 
					global_data, 
					((__global char *)global_offset+unit_size*i), 
					&index_offset, 
					&pool_offset);
						
					if(!success)
					{
						full = 1;
						break;
					}
			}

			if(first&&(i>=offset_number))
			{
				atomic_add(&did, 1);
				first = 0;
			}
					
			barrier(CLK_LOCAL_MEM_FENCE);

			if(full)
			{				
				if(local_id==0)
				{
					//set current buffer, indicate it to be full
					bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].num = index_offset;

					//if(index_offset!=0)
					//testdata[block_index]+= 1;

					bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].full = 1;

					//change to next available buffer
					current_buffer = (current_buffer + 1) % NUM_BUFFERS_IN_BLOCK;    //round robin

					//re-initialize the offset information
					index_offset = 0;

					pool_offset = 0;

					full = 0;
				} 
			}
		}

		if(local_id==0)
		{
			if(index_offset!=0)
			{
				bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].num = index_offset;

				bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].full = 1;
			}
		
			for(int i = 0; i < NUM_BUFFERS_IN_BLOCK; i++)
			{
				bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK+i].finish = 1;
			}
		}
}

__kernel void reduce_worker(
			
			__global Kvbuffer *buffers,
			__global Reduction_Object_G *object_g,
			__global Reduction_Object_C *object_c,
			__global Bufferinfo *bufferinfos,
			//__global int *testdata,
			int use_gc
	)
{
	__local Reduction_Object_S objects[NUM_GROUPS];
	const uint local_id =  get_local_id(0);	
	const uint global_id = get_global_id(0);
	int global_size = get_global_size(0);
	int local_size = get_local_size(0); 
	int total_blocks = global_size/local_size;
	int total_buffers;

	const uint group_size = local_size/NUM_GROUPS;
	const uint gid = local_id/group_size;
	int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
	int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;
	__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];
	__local unsigned int buffer_index;
	__local unsigned int finish;
	__local unsigned int num;
	
	unsigned int finished_buffer;
	unsigned int inner_buffer = 0;
	

	if(use_gc==TYPE_GPU)
	{
		linit(&objects[gid], TYPE_GPU);
		total_buffers = NUM_BUFFERS_IN_BLOCK*cpu_blocks;
	}
	else
	{
		linit(&objects[gid], TYPE_CPU);
		total_buffers = NUM_BUFFERS_IN_BLOCK*gpu_blocks;
	}

	int block_index = get_group_id(0);
	int my_share = total_buffers/total_blocks;
	int my_start = total_buffers*block_index/total_blocks;
	unsigned int my_buffer_index = inner_buffer + my_start;

	if(use_gc==TYPE_GPU)
	{
		ginit(global_object_offset);
	}
	
	else
	cinit(object_c);

	if(local_id==0)
	{
		//buffer_index = my_start + inner_buffer;
		finished_buffer = 0;
		finish = 0;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	while(1)
	{	
		if(local_id==0)
		{
			while(1)
			{
				if(bufferinfos[my_buffer_index].full!=1)
				{
					my_buffer_index = my_start+inner_buffer;
					if(bufferinfos[my_buffer_index].finish==1)
					{
						finished_buffer++;
						bufferinfos[my_buffer_index].finish==-1;
					}

					inner_buffer = (inner_buffer+1)%my_share;

					if(finished_buffer==my_share)
					{
						finish = 1;
						break;
					}
				}

				else
				{
					break;
				}
			}
			num = bufferinfos[my_buffer_index].num;
			buffer_index = my_buffer_index;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(finish)
			break;

		int i = local_id;
		__global Kvbuffer *buffer = &buffers[buffer_index];
		
		for(; i < num; i += local_size)
		{
			short key_size = kget_key_size(buffer, i);
			short value_size = kget_value_size(buffer, i);
			__global void *key = kget_key_address(buffer, i);
			__global void *value = kget_value_address(buffer, i);
			//cinsert_from_global(object_c, key, key_size, value, value_size);
			linsert_from_global(&objects[gid], key, key_size, value, value_size);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id==0)
		{
				bufferinfos[buffer_index].full = 0;
			//bufferinfos[buffer_index].num = 0;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if(use_gc==TYPE_GPU)
		merge(object_g, &objects[gid], global_object_offset);
	else
		cmerge(object_c, &objects[gid]);
}

__kernel void get_size_gpu(__global Reduction_Object_G *object_g, __global unsigned int *num_key, 
__global unsigned int *size_of_key, __global unsigned int *size_of_value)
{
    /*scan the arrays and get prefix sum info*/
    int global_id = get_global_id(0);
    if(global_id == 0)
    {
		for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->pairs_per_bucket)[i] += (object_g->pairs_per_bucket)[i-1];
        }
    }

    if(global_id == 1)
    {
        for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->key_size_per_bucket)[i] += (object_g->key_size_per_bucket)[i-1];
        }
    }

    if(global_id == 2)
    {
        for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->value_size_per_bucket)[i] += (object_g->value_size_per_bucket)[i-1];
        }
    }

    *num_key = (object_g->pairs_per_bucket)[NUM_BUCKETS_G - 1];
    *size_of_key = (object_g->key_size_per_bucket)[NUM_BUCKETS_G - 1];
    *size_of_value = (object_g->value_size_per_bucket)[NUM_BUCKETS_G - 1];
}

__kernel void get_size_cpu(__global Reduction_Object_C *object_g, __global unsigned int *num_key, 
__global unsigned int *size_of_key, __global unsigned int *size_of_value)
{
    /*scan the arrays and get prefix sum info*/
    int global_id = get_global_id(0);
    if(global_id == 0)
    {
		for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->pairs_per_bucket)[i] += (object_g->pairs_per_bucket)[i-1];
        }
    }

    if(global_id == 1)
    {
        for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->key_size_per_bucket)[i] += (object_g->key_size_per_bucket)[i-1];
        }
    }

    if(global_id == 2)
    {
        for(int i = 1; i < NUM_BUCKETS_G; i++)
        {
            (object_g->value_size_per_bucket)[i] += (object_g->value_size_per_bucket)[i-1];
        }
    }

    *num_key = (object_g->pairs_per_bucket)[NUM_BUCKETS_G - 1];
    *size_of_key = (object_g->key_size_per_bucket)[NUM_BUCKETS_G - 1];
    *size_of_value = (object_g->value_size_per_bucket)[NUM_BUCKETS_G - 1];
}

__kernel void copy_to_array_gpu(__global Reduction_Object_G *object_g, __global void *key_array, 
    __global void *value_array, __global unsigned int *key_index, __global unsigned int *value_index)
{
	int i = get_global_id(0);
    for(; i < NUM_BUCKETS_G; i += CPU_GLOBAL_THREADS)
    {
        if(!((object_g->buckets)[i].x==0&&(object_g->buckets)[i].y==0))
        {
            int key_size;
            int value_size;
            
			if(i!=0)
			{
				key_size = gget_key_size(object_g, i) - gget_key_size(object_g, i-1);
				value_size = gget_value_size(object_g, i) - gget_value_size(object_g, i-1);
			}

			else
			{
				key_size = gget_key_size(object_g, i);
				value_size = gget_value_size(object_g, i);
			}
			
			__global void *key = gget_key_address(object_g, i);
            __global void *value = gget_value_address(object_g, i);

			//int key_test;
			//copyVal_global_to_private(&key_test, key, key_size);

			//printf("the key_size is: %d and key is: %d\n", key_size, key_test);

            unsigned int key_array_start = (object_g->key_size_per_bucket)[i] - key_size;
            unsigned int value_array_start = (object_g->value_size_per_bucket)[i] - value_size;
            unsigned int offset_pos = (object_g->pairs_per_bucket)[i] - 1;
            copyVal_global_to_global((__global char *)key_array + key_array_start, key, key_size);
            copyVal_global_to_global((__global char *)value_array + value_array_start, value, value_size);
            key_index[offset_pos] = key_array_start;
            value_index[offset_pos] = value_array_start;
        }
    }
}

__kernel void copy_to_array_cpu(__global Reduction_Object_C *object_g, __global void *key_array, 
    __global void *value_array, __global unsigned int *key_index, __global unsigned int *value_index)
{
	int i = get_global_id(0);
    for(; i < NUM_BUCKETS_C; i += CPU_GLOBAL_THREADS)
    {
        if(!((object_g->buckets)[i].x==0&&(object_g->buckets)[i].y==0))
        {
            int key_size;
            int value_size;
            
			if(i!=0)
			{
				key_size = cget_key_size(object_g, i) - cget_key_size(object_g, i-1);
				value_size = cget_value_size(object_g, i) - cget_value_size(object_g, i-1);
			}

			else
			{
				key_size = cget_key_size(object_g, i);
				value_size = cget_value_size(object_g, i);
			}
			
			__global void *key = cget_key_address(object_g, i);
            __global void *value = cget_value_address(object_g, i);

			//int key_test;
			//copyVal_global_to_private(&key_test, key, key_size);

			//printf("the key_size is: %d and key is: %d\n", key_size, key_test);

            unsigned int key_array_start = (object_g->key_size_per_bucket)[i] - key_size;
            unsigned int value_array_start = (object_g->value_size_per_bucket)[i] - value_size;
            unsigned int offset_pos = (object_g->pairs_per_bucket)[i] - 1;
            copyVal_global_to_global((__global char *)key_array + key_array_start, key, key_size);
            copyVal_global_to_global((__global char *)value_array + value_array_start, value, value_size);
            key_index[offset_pos] = key_array_start;
            value_index[offset_pos] = value_array_start;
        }
    }
}