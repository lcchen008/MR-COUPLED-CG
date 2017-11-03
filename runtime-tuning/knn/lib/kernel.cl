#include "lib\kvbuffer.h"
#include "map.h"
#include "lib\configuration.h"
#include "lib\worker_info.h"
#include "lib\rog.h"
#include "lib\roc.h"

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
		//printf("x: %d y: %d\n", (object_s->buckets)[index].x, (object_s->buckets)[index].y);
		if(!((object_s->buckets)[index].x==0&&(object_s->buckets)[index].y==0))
		{
			
			int key_size = lget_key_size(object_s,index);
			int value_size = lget_value_size(object_s, index);
			__local void *key = lget_key_address(object_s, index);
			__local void *value = lget_value_address(object_s, index);

			//printf("key is: %c\n", *(__local char *)key);

			ginsert_from_local(object_g, key, key_size, value, value_size, global_object_offset);
		}
	}
}


void cmerge(__global Reduction_Object_C *object_c, __local Reduction_Object_S *object_s)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = GPU_LOCAL_THREADS/NUM_GROUPS;
	
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


void lmerge(__local Reduction_Object_S *object2, __local Reduction_Object_S *object1, __local float *threshold)
{
	const unsigned int local_id = get_local_id(0);
	for(int index = local_id; index < SORT_REMAIN; index += GPU_LOCAL_THREADS)
	{
		if((object1->buckets)[index].y!=0)
		{
			int key_size = lget_key_size(object1,index);
			//printf("key size is: %d\n", key_size);
			int value_size = lget_value_size(object1, index);
			__local void *key = lget_key_address(object1, index);
			__local void *value = lget_value_address(object1, index);

			linsert_from_local(object2, key, key_size, value, value_size);
			if(index==SORT_REMAIN-1)
				*threshold = *(__local float *)value;
		}
	}
}

__kernel void mergecg(__global Reduction_Object_C *object_c, __global Reduction_Object_G *object_g)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int global_id = get_global_id(0);
	
	for(int index = global_id; index < NUM_BUCKETS_G; index += CPU_GLOBAL_THREADS)
	{
		//printf("x: %d y: %d\n", (object_g->buckets)[index].x, (object_g->buckets)[index].y);
		if((object_g->buckets)[index].y!=0)
		{
			int key_size = gget_key_size(object_g,index);
			int value_size = gget_value_size(object_g, index);
			__global void *key = gget_key_address(object_g, index);
			__global void *value = gget_value_address(object_g, index);

			//printf("key is: %d\n", *(__global int *)key);

			cinsert_from_global(object_c, key, key_size, value, value_size);
		}
	}
}

__kernel void test(
			__global const void *global_offset, 
		    unsigned int offset_number, 
		    unsigned int unit_size,
			__global char *global_data,
			int device_type,
			__global Worker *workers,
			__global Reduction_Object_C *object_c,
			__global Reduction_Object_G *object_g
			)
{
		int device = device_type;
		__local Reduction_Object_S objects[2];
		__local float threshold;
		__local int task_index;
		__local int has_task;
		__local int finish;

		__local int use_index;
		__local int merge_index;
		__local int do_merge;
		__local int finished;
		__local int task_block_size;

		__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];

		const uint local_id =  get_local_id(0);	
		const uint global_id = get_global_id(0);

		linit(&objects[0]);
		linit(&objects[1]);

		int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
		
		const uint block_id = get_group_id(0);
		int block_index = block_id;

		if(device==TYPE_GPU)
		{
			block_index = block_id + cpu_blocks;
			ginit(global_object_offset);
		}

		if(device==TYPE_CPU)
		{
			cinit(object_c);
		}

		if(local_id==0)
		{
			has_task = 0;
			finish = 0;
			use_index = 0;
			merge_index = 1;
			do_merge = 0;
			finished = 0;
			threshold = 65535*1000;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		while(!finish)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if(local_id==0)
			{
				has_task = workers[block_index].has_task;

				if(has_task==1)
				{
					task_index = workers[block_index].task_num;
					task_block_size = workers[block_index].task_block_size;
					workers[block_index].has_task = 0;
					//printf("task index: %d\n", task_index);	
					finished = 0;
				}

				if(has_task==-1)
					finish = true;
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			if(has_task==1)
			{
				bool flag = true;
				int i = task_index + local_id;
				int block_size = task_block_size;
				
				while(finished!=GPU_LOCAL_THREADS)
				{
					barrier(CLK_LOCAL_MEM_FENCE);
			
					for(; (i < (task_index + block_size)) && i < offset_number; i+= CPU_LOCAL_THREADS)
					{
						if(do_merge)
						break;

						int success = map_local(&objects[use_index], global_data, ((global char *)global_offset+unit_size*i), threshold);
				
						if(!success)
						{
							do_merge = 1;
							break;
						}
					}

					if(flag&&(i>= (task_index + block_size) ||i>=offset_number))
					{
						flag = false;
						atomic_add(&finished, 1);
					}

					barrier(CLK_LOCAL_MEM_FENCE);
	
					lbitonic_sort(&objects[use_index]);

					barrier(CLK_LOCAL_MEM_FENCE);

					lmerge(&objects[merge_index], &objects[use_index], &threshold);

					barrier(CLK_LOCAL_MEM_FENCE);

					linit(&objects[use_index]);

					barrier(CLK_LOCAL_MEM_FENCE);

					if(local_id==0)
					{
						unsigned int tmp = use_index;
						use_index = merge_index;
						merge_index = tmp;
						do_merge = 0;
					}
				}
			}
		}
		//barrier(CLK_LOCAL_MEM_FENCE);

		if(device==TYPE_GPU)
			merge(object_g, &objects[use_index], global_object_offset);
		else
			cmerge(object_c, &objects[use_index]);
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

__kernel void sort(__global Reduction_Object_C *object_c, unsigned int k, unsigned int j)
{
	cbitonic_merge(object_c, k, j);
	/*if(get_global_id(0)==0)
		printf("global threads are: %d\n", get_global_size(0));*/
}