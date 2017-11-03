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

__kernel void mergecg(__global Reduction_Object_C *object_c, __global Reduction_Object_G *object_g)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int global_id = get_global_id(0);

	/*CPU has 4 cores*/

	for(int index = global_id; index < NUM_BUCKETS_G; index += CPU_GLOBAL_THREADS)
	{
		//printf("global id: %d index is: %d\n",global_id, index);
		//printf("x: %d y: %d\n", (object_g->buckets)[index].x, (object_g->buckets)[index].y);
		if((object_g->buckets)[index].y!=0)
		{
			int key_size = gget_key_size(object_g,index);
			int value_size = gget_value_size(object_g, index);
			__global void *key = gget_key_address(object_g, index);
			__global void *value = gget_value_address(object_g, index);

			//int tmp;
			//copyVal_global_to_private(&tmp, key, key_size);

			//printf("index is: %d key is: %d\n", index, tmp);

			cinsert_from_global(object_c, key, key_size, value, value_size);
		}
	}
}

__kernel void mergegc(__global Reduction_Object_G *object_g, __global Reduction_Object_C *object_c)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int global_id = get_global_id(0);
	const uint block_id = get_group_id(0);
	__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];

	if(local_id%WAVEFRONT_SIZE==0)
	global_object_offset[local_id/WAVEFRONT_SIZE] = (object_g->offsets)[block_id*(local_id/WAVEFRONT_SIZE)];

	barrier(CLK_LOCAL_MEM_FENCE);

	/*CPU has 4 cores*/

	for(int index = global_id; index < NUM_BUCKETS_C; index += GPU_GLOBAL_THREADS)
	{
		//printf("global id: %d index is: %d\n",global_id, index);
		//printf("x: %d y: %d\n", (object_g->buckets)[index].x, (object_g->buckets)[index].y);
		if((object_c->buckets)[index].y!=0)
		{
			int key_size = cget_key_size(object_c,index);
			int value_size = cget_value_size(object_c, index);
			__global void *key = cget_key_address(object_c, index);
			__global void *value = cget_value_address(object_c, index);

			//int tmp;
			//copyVal_global_to_private(&tmp, key, key_size);

			//printf("index is: %d key is: %d\n", index, tmp);

			ginsert_from_global(object_g, key, key_size, value, value_size, global_object_offset);
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

	if(device==TYPE_CPU)
	{
		const uint local_id =  get_local_id(0);	
		const uint global_id = get_global_id(0);
	
		int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
		int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;

		//int num_blocks = cpu_blocks + gpu_blocks;

		int block_index = get_group_id(0);

		cinit(object_c);

		__local int count;

		__local int task_index;
		__local int has_task;
		__local int finish;

		if(local_id==0)
		{
			count = 0;
			has_task = 0;
			finish = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		while(!finish)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if(local_id==0)
			{
				has_task = workers[block_index].has_task;
				task_index = workers[block_index].task_num;
				if(has_task==1)
				{
					count++;
					workers[block_index].has_task = 0;
				}
			
				if(has_task==-1)
				{
					finish = 1;
				}
			}
			
			barrier(CLK_LOCAL_MEM_FENCE);

			if(has_task==1)
			{
				for(int i = task_index + local_id; i < task_index + TASK_BLOCK_SIZE && i < offset_number; i += CPU_LOCAL_THREADS)
				{
					bool success = map_cpu(object_c, global_data, ((__global char *)global_offset+unit_size*i));
				}
			}
		}
	}
	
	if(device==TYPE_GPU)
	{
		__local Reduction_Object_S objects[NUM_GROUPS];
		const uint local_id =  get_local_id(0);	
		const uint global_id = get_global_id(0);
		const unsigned int group_size = GPU_LOCAL_THREADS/NUM_GROUPS;
		const unsigned int gid = local_id/group_size;

		linit(&objects[gid]);
	
		int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
		int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;

		//int num_blocks = cpu_blocks + gpu_blocks;
		
		const uint block_id = get_group_id(0);
		int block_index = block_id + cpu_blocks;

		__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];

		ginit(global_object_offset);
		
		__local int task_index;
		__local int has_task;
		__local int finish;
		__local count;

		if(local_id==0)
		{
			has_task = 0;
			finish = false;
			count = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		while(!finish)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if(local_id==0)
			{
				has_task = workers[block_index].has_task;
				task_index = workers[block_index].task_num;

				if(has_task==1)
				{
					count++;
					workers[block_index].has_task = 0;
				}

				if(has_task==-1)
					finish = true;
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			if(has_task==1)
			{
				#ifdef USE_LOCAL
				for(int i = task_index + local_id; i < task_index + TASK_BLOCK_SIZE && i < offset_number; i += GPU_LOCAL_THREADS)
				{
					map_local(&objects[gid], global_data, ((__global char *)global_offset+unit_size*i));
				}
				#else

				for(int i = task_index + local_id; i < task_index + TASK_BLOCK_SIZE && i < offset_number; i += GPU_LOCAL_THREADS)
				{
					//printf("I am thread: %d and I am doing %d\n", get_global_id(0), i);
					map_to_global(object_g, global_data, ((__global char *)global_offset+unit_size*i), global_object_offset);
				}
				#endif
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id%WAVEFRONT_SIZE==0)
		(object_g->offsets)[block_id*(local_id/WAVEFRONT_SIZE)] = global_object_offset[local_id/WAVEFRONT_SIZE];

		#ifdef USE_LOCAL
		merge(object_g, &objects[gid], global_object_offset);
		#endif
	}
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
    for(; i < NUM_BUCKETS_G; i += GPU_GLOBAL_THREADS)
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