#include "lib\rog.h"
#include "lib\rol.h"
#include "lib\rol_f.h"
#include "lib\rog"
#include "lib\configuration.h"
#include "map.h"
#include "lib\util_device.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

void merge(__global Reduction_Object_G *object_g, __local Reduction_Object_S *object_s, 
__local unsigned int *global_object_offset)
{
	const unsigned int local_id = get_local_id(0);
	const unsigned int group_size = LOCAL_THREADS/NUM_GROUPS;
	//printf("I am thread %d\n", get_global_id(0));
	for(int index = local_id%group_size; index < NUM_BUCKETS_S; index += group_size)
	{
		if(!((object_s->buckets)[index].x==0&&(object_s->buckets)[index].y==0))
		{
			int key_size = lget_key_size(object_s,index);
			//printf("key size is: %d\n", key_size);
			int value_size = lget_value_size(object_s, index);
			__local void *key = lget_key_address(object_s, index);
			__local void *value = lget_value_address(object_s, index);

			//int key_tmp;
			//copyVal_local_to_private(&key_tmp, key, key_size);
			//printf("index is: %d and key is: %d\n", index, key_tmp);

			ginsert_from_local(object_g, key, key_size, value, value_size, global_object_offset);
		}
	}
}

void lmerge(__local Reduction_Object_G *object2, __local Reduction_Object_S *object1)
{
	const unsigned int local_id = get_local_id(0);
	for(int index = local_id; index < SORT_REMAIN; index += LOCAL_THREADS)
	{
		if((object1->buckets)[index].y!=0)
		{
			int key_size = lget_key_size(object1,index);
			//printf("key size is: %d\n", key_size);
			int value_size = lget_value_size(object1, index);
			__local void *key = lget_key_address(object1, index);
			__local void *value = lget_value_address(object1, index);

			//int key_tmp;
			//copyVal_local_to_private(&key_tmp, key, key_size);
			//printf("index is: %d and key is: %d\n", index, key_tmp);

			linsert_from_local(object_g, key, key_size, value, value_size, global_object_offset);
		}
	}
}

__kernel void start(global Reduction_Object_G *object_g, 
					global const void *global_data, 
					global const void *global_offset, 
					unsigned int offset_number, 
					/*global const void *local_data, 
					local const void *shared_local_data, 
					unsigned int local_data_size, */
					unsigned int unit_size, 
					unsigned int uselocal)
{	
	__local unsigned int global_object_offset[LOCAL_THREADS/WAVEFRONT_SIZE];
	ginit(global_object_offset);


    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
	//object_g->memory_offset = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
	
    if(uselocal)
	{
		__local Reduction_Object_S objects[2];
		__local use_index;
		__local merge_index;
		__local int do_merge;
		__local int finished;
		const unsigned int group_size = LOCAL_THREADS/NUM_GROUPS;
		const unsigned int gid = get_local_id(0)/group_size;
		
		linit(&objects[0]);
		linit(&objects[1]);
		if(local_id == 0)
		{
			use_index = 0;
			merge_index = 1;
			do_merge = 0;
			finished = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//printf("num of offsets: %d\n", offset_number);
		//printf("0 is: %s\n", ((__global char *)global_data + 2));
		
		bool flag = true;
		int i = global_id;
		while(finished!=LOCAL_THREADS)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			
			for(; i < offset_number; i+= GLOBAL_THREADS)
			{
				//printf("i is: %d\n", i);
				
				if(do_merge)
				break;

				//int success = 1;
				//printf("offset: %d word: %s\n", ((__global int *)global_offset)[i], (__global char *)(global_data + ((__global int *)global_offset)[i]));
				int success = map_to_local(&objects[gid], global_data, ((global char *)global_offset+unit_size*i));
				if(!success)
				{
					atomic_xchg(&do_merge, 1);
					break;
				}
			}

			if(flag&&i>=offset_number)
			{
				flag = false;
				atomic_add(&finished, 1);
			}

			//printf("I am here..\n");

			//printf("num of buckets: %d\n", objects[gid].num_buckets);

			barrier(CLK_LOCAL_MEM_FENCE);
			//merge(object_g, &objects[gid], global_object_offset);
			lbitonic_sort(&objects[use_index]);

			barrier(CLK_LOCAL_MEM_FENCE);

			lmerge(&objects[merge_index], &objects[use_index]);

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
	
    else
	{
		for(int i = get_global_id(0); i < offset_number; i += GLOBAL_THREADS)
		{
			//printf("I am thread: %d and I am doing %d\n", get_global_id(0), i);
			map_to_global(object_g, global_data, ((__global char *)global_offset+unit_size*i), global_object_offset);
		}
		//barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


get_size(__global Reduction_Object_G *object_g, __global unsigned int *num_key, 
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


__kernel void copy_to_array(__global Reduction_Object_G *object_g, __global void *key_array, 
    __global void *value_array, __global unsigned int *key_index, __global unsigned int *value_index)
{
	int i = get_global_id(0);
    for(; i < NUM_BUCKETS_G; i += GLOBAL_THREADS)
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

__kernel void sort(__global Reduction_Object_G *object_g, unsigned int k, unsigned int j)
{
	//gbitonic_merge(object_g, k, j);
}