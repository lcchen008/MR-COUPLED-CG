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

/*works on GPU*/
__kernel void map_worker(
			__global const void *global_offset, 
			unsigned int offset_number, 
			unsigned int unit_size,
			__global char *global_data,
			__global Worker *workers,
			__global Kvbuffer *buffers,
			__global Reduction_Object_G *object_g,
			__global Reduction_Object_C *object_c,
			__global Bufferinfo *bufferinfos,
			//__global int *testdata,
			int use_gc
	)
{
	/*cpu does reduce, which means, gpu does map*/
	if(use_gc==TYPE_CPU)
	{
		//__local Reduction_Object_S objects[NUM_GROUPS];
		const uint local_id =  get_local_id(0);	
		const uint global_id = get_global_id(0);
		const unsigned int group_size = GPU_LOCAL_THREADS/NUM_GROUPS;
		const unsigned int gid = local_id/group_size;

		__local int index_offset;
		__local int pool_offset;
		__local int current_buffer;
		__local bool use_buffer;

		if(local_id==0)
		{
			index_offset = 0;
			pool_offset = 0;
			current_buffer = 0;
			use_buffer = 1;
		}

		//linit(&objects[gid]);
	
		int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
		int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;

		int num_blocks = cpu_blocks + gpu_blocks;

		const uint block_id = get_group_id(0);
		int block_index = block_id + cpu_blocks;         //the index used to access the worker info array

		__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];

		ginit(global_object_offset);

		__local int task_index;
		__local int has_task;
		__local int finish;
		__local int did;
		__local int full;
		
		if(local_id==0)
		{
			has_task = 0;
			finish = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		
		while(finish!=1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if(local_id==0)
			{
				has_task = workers[block_index].has_task;
				task_index = workers[block_index].task_num;
				did = 0;
				full = 0;

				if(has_task==1)
				{
					workers[block_index].has_task = 0;
				}

				if(has_task==-1)
				{
					if(index_offset!=0)
					{
						//buffers[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].index_offset = index_offset;
						bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].num = index_offset;
						bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].full = 1;
						//testdata[block_index]+= 1;
					}

					finish = 1 ;
					workers[block_index].finish_all = 1;
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			if(has_task==1)
			{
				int i = task_index + local_id;
				bool first = 1;
				while(did!=GPU_LOCAL_THREADS)
				{
					if(local_id==0)
					while(bufferinfos[block_id*NUM_BUFFERS_IN_BLOCK + current_buffer].full==1)
					{
						current_buffer = (current_buffer + 1)%NUM_BUFFERS_IN_BLOCK;
					}

					barrier(CLK_LOCAL_MEM_FENCE);
					
					for(; (i < (task_index + TASK_BLOCK_SIZE))&&i<offset_number; i += GPU_LOCAL_THREADS)
					{
						//map_local(&objects[gid], global_data, ((__global char *)global_offset+unit_size*i));
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

					if(first&&i>=(task_index + TASK_BLOCK_SIZE))
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
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}
		}
	}

	/*cpu does map*/
	else
	{
		const uint local_id =  get_local_id(0);	
		const uint global_id = get_global_id(0);

		__local int index_offset;
		__local int pool_offset;
		__local int current_buffer;
		//__local bool use_buffer;

		if(local_id==0)
		{
			index_offset = 0;
			pool_offset = 0;
			current_buffer = 0;
			//use_buffer = 1;
		}

		int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
		int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;

		int num_blocks = cpu_blocks + gpu_blocks;

		const uint block_id = get_group_id(0);
		int block_index = block_id;         //the index used to access the worker info array

		__local int task_index;
		__local int has_task;
		__local int finish;
		__local int did;
		__local int full;
		
		if(local_id==0)
		{
			has_task = 0;
			finish = 0;
		}

		cinit(object_c);

		barrier(CLK_LOCAL_MEM_FENCE);
		
		while(!finish)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if(local_id==0)
			{
				has_task = workers[block_index].has_task;
				task_index = workers[block_index].task_num;
				did = 0;
				full = 0;

				if(has_task==1)
				{
					//printf("%d receive a task %d\n", block_index, task_index);
					workers[block_index].has_task = 0;
				}

				if(has_task==-1)
				{
					if(index_offset!=0)
					{
						bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].num = index_offset;
						bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].full = 1;
						//testdata[block_index] += 1;
					}

					finish = 1 ;
					workers[block_index].finish_all = 1;
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			if(has_task==1)
			{
				int i = task_index + local_id;
				bool first = 1;
				while(did!=CPU_LOCAL_THREADS)
				{
					if(local_id==0)
					while(bufferinfos[block_id*NUM_BUFFERS_IN_BLOCK + current_buffer].full==1)
					{
						current_buffer = (current_buffer + 1)%NUM_BUFFERS_IN_BLOCK;
					}

					barrier(CLK_LOCAL_MEM_FENCE);
					
					for(; (i < (task_index + TASK_BLOCK_SIZE))&&i<offset_number; i += CPU_LOCAL_THREADS)
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

					if(first&&i>=(task_index + TASK_BLOCK_SIZE))
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

							bufferinfos[block_id * NUM_BUFFERS_IN_BLOCK + current_buffer].full = 1;

							//change to next available buffer
							current_buffer = (current_buffer + 1) % NUM_BUFFERS_IN_BLOCK;    //round robin

							//re-initialize the offset information
							index_offset = 0;

							pool_offset = 0;

							full = 0;
						} 
					}
					//barrier(CLK_LOCAL_MEM_FENCE);
				}
			}
		}
	}
	//barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void reduce_worker(
			__global Worker *workers,
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

	unsigned int group_size;
	unsigned int gid;
	
	if(use_gc==TYPE_GPU)
	{
		group_size = GPU_LOCAL_THREADS/NUM_GROUPS;
		gid = local_id/group_size; 
		linit(&objects[gid], TYPE_GPU);
	}
	else
	{
		group_size = CPU_LOCAL_THREADS/NUM_GROUPS;
		gid = local_id/group_size; 
		linit(&objects[gid], TYPE_CPU);
	}

	__local unsigned int global_object_offset[GPU_LOCAL_THREADS/WAVEFRONT_SIZE];
	
	int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
	int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;

	int num_blocks = cpu_blocks + gpu_blocks;

	int block_index = get_group_id(0);

	if(use_gc==TYPE_GPU)
	{
		block_index = block_index + cpu_blocks;
		ginit(global_object_offset);
	}
	
	else
	cinit(object_c);
	
	__local int task_index;
	__local int has_task;
	__local int finish;
	__local int num;

	if(local_id==0)
	{
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
			num = bufferinfos[task_index].num;

			if(has_task==1)
			{
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
			int i = local_id;
			__global Kvbuffer *buffer = &buffers[task_index];

			/*if use cpu, insert directly to the CPU reduction object*/
			if(use_gc == TYPE_CPU)
			{
				for(; i < num; i += CPU_LOCAL_THREADS)
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
					bufferinfos[task_index].full = 0;
					bufferinfos[task_index].scheduled = 0;
				}
			}

			/*if use gpu, use shared memory reduction objects*/
			else if(use_gc == TYPE_GPU)
			{
				for(; i < num; i += GPU_LOCAL_THREADS)
				{
					short key_size = kget_key_size(buffer, i);
					short value_size = kget_value_size(buffer, i);
					__global void *key = kget_key_address(buffer, i);
					__global void *value = kget_value_address(buffer, i);
					linsert_from_global(&objects[gid], key, key_size, value, value_size);
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if(local_id==0)
				{
					bufferinfos[task_index].full = 0;
					bufferinfos[task_index].scheduled = 0;	
				}
			}
		}
		//barrier(CLK_LOCAL_MEM_FENCE);
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