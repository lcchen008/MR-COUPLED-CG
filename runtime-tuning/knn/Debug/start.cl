#include "lib\rog.h"
#include "lib\rol.h"
#include "lib\rog_f.h"
#include "lib\rol_f.h"
#include "lib\configuration.h"
#include "..\mapreduce.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable


__kernel void start(global Reduction_Object_G *object_g, global void *global_data, 
global void *global_offset, unsigned int offset_number, unsigned int unit_size)
{
	//object_g->num_buckets = object_g->num_buckets + 9;
	//atom_add(&(object_g->num_buckets), 1);
	local unsigned int global_object_offset[LOCAL_THREADS/WAVEFRONT_SIZE];
    //initialize the offsets for global memory
	ginit(object_g, global_object_offset);
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_global_id(0);
    for(; i < offset_number; i++)
    {
        map_to_global(object_g, global_data, ((global char *)global_offset+unit_size*i), global_object_offset);
    }
}