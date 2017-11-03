#ifndef MAP
#define MAP
#include "lib\util_device.h"
#include "kmeans.h"

#include "lib\rog_f.h"
#include "lib\rol_f.h"

/*this map function maps inputs to the global reduction object*/
bool map_to_global(__global Reduction_Object_G *object, __global void *global_data, 
	__global void *offset, __local unsigned int *global_object_offset)
{
	return true;
}

bool map_to_local(__local Reduction_Object_S *object, __global void *global_data, 
	__global void *offset)
{
		int key = ((__global int *)global_data)[*(__global int *)offset];
		int dim1 = ((__global int *)global_data)[*(__global int *)offset + 1];
		int dim2 = ((__global int *)global_data)[*(__global int *)offset + 2];
		int dim3 = ((__global int *)global_data)[*(__global int *)offset + 3];

		float dist = (100-dim1)*(100-dim1)+(100-dim2)*(100-dim2)+(100-dim3)*(100-dim3);
		dist = sqrt(dist);

		float value = dist;

		return linsert(object, &key, sizeof(int), &value, sizeof(float));
}

#endif