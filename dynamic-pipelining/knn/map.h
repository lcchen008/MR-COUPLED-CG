#ifndef MAP
#define MAP

#include "lib\util_device.h"
#include "lib\roc_f.h"
#include "lib\rog_f.h"
#include "lib\rol_f.h"
#include "lib\kvbuffer.h"
#include "lib\kvbuffer_f.h"

bool map_cpu(__global Reduction_Object_C *object, __global void *global_data, __global void *offset)
{
	return true;
}

bool map_local(__local Reduction_Object_S *object, __global void *global_data, 
	__global void *offset)
{
	int key = ((__global int *)global_data)[*(__global int *)offset];
	int dim1 = ((__global int *)global_data)[*(__global int *)offset + 1];
	int dim2 = ((__global int *)global_data)[*(__global int *)offset + 2];
	int dim3 = ((__global int *)global_data)[*(__global int *)offset + 3];

	float dist = (100-dim1)*(100-dim1)+(100-dim2)*(100-dim2)+(100-dim3)*(100-dim3);
	dist = sqrt(dist);

	float value = dist;

	/*if(key > 10000)
		printf("key is: %d\n", key);*/

	return linsert(object, &key, sizeof(int), &value, sizeof(float));
}

bool map_buffer(__global Kvbuffer *buffer, __global void *global_data, 
	__global void *offset, __local unsigned int *index_offset, __local unsigned int *pool_offset)
{
	int key = ((__global int *)global_data)[*(__global int *)offset];
	int dim1 = ((__global int *)global_data)[*(__global int *)offset + 1];
	int dim2 = ((__global int *)global_data)[*(__global int *)offset + 2];
	int dim3 = ((__global int *)global_data)[*(__global int *)offset + 3];

	float dist = (100-dim1)*(100-dim1)+(100-dim2)*(100-dim2)+(100-dim3)*(100-dim3);
	dist = sqrt(dist);

	float value = dist;

	//printf("one key value pair...\n");
	int ret = kinsert_g(buffer, &key, sizeof(key), &value, sizeof(float), index_offset, pool_offset);
	if(ret != -1)
	return true;
	else
		return false;
}

#endif