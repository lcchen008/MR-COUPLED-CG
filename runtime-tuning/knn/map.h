#ifndef MAP
#define MAP

#include "lib\util_device.h"
#include "lib\roc_f.h"
#include "lib\rog_f.h"
#include "lib\rol_f.h"
#include "lib\kvbuffer.h"


bool map_cpu(__global Reduction_Object_C *object, __global void *global_data, __global void *offset)
{
		return true;
}

bool map_local(__local Reduction_Object_S *object, __global void *global_data, 
	__global void *offset, float threshold)
{
	int key = ((__global int *)global_data)[*(__global int *)offset];
	int dim1 = ((__global int *)global_data)[*(__global int *)offset + 1];
	int dim2 = ((__global int *)global_data)[*(__global int *)offset + 2];
	int dim3 = ((__global int *)global_data)[*(__global int *)offset + 3];

	float dist = (100-dim1)*(100-dim1)+(100-dim2)*(100-dim2)+(100-dim3)*(100-dim3);
	dist = sqrt(dist);

	float value = dist;

	if(value >= threshold)
		return true;
	else
	return linsert(object, &key, sizeof(int), &value, sizeof(float));
}

#endif