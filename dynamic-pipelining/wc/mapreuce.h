#ifndef MAPREDUCE
#define MAPREDUCE
#include "lib\util_device.h"
#include "lib\rog.h"
#include "lib\rol.h"
#include "kmeans.h"
#include "lib\roc.h"
#include "lib\roc_f.h"
#include "lib\rog_f.h"
#include "lib\rol_f.h"
#include "lib\kvbuffer.h"
#include "lib\kvbuffer_f.h"
//#pragma OPENCL EXTENSION cl_amd_printf : enable

/*this map function maps inputs to the global reduction object*/
bool map(__global Reduction_Object_G *object, __global void *global_data, 
	__global void *offset, __local unsigned int *global_object_offset)
{
	unsigned int off = *(__global unsigned int *)offset;
	
	int dim1 = ((__global int *)global_data)[off];
	int dim2 = ((__global int *)global_data)[off+1];
	int dim3 = ((__global int *)global_data)[off+2];

	//printf("%d %d %d\n", dim1, dim2, dim3);

	unsigned int key = 0;
	float min_dist = 65536*65, dist;

	for(int i = 0; i < K; i++)
	{
		dist = 0;
		unsigned int cluster_dim1 = ((__global unsigned int *)global_data)[DIM*i];//((__global int *)global_data)[DIM*i];
		unsigned int cluster_dim2 = ((__global unsigned int *)global_data)[DIM*i+1];//((__global int *)global_data)[DIM*i+1];
		unsigned int cluster_dim3 = ((__global unsigned int *)global_data)[DIM*i+2];//((__global int *)global_data)[DIM*i+2];

		dist =	(cluster_dim1-dim1)*(cluster_dim1-dim1)+
				(cluster_dim2-dim2)*(cluster_dim2-dim2)+
				(cluster_dim3-dim3)*(cluster_dim3-dim3);
		dist = sqrt(dist);
		if(dist < min_dist)
		{
			min_dist = dist;
			key = i;
		}
	}

	float value[5];
	value[0] = dim1;
	value[1] = dim2;
	value[2] = dim3;
	value[3] = 1;
	value[4] = min_dist;

	return ginsert_from_private(object, &key, sizeof(key), value, sizeof(float)*5, global_object_offset);
}

/*maps to cpu ro*/
bool map_cpu(__global Reduction_Object_C *object, __global void *global_data, __global void *offset)
{
	unsigned int off = *(__global unsigned int *)offset;
	
	int dim1 = ((__global int *)global_data)[off];
	int dim2 = ((__global int *)global_data)[off+1];
	int dim3 = ((__global int *)global_data)[off+2];

	//printf("%d %d %d\n", dim1, dim2, dim3);

	unsigned int key = 0;
	float min_dist = 65536*65, dist;

	for(int i = 0; i < K; i++)
	{
		dist = 0;
		unsigned int cluster_dim1 = ((__global unsigned int *)global_data)[DIM*i];//((__global int *)global_data)[DIM*i];
		unsigned int cluster_dim2 = ((__global unsigned int *)global_data)[DIM*i+1];//((__global int *)global_data)[DIM*i+1];
		unsigned int cluster_dim3 = ((__global unsigned int *)global_data)[DIM*i+2];//((__global int *)global_data)[DIM*i+2];

		dist =	(cluster_dim1-dim1)*(cluster_dim1-dim1)+
				(cluster_dim2-dim2)*(cluster_dim2-dim2)+
				(cluster_dim3-dim3)*(cluster_dim3-dim3);
		dist = sqrt(dist);
		if(dist < min_dist)
		{
			min_dist = dist;
			key = i;
		}
	}

	//printf("I am %d and key is: %d\n", get_global_id(0), key);

	float value[5];
	value[0] = dim1;
	value[1] = dim2;
	value[2] = dim3;
	value[3] = 1;
	value[4] = min_dist;

	return cinsert_from_private(object, &key, sizeof(key), value, sizeof(float)*5);
}

/*this map function maps inputs to the local reduction object*/
bool map_local(__local Reduction_Object_S *object, __global void *global_data, __global void *offset)
{
	unsigned int off = *(__global unsigned int *)offset;
	
	int dim1 = ((__global int *)global_data)[off];
	int dim2 = ((__global int *)global_data)[off+1];
	int dim3 = ((__global int *)global_data)[off+2];

	//printf("%d %d %d\n", dim1, dim2, dim3);


	unsigned int key = 0;
	float min_dist = 65536*65, dist;

	for(int i = 0; i < K; i++)
	{
		dist = 0;
		unsigned int cluster_dim1 = ((__global unsigned int *)global_data)[DIM*i];//((__global int *)global_data)[DIM*i];
		unsigned int cluster_dim2 = ((__global unsigned int *)global_data)[DIM*i+1];//((__global int *)global_data)[DIM*i+1];
		unsigned int cluster_dim3 = ((__global unsigned int *)global_data)[DIM*i+2];//((__global int *)global_data)[DIM*i+2];

		dist =	(cluster_dim1-dim1)*(cluster_dim1-dim1)+
				(cluster_dim2-dim2)*(cluster_dim2-dim2)+
				(cluster_dim3-dim3)*(cluster_dim3-dim3);
		dist = sqrt(dist);
		if(dist < min_dist)
		{
			min_dist = dist;
			key = i;
		}
	}

	float value[5];
	value[0] = dim1;
	value[1] = dim2;
	value[2] = dim3;
	value[3] = 1;
	value[4] = min_dist;

	return linsert(object, &key, sizeof(key), value, sizeof(float)*5);
}

bool map_buffer(__global Kvbuffer *buffer, __global void *global_data, 
	__global void *offset, __local unsigned int *index_offset, __local unsigned int *pool_offset)
{
	unsigned int off = *(__global unsigned int *)offset;
	
	int dim1 = ((__global int *)global_data)[off];
	int dim2 = ((__global int *)global_data)[off+1];
	int dim3 = ((__global int *)global_data)[off+2];

	//printf("%d %d %d\n", dim1, dim2, dim3);

	unsigned int key = 0;
	float min_dist = 65536*65, dist;

	for(int i = 0; i < K; i++)
	{
		dist = 0;
		unsigned int cluster_dim1 = ((__global unsigned int *)global_data)[DIM*i];//((__global int *)global_data)[DIM*i];
		unsigned int cluster_dim2 = ((__global unsigned int *)global_data)[DIM*i+1];//((__global int *)global_data)[DIM*i+1];
		unsigned int cluster_dim3 = ((__global unsigned int *)global_data)[DIM*i+2];//((__global int *)global_data)[DIM*i+2];

		dist =	(cluster_dim1-dim1)*(cluster_dim1-dim1)+
				(cluster_dim2-dim2)*(cluster_dim2-dim2)+
				(cluster_dim3-dim3)*(cluster_dim3-dim3);
		dist = sqrt(dist);
		if(dist < min_dist)
		{
			min_dist = dist;
			key = i;
		}
	}

	float value[5];
	value[0] = dim1;
	value[1] = dim2;
	value[2] = dim3;
	value[3] = 1;
	value[4] = min_dist;

	return kinsert_g(buffer, &key, sizeof(key), value, sizeof(float)*5, index_offset, pool_offset);
}

#endif
