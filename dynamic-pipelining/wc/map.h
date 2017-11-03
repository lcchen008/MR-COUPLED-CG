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
	if(*(__global int *)offset < 0)
		return true;

	__global char *key = (__global char *)global_data + *(__global int *)offset;

	//printf("char: %c\n", *key);

	__global char *p = key;

	int key_size = 1;
	while(*p!='\0')
	{
		key_size++;
		p++;
	};

	int value = 1;

	//return ginsert_from_gp(object, key, key_size, &value, sizeof(int), global_object_offset);
	return cinsert_from_gp(object, key, key_size, &value, sizeof(int));
}

bool map_local(__local Reduction_Object_S *object, __global void *global_data, 
	__global void *offset)
{
	//printf("offset is: %d\n", *(__global int *)offset);
	
	if(*(__global int *)offset < 0)
		return true;

	__global char *key = (__global char *)global_data + *(__global int *)offset;

	//printf("word: %c\n", *key);

	__global char *p = key;

	int key_size = 1;
	while(*p!='\0')
	{
		key_size++;
		p++;
	};

	int value = 1;

	//return ginsert_from_private(object, key, key_size, &value, sizeof(float)*5, global_object_offset);

	return linsert_from_gp(object, key, key_size, &value, sizeof(int));
}

bool map_buffer(__global Kvbuffer *buffer, __global void *global_data, 
	__global void *offset, __local unsigned int *index_offset, __local unsigned int *pool_offset)
{
	if(*(__global int *)offset < 0)
		return true;

	__global char *key = (__global char *)global_data + *(__global int *)offset;

	//printf("word: %c\n", *key);

	__global char *p = key;

	int key_size = 1;
	while(*p!='\0')
	{
		key_size++;
		p++;
	};

	int value = 1;

	return kinsert_from_gp(buffer, key, key_size, &value, sizeof(int), index_offset, pool_offset);
}

#endif