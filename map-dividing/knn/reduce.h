#ifndef REDUCE
#define REDUCE

void reduce_local_to_global(__global void *value1, unsigned short value1_size, __local void *value2, unsigned short value2_size)
{
	int temp = *(__global int *)value1 + *(__local int *)value2;
	copyVal_private_to_global(value1, &temp, sizeof(int));
}

void reduce_private_to_global(__global void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
{
	int temp = *(__global int *)value1 + *(int *)value2;
	copyVal_private_to_global(value1, &temp, sizeof(int));
}

void reduce_private_to_local(__local void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
{
	int temp = *(__local int *)value1 + *(int *)value2;
	copyVal_private_to_local(value1, &temp, sizeof(int));
}


bool equal_global_and_global(__global void *key1, unsigned size1, __global void *key2, unsigned size2)
{
	if(size1!=size2)
		return false;

	//printf("equal 1\n");

	__global char *k1 = (__global char *)key1;
	__global char *k2 = (__global char *)key2;
	
	for(int i = 0; i< size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}

	//printf("equal 2...\n");
	return true;
}


bool equal_global_and_local(__global void *key1, unsigned size1, __local void *key2, unsigned size2)
{
	if(size1!=size2)
		return false;

	//printf("equal 1\n");

	__global char *k1 = (__global char *)key1;
	__local char *k2 = (__local char *)key2;
	
	for(int i = 0; i< size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}

	//printf("equal 2...\n");
	return true;
}


bool equal_private_and_global(void *key1, unsigned size1, __global void *key2, unsigned size2)
{
	if(size1!=size2)
		return false;

	//printf("equal 1\n");

	char *k1 = (char *)key1;
	__global char *k2 = (__global char *)key2;
	
	for(int i = 0; i< size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}

	//printf("equal 2...\n");
	return true;
}

bool equal_local_and_global(__local void *key1, unsigned size1, __global void *key2, unsigned size2)
{
	if(size1!=size2)
		return false;

	//printf("equal 1\n");
	__local char *k1 = (__local char *)key1;
	__global char *k2 = (__global char *)key2;

	for(int i = 0; i< size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}
	return true;
	//printf("equal 2\n");
}

bool equal_private_and_local(void *key1, unsigned size1, __local void *key2, unsigned size2)
{
	if(size1!=size2)
		return false;
	char *k1 = (char *)key1;
	__local char *k2 = (__local char *)key2;
	
	for(int i = 0; i< size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}
	
	return true;
}

int compare_local(__local void *key1, unsigned int key_size1, __local void *key2, unsigned int key_size2, 
	__local void *value1, unsigned int value_size1, __local void *value2, unsigned int value_size2)
{
	float a = *(__local float *)value1;
	float b = *(__local float *)value2;

	if(a > b)
		return 1;
	else if(a < b)
		return -1;
	else
		return 0;
}

int compare_global(__global void *key1, unsigned int key_size1, __global void *key2, unsigned int key_size2, 
	__global void *value1, unsigned int value_size1, __global void *value2, unsigned int value_size2)
{
	float a = *(__global float *)value1;
	float b = *(__global float *)value2;

	if(a > b)
		return 1;
	else if(a < b)
		return -1;
	else
		return 0;
}

#endif