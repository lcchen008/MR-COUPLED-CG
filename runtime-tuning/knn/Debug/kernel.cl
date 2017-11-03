#pragma OPENCL EXTENSION cl_amd_printf : enable
void test(__global const void *data, uint size)
{
	const uint local_id =  get_local_id(0);	
	const uint global_id = get_global_id(0);

	if(global_id==0)
	{
		for(int i = 0; i<size/sizeof(int); i++)
		{
			printf("%d\n", ((unsigned int *)data)[i]);		
		}
	}
}
