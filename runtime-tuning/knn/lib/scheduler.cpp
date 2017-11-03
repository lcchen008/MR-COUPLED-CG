#include <cstdio>
#include <cstdlib>
#include <iostream>
#define __NO_STD_STRING
#include <CL/cl.hpp>
#include "scheduler.h"
#include "util_host.h"
#include <time.h>
#include "ds.h"
#include "error_handling.h"
#include "worker_info.h"
#include <pthread.h>

Scheduler::Scheduler(void *global_data, 
		unsigned int global_data_size,
		void *global_data_offset,
		unsigned int global_data_offset_number,
		unsigned int offset_unit_size)
{
	this->global_data = global_data;
	this->global_data_size = global_data_size;
	this->global_offset = global_data_offset;
	this->offset_number = global_data_offset_number;
	this->unit_size = offset_unit_size;

	//std::cout<<"in the scheduler..."<<std::endl;
	cl_int err;
	std::vector<cl::Platform> platforms;
    std::cout<<"Getting Platform Information\n";
    err = cl::Platform::get(&platforms);
    if(err != CL_SUCCESS)
    {
        printf("%s\n",print_cl_errstring(err));
        return;
    }

	std::cout<<"Number of platforms found: "<<platforms.size()<<std::endl;

	std::vector<cl::Platform>::iterator i;
   	if(platforms.size() > 0)
    {
        for(i = platforms.begin(); i != platforms.end(); ++i)
        {
            	std::cout<<(*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str()<<std::endl;
			
				if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
            	{
                	break;
            	}
        }
    }

    if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}	

	platform = *i;

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };

	std::cout<<"Creating a context AMD platform\n";
    	context = cl::Context(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    	if(err != CL_SUCCESS)
    	{
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	std::cout<<"Getting device info\n";
    	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	std::cout<<"Num of devices: "<<devices.size()<<std::endl;

	cl_device_type type;
	devices[0].getInfo(CL_DEVICE_TYPE, &type);

	if(type==CL_DEVICE_TYPE_CPU)
	{
	std::cout<<"first device is cpu"<<std::endl;
	cpuqueue = cl::CommandQueue(context, devices[0], 0); 
	gpuqueue = cl::CommandQueue(context, devices[1], 0);
	}

	else if(type==CL_DEVICE_TYPE_GPU)
	{
	std::cout<<"first device is gpu"<<std::endl;
	gpuqueue = cl::CommandQueue(context, devices[0], 0); 
	cpuqueue = cl::CommandQueue(context, devices[1], 0);
	}

	rochcpu = (Reduction_Object_C *)malloc(sizeof(Reduction_Object_C));
	memset(rochcpu, 0, sizeof(Reduction_Object_C));
	rochcpu->num_buckets = NUM_BUCKETS_C;

	roghgpu = (Reduction_Object_G *)malloc(sizeof(Reduction_Object_G));
	memset(roghgpu, 0, sizeof(Reduction_Object_G));
	roghgpu->num_buckets = NUM_BUCKETS_G;

	int num_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS + CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
	workersh = (Worker *)malloc(sizeof(Worker)*num_blocks);

	std::cout<<"the number of blocks is: "<<num_blocks<<std::endl;

	for(int i = 0; i < num_blocks; i++)
	{
		workersh[i].has_task = 0;
		workersh[i].task_num = 0;
		workersh[i].test_data = 0;
	}
	
	workers = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(Worker)*num_blocks, workersh, &err);

	global_data_d =	cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, global_data_size, global_data, &err); 

	global_offset_d = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, offset_number*unit_size, global_offset, &err); 

	roccpu = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(Reduction_Object_C), rochcpu, &err);

	roggpu = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(Reduction_Object_G), roghgpu, &err);

	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	std::cout<<"Loading and compiling CL source: kernel.cl...\n";

	streamsdk::SDKFile sdkfile;

	if(!sdkfile.open("kernel.cl"))
	{
		std::cerr<<"We couldn't load CL source code\n";
        	return;
	}
	
	cl::Program::Sources sources_start(1, std::make_pair(sdkfile.source().c_str(), sdkfile.source().size()));

	program = cl::Program(context, sources_start, &err);

	if (err != CL_SUCCESS) 
	{
        	printf("%s\n",print_cl_errstring(err));
    }

    err = program.build(devices, "-I . -I lib");
    if (err != CL_SUCCESS) 
	{

			cl::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << " \n\t\t\tBUILD LOG\n";
			std::cout << " ************************************************\n";
			std::cout << str.c_str() << std::endl;
            std::cout << " ************************************************\n";
       
        	std::cerr << "Program::build() failed (" << err << ")\n";
			printf("%s\n",print_cl_errstring(err));
        	return;
    }
}


void Scheduler::do_mapreduce()
{
	
	//start();
	//start1();
	usetest();
	mergecg();
	sort_objectc();
	get_size_cpu();
	get_result_cpu();
}

void Scheduler::usetest()
{
	pthread_t tid[2];
	int device_types[2];
	device_types[0] = TYPE_CPU;
	device_types[1] = TYPE_GPU;

	//void*  arg0[1]={this};
	void*  arg1[2]={&device_types[0],   this};
	void*  arg2[2]={&device_types[1],   this};

	//pthread_create(&tid[0],NULL, master, arg0); //master
	pthread_create(&tid[0],NULL, test, arg1);   //cpu
	pthread_create(&tid[1],NULL, test, arg2); //gpu
	
	schedule();

	pthread_join(tid[0], NULL);
	printf("CPU finish...\n");
	pthread_join(tid[1], NULL);
	printf("GPU finish...\n");
}

void *Scheduler::test(void *arg)
{
	int* ptr_type = (int *)((void **)arg)[0];
	void* object = ((void **)arg)[1];

	printf("device type: %d\n", *ptr_type);

	int device_type = *ptr_type;
	Scheduler *scheduler = reinterpret_cast<Scheduler*>(object);

	cl_int err;
	cl::Kernel kernel(scheduler->program, "test", &err);

	err = kernel.setArg(0, scheduler->global_offset_d);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(1, scheduler->offset_number);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(2, scheduler->unit_size);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(3, scheduler->global_data_d);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(4, device_type);
		if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(5, scheduler->workers);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(6, scheduler->roccpu);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));
	err = kernel.setArg(7, scheduler->roggpu);
	if (err != CL_SUCCESS) 
		printf("%s\n",print_cl_errstring(err));

	if(device_type==TYPE_CPU)
	{
		printf("CPU queue enqueuing kernel...\n");
		
		
		err = scheduler->cpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS), 0, 0); 
		

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforequeue = clock();

		scheduler->cpuqueue.finish();

		clock_t afterqueue = clock();

		printf("CPU time: %d\n", afterqueue-beforequeue);
	}

	else if(device_type==TYPE_GPU)
	{	
		printf("GPU queue enqueuing kernel...\n");
		
		err = scheduler->gpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		cl::NDRange(GPU_GLOBAL_THREADS), cl::NDRange(GPU_LOCAL_THREADS), 0, 0);
		

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforequeue = clock();

		scheduler->gpuqueue.finish();

		clock_t afterqueue = clock();

		printf("GPU time: %d\n", afterqueue-beforequeue);
	}
	
	return ((void *)0);
}

void Scheduler::schedule()
{
	int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
	int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;
	int num_blocks = cpu_blocks + gpu_blocks;

	unsigned int task_offset = 0;
	int cpu = 0;
	int gpu = 0;
	int total_schedule = 0;

	Worker *workersp = (Worker *)cpuqueue.enqueueMapBuffer(workers, CL_TRUE, CL_MAP_WRITE, 0, num_blocks*sizeof(Worker), NULL, NULL, NULL);

	int divide = cpu_blocks;

	/*the first iteration allocates a size of default block size*/
	for(int i = 0; i < num_blocks; i++)
	{
		(this->worker_tuning)[i].total_num = 0;
	}

	bool flag_start = true;
	bool flag_end = true;
	bool gpu_start = false;

	while(1)
	{
		for(int i = 0; i < num_blocks; i++)
		{
			if(workersp[i].has_task==0)    //The thread block i has consumed all the tasks
			{
					if(i>=3&&flag_start)
					{
						flag_start = false;
						gpu_start = true;
					}

					int schedule_size;

					if((float)task_offset/(float)offset_number<0.1)
					{
						schedule_size = SMALL_SIZE;
					}

					/*profiling the speed of each processing core*/
					else if((float)task_offset/(float)offset_number < 0.2)
					{
						
						if(gpu_start)
						{
							total_schedule++;					
							worker_tuning[i].total_num++;
						}

						schedule_size = SMALL_SIZE;
					}

					else if((float)task_offset/(float)offset_number < 0.8)
					{
						if(flag_end)
						{
							for(int m = 0; m < num_blocks; m++)
							{
								worker_tuning[m].old_task_block_size = TASK_BLOCK_SIZE*worker_tuning[m].total_num/(total_schedule/num_blocks);
							}

							schedule_size = worker_tuning[i].old_task_block_size;

							flag_end = false;
						}

						else
							schedule_size = worker_tuning[i].old_task_block_size;
					}

					else
					{
						schedule_size = worker_tuning[i].old_task_block_size/40;
					}

					//schedule_size = TASK_BLOCK_SIZE;

					workersp[i].task_num = task_offset;

					workersp[i].task_block_size = schedule_size;

					workersp[i].has_task = 1;
			
					if(i<cpu_blocks)
							cpu++;
					else
							gpu++;

					task_offset += schedule_size;     //increase the task offset

					if(task_offset >= offset_number)
					{
						printf("finish scheduling..\n");
						break;
					}
			}
		}

		if(task_offset >= offset_number)
		{
			printf("final finish...\n");
			break;
		}
	}

	printf("master sending finish signal...\n");
			
	int count = 0;

	while(count != num_blocks)
	{
		for(int j = 0; j < num_blocks; j++)
		{
			if(workersp[j].has_task == 0)
			{
				workersp[j].has_task =-1;
				count++;
			}
		}
	}
		
	printf("******CPU: %d and GPU: %d******\n", cpu, gpu);

	for(int k = 0; k < num_blocks; k++)
	{
		printf("block size of %d: %d\n", k, worker_tuning[k].old_task_block_size);
	}
}


void *Scheduler::master(void *arg)
{
	return ((void *)0);
}

void Scheduler::get_size_cpu()
{
	printf("In the get size cpu\n");
	cl_int err;
	cl::Kernel kernel(program, "get_size_cpu", &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	key_num_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}
	key_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}
	value_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	kernel.setArg(0, roccpu);
	kernel.setArg(1, key_num_d);
	kernel.setArg(2, key_size_d);
	kernel.setArg(3, value_size_d);

	cpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS, 0, NULL));

	cpuqueue.finish();

	err = cpuqueue.enqueueReadBuffer(key_num_d, true, 0, sizeof(unsigned int), &key_num, 0, 0);
	err = cpuqueue.enqueueReadBuffer(key_size_d, true, 0, sizeof(unsigned int), &key_size, 0, 0);
	err = cpuqueue.enqueueReadBuffer(value_size_d, true, 0, sizeof(unsigned int), &value_size, 0, 0);

	cpuqueue.finish();

	value_num = key_num;

	if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::enqueueReadBuffer() failed (" << err << ")\n";
		return;
    }
	std::cout<<"The num of keys: "<<key_num<<std::endl;
	std::cout<<"The size of keys: "<<key_size<<std::endl;
	std::cout<<"The size of values: "<<value_size<<std::endl;

	std::cout<<"The number of buckets: "<<rochcpu->num_buckets<<std::endl;
}

void Scheduler::get_size_gpu()
{
	cl_int err;
	cl::Kernel kernel(program, "get_size_gpu", &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	key_num_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	key_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	value_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);

	kernel.setArg(0, roggpu);
	kernel.setArg(1, key_num_d);
	kernel.setArg(2, key_size_d);
	kernel.setArg(3, value_size_d);

	cpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS, 0, NULL));

	cpuqueue.finish();

	err = cpuqueue.enqueueReadBuffer(key_num_d, true, 0, sizeof(unsigned int), &key_num, 0, 0);
	err = cpuqueue.enqueueReadBuffer(key_size_d, true, 0, sizeof(unsigned int), &key_size, 0, 0);
	err = cpuqueue.enqueueReadBuffer(value_size_d, true, 0, sizeof(unsigned int), &value_size, 0, 0);

	value_num = key_num;

	if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::enqueueReadBuffer() failed (" << err << ")\n";
		return;
    }
	std::cout<<"The num of keys: "<<key_num<<std::endl;
	std::cout<<"The size of keys: "<<key_size<<std::endl;
	std::cout<<"The size of values: "<<value_size<<std::endl;

	std::cout<<"The number of buckets: "<<rochcpu->num_buckets<<std::endl;
}

void Scheduler::get_result_cpu()
{
	cl_int err;
	cl::Kernel kernel(program, "copy_to_array_cpu", &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	cl::Buffer key_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_size, NULL, &err);
	cl::Buffer value_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_size, NULL, &err);
	cl::Buffer key_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_num*sizeof(unsigned int), NULL, &err);
	cl::Buffer value_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_num*sizeof(unsigned int), NULL, &err);

	//set kernel args
	kernel.setArg(0, roccpu);
	kernel.setArg(1, key_array_d);
	kernel.setArg(2, value_array_d);
	kernel.setArg(3, key_index_d);
	kernel.setArg(4, value_index_d);

	cpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS, 0, NULL));
	cpuqueue.finish();

	char *key_array = (char *)malloc(key_size); 
	char *value_array = (char *)malloc(value_size);
	unsigned int *key_index = (unsigned int *)malloc(sizeof(unsigned int)*key_num);
	unsigned int *value_index = (unsigned int *)malloc(sizeof(unsigned int)*value_num);

	cpuqueue.enqueueReadBuffer(key_array_d, true, 0, key_size, key_array, 0, 0);
	cpuqueue.enqueueReadBuffer(value_array_d, true, 0, value_size, value_array, 0, 0);
	cpuqueue.enqueueReadBuffer(key_index_d, true, 0, sizeof(unsigned int)*key_num, key_index, 0, 0);
	cpuqueue.enqueueReadBuffer(value_index_d, true, 0, sizeof(unsigned int)*value_num, value_index, 0, 0);

	std::cout<<"The number of keys: "<<key_num<<std::endl;

	output.key_index = key_index;
	output.val_index = value_index;
	output.output_keys = key_array;
	output.output_vals = value_array;
}

void Scheduler::get_result_gpu()
{
	cl_int err;
	cl::Kernel kernel(program, "copy_to_array_gpu", &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	cl::Buffer key_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_size, NULL, &err);
	cl::Buffer value_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_size, NULL, &err);
	cl::Buffer key_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_num*sizeof(unsigned int), NULL, &err);
	cl::Buffer value_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_num*sizeof(unsigned int), NULL, &err);

	//set kernel args
	kernel.setArg(0, roggpu);
	kernel.setArg(1, key_array_d);
	kernel.setArg(2, value_array_d);
	kernel.setArg(3, key_index_d);
	kernel.setArg(4, value_index_d);

	gpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS, 0, NULL));
	gpuqueue.finish();

	char *key_array = (char *)malloc(key_size); 
	char *value_array = (char *)malloc(value_size);
	unsigned int *key_index = (unsigned int *)malloc(sizeof(unsigned int)*key_num);
	unsigned int *value_index = (unsigned int *)malloc(sizeof(unsigned int)*value_num);

	gpuqueue.enqueueReadBuffer(key_array_d, true, 0, key_size, key_array, 0, 0);
	gpuqueue.enqueueReadBuffer(value_array_d, true, 0, value_size, value_array, 0, 0);
	gpuqueue.enqueueReadBuffer(key_index_d, true, 0, sizeof(unsigned int)*key_num, key_index, 0, 0);
	gpuqueue.enqueueReadBuffer(value_index_d, true, 0, sizeof(unsigned int)*value_num, value_index, 0, 0);

	std::cout<<"The number of keys: "<<key_num<<std::endl;

	output.key_index = key_index;
	output.val_index = value_index;
	output.output_keys = key_array;
	output.output_vals = value_array;
}

void Scheduler::mergecg()
{
	cl_int err;
	cl::Kernel kernel(program, "mergecg", &err);
	if(err != CL_SUCCESS)
    {
        	printf("%s\n",print_cl_errstring(err));
        	return;
	}

	kernel.setArg(0, roccpu);
	kernel.setArg(1, roggpu);

	cpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CPU_GLOBAL_THREADS), cl::NDRange(CPU_LOCAL_THREADS, 0, NULL));
	cpuqueue.finish();

	printf("finish merge...\n");
}

void Scheduler::sort_objectc()
{
	int block_size = 32;
	int global_size = (int)ceil((double)NUM_BUCKETS_C/(double)block_size) * block_size;

	int err = 0;

	cl::Kernel kernel(program, "sort", &err);

	for(unsigned int k = 2; k <= NUM_BUCKETS_C; k*=2)
		for(unsigned int j = k/2; j > 0; j/=2)
		{
			kernel.setArg(0, roccpu);
			kernel.setArg(1, k);
			kernel.setArg(2, j);

			//printf("enqueueing kernel...\n");

			err = cpuqueue.enqueueNDRangeKernel(
			kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(block_size)
			);

			if (err != CL_SUCCESS) 
			{
				char *msg = print_cl_errstring(err);
				std::cout<<msg<<std::endl;
				std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
				" failed (" << err << ")\n";
				return;
			}

			err = cpuqueue.finish();
		}
}

unsigned int Scheduler::get_key_num()
{
	return key_num;
}

struct output Scheduler::get_output()
{
	return output;
}