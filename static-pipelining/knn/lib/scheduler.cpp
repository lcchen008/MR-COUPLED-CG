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

	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};

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

	int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;
	int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;

	int num_blocks = gpu_blocks + cpu_blocks;

	int num_buffers = 0;

	#ifdef CMGR
	num_buffers = cpu_blocks;

	#else
	num_buffers = gpu_blocks;
	#endif

	/*allocating key value buffers*/
	buffersh = (Kvbuffer *)malloc(sizeof(Kvbuffer)*num_buffers*NUM_BUFFERS_IN_BLOCK);

	memset(buffersh, 0, sizeof(Kvbuffer)*num_buffers*NUM_BUFFERS_IN_BLOCK);

	buffers = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(Kvbuffer)*num_buffers*NUM_BUFFERS_IN_BLOCK, buffersh, &err);

	bufferinfosh = (Bufferinfo *)malloc(sizeof(Bufferinfo)*num_buffers*NUM_BUFFERS_IN_BLOCK);

	memset(bufferinfosh, 0, sizeof(Bufferinfo)*num_buffers*NUM_BUFFERS_IN_BLOCK);

	bufferinfos = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(Bufferinfo)*num_buffers*NUM_BUFFERS_IN_BLOCK, bufferinfosh, &err);

	if(err != CL_SUCCESS)
    {
        printf("%s\n",print_cl_errstring(err));
        return;
	}

	/*allocating global data and global offsets*/
	global_data_d =	cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, global_data_size, global_data, &err); 

	global_offset_d = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, offset_number*unit_size, global_offset, &err); 

	float threshold_h = 65535;

	threshold = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(float), &threshold_h, &err); 

	/*allocating reduction objects for both CPU and GPU*/
	rochcpu = (Reduction_Object_C *)malloc(sizeof(Reduction_Object_C));
	memset(rochcpu, 0, sizeof(Reduction_Object_C));
	rochcpu->num_buckets = NUM_BUCKETS_C;

	roghgpu = (Reduction_Object_G *)malloc(sizeof(Reduction_Object_G));
	memset(roghgpu, 0, sizeof(Reduction_Object_G));
	roghgpu->num_buckets = NUM_BUCKETS_G;

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
	usetest1();
	//mergecg();
	//begin();

#ifdef CMGR
	sort_objectg();
	get_size_gpu();
	get_result_gpu();
#else
	sort_objectc();
	get_size_cpu();
	get_result_cpu();
#endif
}

void Scheduler::begin()
{
	cl_int err;
	
	int mapper, reducer;

	cl::CommandQueue mapqueue;
	cl::CommandQueue reducequeue;

	int map_global_threads;
	int map_local_threads;

	int reduce_global_threads;
	int reduce_local_threads;
	int use_gc;

	#ifdef CMGR             //cpu map and gpu reduce

	mapper = TYPE_CPU;
	reducer = TYPE_GPU;
	mapqueue = cpuqueue;
	reducequeue = gpuqueue;
	map_global_threads = CPU_GLOBAL_THREADS;
	map_local_threads = CPU_LOCAL_THREADS;
	reduce_global_threads = GPU_GLOBAL_THREADS;
	reduce_local_threads = GPU_LOCAL_THREADS;
	use_gc = TYPE_GPU;

	#else

	mapper = TYPE_GPU;
	reducer = TYPE_CPU;
	mapqueue = gpuqueue;
	reducequeue = cpuqueue;
	map_global_threads = GPU_GLOBAL_THREADS;
	map_local_threads = GPU_LOCAL_THREADS;
	reduce_global_threads = CPU_GLOBAL_THREADS;
	reduce_local_threads = CPU_LOCAL_THREADS;
	use_gc = TYPE_CPU;

	#endif	
	
	
	/*do map*/
		printf("mapper queue enqueuing kernel...\n");
		
		cl::Kernel map_kernel(program, "map_worker", &err);

		err = map_kernel.setArg(0, global_offset_d);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(1, offset_number);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(2, unit_size);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(3, global_data_d);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(4, buffers);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(5, roggpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(6, roccpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(7, bufferinfos);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(8, use_gc);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = map_kernel.setArg(9, threshold);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		

		err = mapqueue.enqueueNDRangeKernel(map_kernel, cl::NullRange, 
		cl::NDRange(map_global_threads), cl::NDRange(map_local_threads), 0, 0);

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforequeue = clock();

		mapqueue.flush();

		clock_t afterqueue = clock();

		std::cout<<"map time: "<<(afterqueue-beforequeue)<<std::endl;

		/*do reduce*/
		printf("reducer queue enqueuing kernel...\n");

		cl::Kernel kernel(program, "reduce_worker", &err);

		err = kernel.setArg(0, buffers);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(1, roggpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(2, roccpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(3, bufferinfos);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(4, use_gc);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(5, threshold);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = reducequeue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		cl::NDRange(reduce_global_threads), cl::NDRange(reduce_local_threads), 0, 0); 

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforereducequeue = clock();

		reducequeue.finish();

		clock_t afterreducequeue = clock();

		std::cout<<"reduce time: "<<(afterreducequeue-beforereducequeue)<<std::endl;


}

void Scheduler::usetest1()
{
	pthread_t tid[2];
	int device_types[2];
	device_types[0] = TYPE_CPU;
	device_types[1] = TYPE_GPU;

	//void*  arg0[1]={this};
	void*  arg1[2]={&device_types[0],   this};
	void*  arg2[2]={&device_types[1],   this};

	//pthread_create(&tid[0],NULL, master, arg0); //master
	pthread_create(&tid[1],NULL, test1, arg2); //gpu
	pthread_create(&tid[0],NULL, test1, arg1);   //cpu

	//schedule();
	//printf("master finishes...\n");

	pthread_join(tid[1], NULL);
	printf("GPU finish...\n");


	pthread_join(tid[0], NULL);
	printf("CPU finish...\n");
}

void *Scheduler::test1(void *arg)
{
	int* ptr_type = (int *)((void **)arg)[0];
	void* object = ((void **)arg)[1];

	printf("device type: %d\n", *ptr_type);

	int device_type = *ptr_type;
	Scheduler *scheduler = reinterpret_cast<Scheduler*>(object);

	cl_int err;
	
	int mapper, reducer;

	cl::CommandQueue mapqueue;
	cl::CommandQueue reducequeue;

	int map_global_threads;
	int map_local_threads;

	int reduce_global_threads;
	int reduce_local_threads;
	int use_gc;

	#ifdef CMGR             //cpu map and gpu reduce

	mapper = TYPE_CPU;
	reducer = TYPE_GPU;
	mapqueue = scheduler->cpuqueue;
	reducequeue = scheduler->gpuqueue;
	map_global_threads = CPU_GLOBAL_THREADS;
	map_local_threads = CPU_LOCAL_THREADS;
	reduce_global_threads = GPU_GLOBAL_THREADS;
	reduce_local_threads = GPU_LOCAL_THREADS;
	use_gc = TYPE_GPU;

	#else

	mapper = TYPE_GPU;
	reducer = TYPE_CPU;
	mapqueue = scheduler->gpuqueue;
	reducequeue = scheduler->cpuqueue;
	map_global_threads = GPU_GLOBAL_THREADS;
	map_local_threads = GPU_LOCAL_THREADS;
	reduce_global_threads = CPU_GLOBAL_THREADS;
	reduce_local_threads = CPU_LOCAL_THREADS;
	use_gc = TYPE_CPU;

	#endif

	/*reducer*/
	if(device_type==reducer)
	{
		printf("reducer queue enqueuing kernel...\n");

		cl::Kernel kernel(scheduler->program, "reduce_worker", &err);

		err = kernel.setArg(0, scheduler->buffers);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(1, scheduler->roggpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(2, scheduler->roccpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(3, scheduler->bufferinfos);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(4, use_gc);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(5, scheduler->threshold);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = reducequeue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		cl::NDRange(reduce_global_threads), cl::NDRange(reduce_local_threads), 0, 0); 

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforequeue = clock();

		reducequeue.finish();

		clock_t afterqueue = clock();

		std::cout<<"reduce time: "<<(afterqueue-beforequeue)<<std::endl;
	}

	/*mapper*/
	else if(device_type==mapper)
	{	
		printf("mapper queue enqueuing kernel...\n");
		
		cl::Kernel kernel(scheduler->program, "map_worker", &err);

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
		
		err = kernel.setArg(4, scheduler->buffers);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = kernel.setArg(5, scheduler->roggpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = kernel.setArg(6, scheduler->roccpu);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		err = kernel.setArg(7, scheduler->bufferinfos);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));
		//err = kernel.setArg(9, scheduler->test_data_d);
		//if (err != CL_SUCCESS) 
		//	printf("%s\n",print_cl_errstring(err));
		err = kernel.setArg(8, use_gc);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = kernel.setArg(9, scheduler->threshold);
		if (err != CL_SUCCESS) 
			printf("%s\n",print_cl_errstring(err));

		err = mapqueue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		cl::NDRange(map_global_threads), cl::NDRange(map_local_threads), 0, 0);

		if (err != CL_SUCCESS) 
		{
        	printf("%s\n",print_cl_errstring(err));
		}

		clock_t beforequeue = clock();

		mapqueue.finish();

		clock_t afterqueue = clock();

		std::cout<<"map time: "<<(afterqueue-beforequeue)<<std::endl;
	}
	
	return ((void *)0);
}


void Scheduler::schedule()
{
	int cpu_blocks = CPU_GLOBAL_THREADS/CPU_LOCAL_THREADS;
	int gpu_blocks = GPU_GLOBAL_THREADS/GPU_LOCAL_THREADS;
	int num_blocks = cpu_blocks + gpu_blocks;

	int map_blocks;
	int reduce_blocks;

	int map_start;
	int map_end;
	int reduce_start;
	int reduce_end;

#ifdef CMGR
	map_blocks = cpu_blocks;
	reduce_blocks = gpu_blocks;
	map_start = 0;
	map_end = cpu_blocks;
	reduce_start = cpu_blocks;
	reduce_end = num_blocks;
	printf("map_start is %d and map_end is: %d\n", map_start, map_end);
	printf("reduce_start is %d and reduce_end is: %d\n", reduce_start, reduce_end);
#else
	map_blocks = gpu_blocks;
	reduce_blocks = cpu_blocks;
	map_start = cpu_blocks;
	map_end = num_blocks;
	reduce_start = 0;
	reduce_end = cpu_blocks;
	printf("map_start is %d and map_end is: %d\n", map_start, map_end);
	printf("reduce_start is %d and reduce_end is: %d\n", reduce_start, reduce_end);
#endif

	unsigned int task_offset = 0;
	unsigned int reduce_task_offset = 0;
	
	int reduce = 0;

	Worker *workersp = (Worker *)cpuqueue.enqueueMapBuffer(workers, CL_TRUE, CL_MAP_WRITE, 0, num_blocks*sizeof(Worker), NULL, NULL, NULL);

	//int *testdatap = (int *)cpuqueue.enqueueMapBuffer(test_data_d, CL_TRUE, CL_MAP_WRITE, 0, num_blocks*sizeof(int), NULL, NULL, NULL);

	Bufferinfo *bufferinfosp = (Bufferinfo *)cpuqueue.enqueueMapBuffer(bufferinfos, CL_TRUE, CL_MAP_WRITE, 0, map_blocks*sizeof(Bufferinfo)*NUM_BUFFERS_IN_BLOCK, NULL, NULL, NULL);

	/*for(int i = 0; i < gpu_blocks*NUM_BUFFERS_IN_BLOCK; i++)
		printf("index: %d\n", bufferinfosp[i].full);*/

	while(1)
	{
		/*schedule map tasks first*/
		if(task_offset < offset_number)
		{	
			for(int i = map_start; i < map_end; i++)
			{
				if(workersp[i].has_task==0)    //The thread block i has consumed all the tasks
				{
					//printf("I scheduled a map task %d to block %d\n", task_offset, i);
					
					workersp[i].task_num = task_offset;
					workersp[i].has_task = 1;
		                			
					task_offset += TASK_BLOCK_SIZE;     //increase the task offset

					if(task_offset >= offset_number)
					{
						//printf("finish map scheduling..\n");
						break;
					}
				}
			}
		}

		/*schedule reduce tasks*/
		for(int j = reduce_start; j < reduce_end; j++)
		{
			int count = 0;
			if(workersp[j].has_task==0)
			{
				while(count < map_blocks*NUM_BUFFERS_IN_BLOCK)
				{
					if(bufferinfosp[reduce_task_offset].full&&(bufferinfosp[reduce_task_offset].scheduled==0))
					{
						//printf("I scheduled a reduce task %d to block %d\n", reduce_task_offset, j);
						if(bufferinfosp[reduce_task_offset].num!=0)
						{
							reduce++;
						}
						bufferinfosp[reduce_task_offset].scheduled = 1; 
						workersp[j].task_num = reduce_task_offset;
						workersp[j].has_task = 1;
						reduce_task_offset = (reduce_task_offset + 1)%(map_blocks*NUM_BUFFERS_IN_BLOCK);
						break;
					}
					reduce_task_offset = (reduce_task_offset + 1)%(map_blocks*NUM_BUFFERS_IN_BLOCK);
					count++;
				}
			}
		}

		if(task_offset >= offset_number)
		{
			break;
		}
	}

	printf("make sure there are empty buffers\n");

	/*make sure that there are kv buffers available for mappers*/
	for(int i = 0; i < map_blocks*NUM_BUFFERS_IN_BLOCK; i++)
	{
		if(bufferinfosp[i].full&&(bufferinfosp[i].scheduled!=1))
		{
			//printf("has task...\n");
			bool scheduled = 0;

			while(!scheduled)
			for(int j = reduce_start; j < reduce_end; j++)
			{
				if(workersp[j].has_task==0)
				{
					//printf("I scheduled a reduce task %d to block %d\n", i, j);
					if(bufferinfosp[reduce_task_offset].num!=0)
					reduce++;
					bufferinfosp[i].scheduled = 1; //has been scheduled
					workersp[j].task_num = i;
					workersp[j].has_task = 1;
					scheduled = 1;
					break;		//after scheduling one task, go out and scheduling for next CPU
				}
			}
		}
	}

	printf("master sending map finish signal...\n");
			
	int count = 0;

	/*send finish signal to gpus (map workers)*/
	while(count != map_blocks)
	{
		for(int j = map_start; j < map_end; j++)
		{
			if(workersp[j].has_task == 0)
			{
				//printf("mapper %d receives signal...\n", j);
				workersp[j].has_task = -1;
				count++;
			}
		}
	}

	printf("all map workers receive end signal...\n");

	count = 0;

	/*wait for all map tasks to be finished*/
	while(count < map_blocks)
	{
		for(int j = map_start; j < map_end; j++)
		{
			//printf("%d is: %d\n", j, workersp[j].finish_all);
			if(workersp[j].finish_all == 1)
			{
				//printf("%d finish all is valid...\n", j);
				workersp[j].finish_all = 0;
				count++;
			}
		}
	}

	printf("scheduling remaining reduce work...\n");

	/*next, schedule the remaining reduce tasks*/
	for(int i = 0; i < map_blocks*NUM_BUFFERS_IN_BLOCK; i++)
	{
		if(bufferinfosp[i].full&&(bufferinfosp[i].scheduled==0))
		{
			bool scheduled = 0;

			while(!scheduled)
			for(int j = reduce_start; j < reduce_end; j++)
			{
				if(workersp[j].has_task==0)
				{
					//printf("I scheduled a reduce task %d to block %d\n", i, j);
					if(bufferinfosp[reduce_task_offset].num!=0)
					reduce++;
					bufferinfosp[i].scheduled = 1; //has been scheduled
					workersp[j].task_num = i;
					workersp[j].has_task = 1;
					scheduled = 1;
					break;		//after scheduling one task, go out and scheduling for next CPU
				}
			}
		}
	}

	/*send finish signal to reduce workers*/
	count = 0;

	printf("master sending reduce finish signal...\n");

	while(count != reduce_blocks)
	{
		for(int j = reduce_start; j < reduce_end; j++)
		{
			if(workersp[j].has_task == 0)
			{
				//printf("reducer %d finishes...\n", j);
				workersp[j].has_task = -1;
				count++;
			}
		}
	}

	int total_reduce = 0;

	for(int i = 0; i < 8; i++)
		total_reduce += workersp[i].test_data;

	/*for(int i = reduce_start; i < reduce_end; i++)
		total_reduce+=testdatap[i];*/

	printf("total reduce: %d\n", total_reduce);

	printf("reduce: %d\n", reduce);
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

	gpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(GPU_GLOBAL_THREADS), cl::NDRange(GPU_LOCAL_THREADS, 0, NULL));

	gpuqueue.finish();

	err = gpuqueue.enqueueReadBuffer(key_num_d, true, 0, sizeof(unsigned int), &key_num, 0, 0);
	err = gpuqueue.enqueueReadBuffer(key_size_d, true, 0, sizeof(unsigned int), &key_size, 0, 0);
	err = gpuqueue.enqueueReadBuffer(value_size_d, true, 0, sizeof(unsigned int), &value_size, 0, 0);

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

	gpuqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(GPU_GLOBAL_THREADS), cl::NDRange(GPU_LOCAL_THREADS, 0, NULL));
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
	printf("In merge CG\n");
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

unsigned int Scheduler::get_key_num()
{
	return key_num;
}

struct output Scheduler::get_output()
{
	return output;
}

void Scheduler::sort_objectc()
{
	int block_size = 32;
	int global_size = (int)ceil((double)NUM_BUCKETS_C/(double)block_size) * block_size;

	int err = 0;

	cl::Kernel kernel(program, "sort_cpu", &err);

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

void Scheduler::sort_objectg()
{
	int block_size = 32;
	int global_size = (int)ceil((double)NUM_BUCKETS_G/(double)block_size) * block_size;

	int err = 0;

	cl::Kernel kernel(program, "sort_gpu", &err);

	for(unsigned int k = 2; k <= NUM_BUCKETS_G; k*=2)
		for(unsigned int j = k/2; j > 0; j/=2)
		{
			kernel.setArg(0, roggpu);
			kernel.setArg(1, k);
			kernel.setArg(2, j);

			//printf("enqueueing kernel...\n");

			err = gpuqueue.enqueueNDRangeKernel(
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

			err = gpuqueue.finish();
		}
}