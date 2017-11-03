#include <cstdio>
#include <cstdlib>
#include <iostream>
#define __NO_STD_STRING
#include <CL/cl.hpp>
#include "scheduler.h"
#include "util_host.h"
#include "configuration.h"
#include <time.h>
#include "ds.h"
#include "error_handling.h"

//initialize the scheduler
Scheduler::Scheduler(void *global_data, 
unsigned int global_data_size, 
//void *local_data,
//unsigned int local_data_size,
void *global_data_offset, 
unsigned int global_data_offset_number, 
unsigned int unit_size, 
bool sort,
bool usegpu,
int uselocal)
{
this->global_data = global_data;
	this->global_data_size = global_data_size;
	this->local_data = local_data;
	this->local_data_size = local_data_size;

	this->global_data_offset = global_data_offset;
	this->global_data_offset_number = global_data_offset_number;
	this->unit_size = unit_size;

	this->sort = sort;
	this->usegpu = usegpu;
	this->uselocal = uselocal;

	cl_int err;
	// Platform info
    std::vector<cl::Platform> platforms;
    std::cout<<"HelloCL!\nGetting Platform Information\n";
    err = cl::Platform::get(&platforms);
    if(err != CL_SUCCESS)
    {
        std::cerr << "Platform::get() failed (" << err << ")" << std::endl;
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
        std::cerr << "Platform::getInfo() failed (" << err << ")" << std::endl;
        return;
    }

	platform = *i;

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };

	std::cout<<"Creating a context AMD platform\n";
    context = cl::Context(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) 
	{
        std::cerr << "Context::Context() failed (" << err << ")\n";
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

#ifdef USE_CPU
	commandqueue = cpuqueue;
#else
	commandqueue = gpuqueue;
#endif

	std::cout<<"Num of devices: "<<devices.size()<<std::endl;

    if (err != CL_SUCCESS) 
	{
        std::cerr << "Context::getInfo() failed (" << err << ")\n";
        return;
    }

    if (devices.size() == 0) 
	{
        std::cerr << "No device available\n";
        return;
    }


    if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::CommandQueue() failed (" << err << ")\n";
    }

		/*Load the kernel: start*/
	std::cout<<"Loading and compiling CL source: start.cl...\n";

	Kernelfile file;
	if(!file.open("start.cl"))
	{
		std::cerr<<"We couldn't load CL source code\n";
        return;
	}

	//std::cout<<"source content: "<<file.source()<<std::endl;

	cl::Program::Sources sources_start(1, std::make_pair(file.source(), file.size()));

	//cl_int err;
	this->program = cl::Program(context, sources_start, &err);

	if (err != CL_SUCCESS) 
	{
        std::cerr << "Program::Program() failed (" << err << ")\n";
        return;
    }

    err = program.build(devices, "-I . -I lib");
    if (err != CL_SUCCESS) 
	{
		if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            cl::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
			std::cout << str.c_str() << std::endl;
            std::cout << " ************************************************\n";
        }
        std::cerr << "Program::build() failed (" << print_cl_errstring(err) << ")\n";
        return;
    }
}

void Scheduler::start()
{
	int err;

	cl::Kernel kernel(program, "start", &err);
    if (err != CL_SUCCESS) 
	{
        std::cerr << "Kernel::Kernel() \"start\" failed (" << err << ")\n";
        return;
    }

	//copy global data from host to device
	global_data_d = cl::Buffer(context, CL_MEM_READ_ONLY, global_data_size, NULL, &err);
	
	if (err != CL_SUCCESS)
	{
		std::cerr << "global_data_d Buffer::Buffer() failed (" << err << ")\n";
        return;
	}

	//copy local data from host to device
	/*local_data_d = cl::Buffer(context, CL_MEM_READ_ONLY, local_data_size, NULL, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "local_data_d Buffer::Buffer() failed (" << err << ")\n";
        return;
	}*/

	//copy offset from host to device
	global_data_offset_d = cl::Buffer(context, CL_MEM_READ_ONLY, global_data_offset_number*unit_size, NULL, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "global_data_offset_d Buffer::Buffer() failed (" << err << ")\n";
        return;
	}

	rogh = (Reduction_Object_G *)malloc(sizeof(Reduction_Object_G));
	memset(rogh, 0, sizeof(Reduction_Object_G));
	rogh->num_buckets = NUM_BUCKETS_G;

	//copy reduction object from host to device
	rog = cl::Buffer(context, CL_MEM_READ_WRITE /*| CL_MEM_ALLOC_HOST_PTR*/, sizeof(Reduction_Object_G), NULL, &err);

	//void *local_data_map = NULL;
	void *global_data_map = NULL;
	void *global_data_offset_map = NULL;
	void *rog_map = NULL;

	cl_int eventStatus = CL_QUEUED;

	//cl::Event ev_local_data_map;
	cl::Event ev_data_map;
    cl::Event ev_offset_map;
    cl::Event ev_rog_map;
	//cl::Event ev_unmap_local_data_map;
	cl::Event ev_unmap_data_map;
    cl::Event ev_unmap_offset_map;
    cl::Event ev_unmap_rog_map;
	cl::Event ev_kernel_finish;

	global_data_map = commandqueue.enqueueMapBuffer(
                    global_data_d,
                    CL_FALSE,
                    CL_MAP_WRITE,
                    0,
                    global_data_size,
                    NULL,
                    &ev_data_map,
                    &err);

	if (err != CL_SUCCESS)
	{
		std::cerr << "global_data_map failed (" << err << ")\n";
        return;
	}

	global_data_offset_map = commandqueue.enqueueMapBuffer(
                    global_data_offset_d,
                    CL_FALSE,
                    CL_MAP_WRITE,
                    0,
                    global_data_offset_number*unit_size,
                    NULL,
                    &ev_offset_map,
                    &err);

	if (err != CL_SUCCESS)
	{
		std::cerr << "global_data_offset_map failed (" << err << ")\n";
        return;
	}

	rog_map = commandqueue.enqueueMapBuffer(
                    rog,
                    CL_FALSE,
                    CL_MAP_WRITE,
                    0,
                    sizeof(Reduction_Object_G),
                    NULL,
                    &ev_rog_map,
                    &err);

	if (err != CL_SUCCESS)
	{
		std::cerr << "rog_map failed (" << err << ")\n";
        return;
	}

	commandqueue.flush();

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_data_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	//memcpy(local_data_map, local_data, local_data_size);

	/*eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_local_data_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }*/

	memcpy(global_data_map, global_data, global_data_size);

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_offset_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	memcpy(global_data_offset_map, global_data_offset, global_data_offset_number*unit_size);

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_rog_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	memcpy(rog_map, rogh, sizeof(Reduction_Object_G));

	//err = commandqueue.enqueueUnmapMemObject(local_data_d, local_data_map, NULL, &ev_unmap_local_data_map);
	err = commandqueue.enqueueUnmapMemObject(global_data_d, global_data_map, NULL, &ev_unmap_data_map);
	err = commandqueue.enqueueUnmapMemObject(global_data_offset_d, global_data_offset_map, NULL, &ev_unmap_offset_map);
	err = commandqueue.enqueueUnmapMemObject(rog, rog_map, NULL, &ev_unmap_rog_map);

	commandqueue.finish();

	/*eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_unmap_local_data_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }*/

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_unmap_data_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_unmap_offset_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_unmap_rog_map.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

	err = kernel.setArg(0, rog);
	err = kernel.setArg(1, global_data_d);
	err = kernel.setArg(2, global_data_offset_d);
	err = kernel.setArg(3, global_data_offset_number);
	//err = kernel.setArg(4, local_data_d);
	//err = kernel.setArg(5, local_data_size, NULL);
	//err = kernel.setArg(6, local_data_size);
	err = kernel.setArg(4, unit_size);
	err = kernel.setArg(5, uselocal);

	//enqueuing the kernel to the command queue
	std::cout<<"Running CL program\n";

	clock_t before = clock();
    err = commandqueue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(GLOBAL_THREADS), cl::NDRange(LOCAL_THREADS), 0, &ev_kernel_finish);

	std::cout<<"enqueue finish..."<<std::endl;

	if (err != CL_SUCCESS) 
	{
       printf("%s\n",print_cl_errstring(err));
       return;
    }

	err = commandqueue.finish();

	eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        err = ev_kernel_finish.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS, 
                    &eventStatus);
        if (err != CL_SUCCESS)
		{
			std::cerr << "rog_map failed (" << err << ")\n";
			return;
		}
    }

    //err = commandqueue.finish();
	std::cout<<"command queue finish..."<<std::endl;
	clock_t after = clock();

	std::cout<<"Execution time: "<<after-before<<std::endl;

    if (err != CL_SUCCESS) 
	{
        std::cerr << "Event::wait() failed (" << err << ")\n";
    }

	err = commandqueue.enqueueReadBuffer(rog, true, 0, sizeof(Reduction_Object_G), rogh, 0, 0);

	if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::enqueueReadBuffer() failed (" << err << ")\n";
		return;
    }

	std::cout<<"The number of buckets: "<<rogh->num_buckets<<std::endl;
}

void Scheduler::get_size()
{
	int err;
	cl::Kernel kernel(program, "get_size", &err);
    if (err != CL_SUCCESS) 
	{
        std::cerr << "Kernel::Kernel() \"get_size\" failed (" << err << ")\n";
        return;
    }

	key_num_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	key_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	value_size_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);

	kernel.setArg(0, rog);
	kernel.setArg(1, key_num_d);
	kernel.setArg(2, key_size_d);
	kernel.setArg(3, value_size_d);

	//enqueuing the kernel to the command queue
	std::cout<<"Running CL program\n";

	int local_threads_get_size = 4;
	int global_threads_get_size = 4;

	clock_t before = clock();
    err = commandqueue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(global_threads_get_size), cl::NDRange(local_threads_get_size)
    );

	if (err != CL_SUCCESS) 
	{
		std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return;
    }

    err = commandqueue.finish();
	std::cout<<"command queue finish..."<<std::endl;
	clock_t after = clock();

	std::cout<<"Execution time of get_size: "<<after-before<<std::endl;

    if (err != CL_SUCCESS) 
	{
        std::cerr << "Event::wait() failed (" << err << ")\n";
    }

	err = commandqueue.enqueueReadBuffer(key_num_d, true, 0, sizeof(unsigned int), &key_num, 0, 0);
	err = commandqueue.enqueueReadBuffer(key_size_d, true, 0, sizeof(unsigned int), &key_size, 0, 0);
	err = commandqueue.enqueueReadBuffer(value_size_d, true, 0, sizeof(unsigned int), &value_size, 0, 0);

	value_num = key_num;

	if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::enqueueReadBuffer() failed (" << err << ")\n";
		return;
    }
	std::cout<<"The num of keys: "<<key_num<<std::endl;
	std::cout<<"The size of keys: "<<key_size<<std::endl;
	std::cout<<"The size of values: "<<value_size<<std::endl;

	std::cout<<"The number of buckets: "<<rogh->num_buckets<<std::endl;
}

void Scheduler::get_result()
{
	int err;
	cl::Kernel kernel(program, "copy_to_array", &err);
    if (err != CL_SUCCESS) 
	{
        std::cerr << "Kernel::copy_to_array() \"get_size\" failed (" << err << ")\n";
        return;
    }

	cl::Buffer key_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_size, NULL, &err);
	cl::Buffer value_array_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_size, NULL, &err);
	cl::Buffer key_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, key_num*sizeof(unsigned int), NULL, &err);
	cl::Buffer value_index_d = cl::Buffer(context, CL_MEM_READ_WRITE, value_num*sizeof(unsigned int), NULL, &err);

	//set kernel args
	err = kernel.setArg(0, rog);
	err = kernel.setArg(1, key_array_d);
	err = kernel.setArg(2, value_array_d);
	err = kernel.setArg(3, key_index_d);
	err = kernel.setArg(4, value_index_d);
	//enqueuing the kernel to the command queue
	std::cout<<"Running CL program\n";

	clock_t before = clock();
    err = commandqueue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(GLOBAL_THREADS), cl::NDRange(LOCAL_THREADS)
    );

	if (err != CL_SUCCESS) 
	{
       char *msg = print_cl_errstring(err);
	   std::cout<<msg<<std::endl;
		std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return;
    }

    err = commandqueue.finish();
	std::cout<<"command queue finish..."<<std::endl;
	clock_t after = clock();

	std::cout<<"Execution time of get_result: "<<after-before<<std::endl;

    if (err != CL_SUCCESS) 
	{
        std::cerr << "Event::wait() failed (" << err << ")\n";
    }

	char *key_array = (char *)malloc(key_size); 
	char *value_array = (char *)malloc(value_size);
	unsigned int *key_index = (unsigned int *)malloc(sizeof(unsigned int)*key_num);
	unsigned int *value_index = (unsigned int *)malloc(sizeof(unsigned int)*value_num);

	err = commandqueue.enqueueReadBuffer(key_array_d, true, 0, key_size, key_array, 0, 0);
	
	err = commandqueue.enqueueReadBuffer(value_array_d, true, 0, value_size, value_array, 0, 0);
	
	err = commandqueue.enqueueReadBuffer(key_index_d, true, 0, sizeof(unsigned int)*key_num, key_index, 0, 0);
	
	err = commandqueue.enqueueReadBuffer(value_index_d, true, 0, sizeof(unsigned int)*value_num, value_index, 0, 0);

	std::cout<<"The number of keys: "<<key_num<<std::endl;

	/*for(int i = 0; i < key_num; i++)
	{
		std::cout<<((unsigned int *)key_array)[i]<<std::endl;
	}*/

	output.key_index = key_index;
	output.val_index = value_index;
	output.output_keys = key_array;
	output.output_vals = value_array;

	if (err != CL_SUCCESS) 
	{
        std::cerr << "CommandQueue::enqueueReadBuffer() failed (" << err << ")\n";
		return;
    }
}

void Scheduler::do_mapreduce()
{
	/*do the mapreduce*/
	start();
	/*get the size info*/
	get_size();
	/*get the result from the global reduction object*/
	get_result();
}

unsigned int Scheduler::get_key_num()
{
	return key_num;
}

struct output Scheduler::get_output()
{
	return output;
}

void Scheduler::sort_object()
{
	int block_size = 32;
	int global_size = (int)ceil((double)NUM_BUCKETS_G/(double)block_size) * block_size;

	int err = 0;

	cl::Kernel kernel(program, "sort", &err);

	for(unsigned int k = 2; k <= NUM_BUCKETS_G; k*=2)
		for(unsigned int j = k/2; j > 0; j/=2)
		{
			kernel.setArg(0, rog);
			kernel.setArg(1, k);
			kernel.setArg(2, j);

			err = commandqueue.enqueueNDRangeKernel(
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

			err = commandqueue.finish();
		}
}

void Scheduler::destroy()
{
	//clReleaseMemObject(global_data_d);
	//clReleaseMemObject(global_offset_d);
	//clReleaseMemObject(rog);
	//clReleaseKernel(kernel);
}

