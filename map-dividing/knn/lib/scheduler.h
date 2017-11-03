#include "rol.h"
#include "rog.h"
#include "ds.h"
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <SDKCommon.hpp>
#include <SDKApplication.hpp>
#include <SDKFile.hpp>

class Scheduler
{
	public:
		void *global_data;	//The input data in global memory
		unsigned int global_data_size; //the size of the global data
		void *local_data;	//The input data in local memory
		unsigned int local_data_size;  //the size of the local data

		void *global_data_offset; //Stores the offset information
		unsigned int global_data_offset_number; //number of offsets
		unsigned int unit_size; //the unit_size of offset, used to jump
		unsigned int key_num;
		unsigned int key_size;
		unsigned int value_num;
		unsigned int value_size;
		struct output output;
		bool sort;
		bool usegpu;
		int uselocal; //whether use shared memory: 0: no, 1:yes

		Reduction_Object_G *rogh;


		cl::Buffer global_data_d;
		cl::Buffer local_data_d;
		cl::Buffer shared_local_data_d;
		cl::Buffer global_data_offset_d;
		cl::Buffer rog;
		cl::Buffer key_num_d;
		cl::Buffer key_size_d;
		cl::Buffer value_size_d;
		//cl::Buffer key_start_per_bucket_d; //used to conduct prefix sum
		//cl::Buffer value_start_per_bucket_d; //used to conduct prefix sum
		//cl::Buffer pair_start_per_bucket_d; //used to conduct prefix sum

		cl::Platform platform;
		std::vector<cl::Device> devices;
		cl::Device device;
		cl::Context context;
		cl::CommandQueue cpuqueue;
		cl::CommandQueue gpuqueue;

		cl::CommandQueue commandqueue;
		cl::Program program;
		
		Scheduler(void *global_data, 
		unsigned int global_data_size, 
		//void *local_data,
		//unsigned int local_data_size,
		void *global_data_offset, 
		unsigned int global_data_offset_number, 
		unsigned int unit_size, 
		bool sort,
		bool usegpu,
		int uselocal);

		~Scheduler()
		{
			free(rogh);
			free(output.key_index);
			free(output.output_keys);
			free(output.output_vals);
			free(output.val_index);
		}

		void do_mapreduce();
		unsigned int get_key_num();
		struct output get_output();
		void destroy();
		void sort_object();

	private:
		void start();
		void get_size();
		void get_result();
};

