#ifndef WORKERINFO
#define WORKERINFO


struct worker_info
{
	int test_data;
	int has_task;		 //indicates whether the task buffer is empty, immediately after processing it, set this flag to 0
	int task_num;        //the task number assigned by master
	int task_block_size;
};

typedef struct worker_info Worker;

struct tuning_info
{
	int total_num;
	int old_task_block_size;
};

typedef struct tuning_info Tuning;

#endif