/*This is the struct containing the buffer info, used
for scheduling the reduce tasks*/

#include "configuration.h"

#ifndef BUFFERINFO
#define BUFFERINFO

struct bufferinfo
{
	int full;						//tells whether the buffer is full
	int scheduled;
	int num;
};

typedef struct bufferinfo Bufferinfo;


#endif