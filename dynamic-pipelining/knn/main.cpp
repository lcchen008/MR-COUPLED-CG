#include <iostream>
using namespace std;
#include "lib\scheduler.h"
#include "kmeans.h"
#include "lib\util_host.h"
#include <time.h>
#include "nbc.h"

#define GRIDSZ 1000
#define OFFSETS 100

int main()
{
	int *points = (int *)malloc(sizeof(float)*(DIM+1) + sizeof(float)*(DIM+1)*BSIZE);
	unsigned int *offsets = (unsigned int *)malloc(sizeof(unsigned int)*BSIZE); 

	//initialize the test sample
	points[0] = -1;
	points[1] = 100;
	points[2] = 100;
	points[3] = 100;

	int *h_Points = points + DIM + 1;

	cout<<"Generating points..."<<endl;
	clock_t beforegen = clock();
	//Generate data
	srand(2006);
	for(int i = 0; i < BSIZE; i++) {
		offsets[i] = (i + 1)*(DIM + 1);
		h_Points[i*(DIM+1)] = i;
		for(int j = 1; j < DIM + 1; j++) {
			h_Points[i * (DIM + 1) + j] = rand() % GRIDSZ;
		}
	}
	clock_t aftergen = clock();

	cout<<"Data generation time: "<<aftergen - beforegen<<endl;

	std::cout<<"Data loaded..."<<std::endl;

	//Scheduler scheduler((void *)points, sizeof(float) * (DIM + 1) * (BSIZE + 1), /*NULL, 0, */&offsets[0], BSIZE, sizeof(int), false, false, 1);

	Scheduler scheduler((void *)points, sizeof(float) * (DIM + 1) * (BSIZE + 1), &offsets[0], BSIZE, sizeof(int));

	scheduler.do_mapreduce();

	struct output output = scheduler.get_output();

	int key_num = scheduler.get_key_num();

	char *output_keys = output.output_keys;
	char *output_vals = output.output_vals;
	unsigned int *key_index = output.key_index;
	unsigned int *val_index = output.val_index;

	cout<<"****************************************"<<endl;

	for(int i = 0; i < key_num; i++)
	{
			char *key_address = output_keys + key_index[i];
            char *val_address = output_vals + val_index[i];

			int key = *(int *)key_address;
			float val = *(float *)val_address;

			cout<<key<<": "<<val<<endl;
	}

	cout<<"****************************************"<<endl;

	
	delete[] points;
	delete[] offsets;

	int b;
	std::cout<<"Enter any number to continue..."<<std::endl;
	std::cin>>b;
}