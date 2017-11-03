#include <iostream>
using namespace std;
#include "lib\scheduler.h"
#include "kmeans.h"
#include "lib\util_host.h"
#include <time.h>

#define GRIDSZ 1000
#define OFFSETS 100

int main()
{
	//Allocate space for all points and clusters. The first K points are cluster centers
	int *points = (int *)malloc(sizeof(float)*DIM*K + sizeof(float)*DIM*BSIZE);
	unsigned int *offsets = (unsigned int *)malloc(sizeof(unsigned int)*BSIZE); 

	int *h_Points = points + DIM*K;
	int *h_means = points;

	cout<<"Generating points..."<<endl;
	clock_t beforegen = clock();
	//Generate data
	srand(2006);
	for(int i = 0; i < BSIZE; i++) {
		offsets[i] = i*DIM + K * DIM;
		for(int j = 0; j < DIM; j++) {
			h_Points[i * DIM + j] = rand() % GRIDSZ;
			//cout<<"Point: "<< h_Points[i * DIM + j]<<" ";
		}
		//cout<<endl;
	}

	cout<<"Generating clusters..."<<endl;
	for(int i = 0; i < K; i++) {
		for(int j = 0; j < DIM; j++) {
			h_means[i * DIM + j] = rand() % GRIDSZ;
		}
	}
	clock_t aftergen = clock();
	cout<<"Data generation time: "<<(aftergen-beforegen)<<" ms"<<endl;

	Scheduler scheduler((void *)h_means, sizeof(int)*DIM*(BSIZE+K), offsets, BSIZE, sizeof(int));
	scheduler.do_mapreduce();

	struct output output = scheduler.get_output();

	int key_num = scheduler.get_key_num();
	int total_num = 0;
	for(int i = 0; i < key_num; i++)
	{
		char *key_address = output.output_keys + (output.key_index)[i];
		char *value_address = output.output_vals + (output.val_index)[i];
		struct kmeans_value value = *(struct kmeans_value *)value_address;
		float number = value.num;
		float dist = value.dist;
		cout<<*(int *)key_address<<": ";
		cout<<"Average point: ("<<value.dim0/number<<", "<<value.dim1/number
			<<", "<<value.dim2/number<<")";
		printf("\t Number of points: %d", (int)number);
		printf("\t Dist: %f", dist);
		total_num += (int)number;
		cout<<endl;
	}
	cout<<"total num of points: "<<total_num<<endl;

	free(points);
	free(offsets);

	int a;
	std::cout<<"Enter any number to continue..."<<std::endl;
	std::cin>>a;
}
