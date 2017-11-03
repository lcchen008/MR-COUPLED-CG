#include <iostream>
using namespace std;
#include "lib\scheduler.h"
#include "lib\util_host.h"
#include <time.h>
#include "nbc.h"

#define GRIDSZ 1000
#define OFFSETS 100

int main()
{
	char color[] = "Yellow";
	char type[] = "SUV";
	char origin[] = "GM";
	char transmission[] = "Auto";
	char stolen[] = "No";
	
	char filename[] = "dataset";

	FILE *file;
	file = fopen(filename, "r");
	fseek(file, 0, SEEK_END);
	int nLen = ftell(file);
	rewind(file);
	char *filebuf = new char[nLen];
	nLen = fread(filebuf, sizeof(char), nLen, file);
	fclose(file);

	//filebuf[nLen] = '\0';

	char *input = new char[5 + nLen];
	input[0] = get_color(color);
	input[1] = get_type(type);
	input[2] = get_origin(origin);
	input[3] = get_transmission(transmission);
	input[4] = get_stolen(stolen);

	for(int i = 0; i<5; i++)
	cout<<(int)input[i]<<endl;

	char *data = input+5;
	memcpy(data, filebuf, nLen);

	vector<unsigned int> offsets;
	offsets.push_back(5);

	for(int i = 0; i<nLen; i++)
	{
		char tmp =data[i];
		if(tmp=='\t')
			data[i] = '\0';
		if(tmp=='\n')
		{
			data[i] = '\0';
			if(i+1<nLen)
			offsets.push_back(i+1+5);
		}
	}

	cout<<"last: "<<&input[offsets[offsets.size()-1]]<<endl;


	/*for(int i = 0; i < offsets.size(); i++)
	{
		cout<<offsets[i]<<": "<<(char *)&filebuf[offsets[i]]<<endl;
	}*/

	cout<<"offset number: "<<offsets.size()<<endl;

	std::cout<<"Data loaded..."<<std::endl;
	Scheduler scheduler((void *)input, 5 + nLen, /*NULL, 0, */&offsets[0], offsets.size(), sizeof(int));
	scheduler.do_mapreduce();

	struct output output = scheduler.get_output();

	int key_num = scheduler.get_key_num();

	char *output_keys = output.output_keys;
	char *output_vals = output.output_vals;
	unsigned int *key_index = output.key_index;
	unsigned int *val_index = output.val_index;
	int total = 0;

	cout<<"****************************************"<<endl;

	for(int i = 0; i < key_num; i++)
	{
			char *key_address = output_keys + key_index[i];
            char *val_address = output_vals + val_index[i];

			int key = *(int *)key_address;
			int val = *(int *)val_address;
			total += val;
			cout<<key<<": "<<val<<endl;
	}

	cout<<"****************************************"<<endl;

	int numbers[30];
	for(int i = 0; i<30; i++)
	numbers[i] = 0;
	for(int i = 0; i<key_num; i++)
	{
		char *key_address = output_keys + key_index[i];
		char *val_address = output_vals + val_index[i];

		int key = *(int *)key_address;
		int val = *(int *)val_address;
		numbers[key] = val;
	}

	double p_age[5];
	for(int h = 0; h<5; h++)
	p_age[h]=(double)numbers[25+h]/offsets.size();

	double p[25];

	for(int k = 0; k<5; k++)
	{
		for(int x = 0; x<5; x++)
		{
			p[k*5 + x] = 0;
			if(numbers[25+k]!=0)
			p[k*5 + x] = (double)numbers[k*5 + x]/(double)numbers[25+k];
		}
	}

	double ps[5];

	for(int y = 0; y<5; y++)
	{
		ps[y] = 1;
		for(int z = 0; z<5;z++)
		ps[y]*=p[y*5+z];
		ps[y]*=p_age[y];
	}

	double max_p = ps[0];
	unsigned int max_index = 0;

	for(int l = 1; l<5; l++)
	{
		if(ps[l]>max_p)
		{
			max_p = ps[l];
			max_index = l;
		}
	}

	cout<<"The age of the car is: "<<max_index+1<<endl;
	cout<<"And the max possibility is: "<<max_p<<endl;
	cout<<"Total is: "<<total<<endl;

	for(int n = 0; n < 5; n++)
	cout<<ps[n]<<endl;

	delete[] filebuf;
	delete[] input;

	int b;
	std::cout<<"Enter any number to continue..."<<std::endl;
	std::cin>>b;
}