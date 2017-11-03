#include <iostream>
using namespace std;
#include "lib\scheduler.h"
#include "lib\util_host.h"
#include <time.h>

#define GRIDSZ 1000
#define OFFSETS 100

int main()
{
	char filename[] = "..\\data2000";

	FILE *file;
	file = fopen(filename, "r");
	fseek(file, 0, SEEK_END);
	int nLen = ftell(file);
	rewind(file);
	char *filebuf = new char[nLen + 1];
	nLen = fread(filebuf, sizeof(char), nLen, file);
	fclose(file);

	filebuf[nLen] = '\0';

	vector<unsigned int> offsets;
	unsigned int offset = 0;

	FILE *fp = fopen(filename, "r");

	bool in_word = false;
	char ch;

	do
	{
		ch = fgetc(fp);	
		if(ch==EOF)
			break;
		if(ch>='a'&&ch<='z'
			||ch>='0'&&ch<='9'
			||ch>='A'&&ch<='Z')
		{
			if(!in_word)
			{
				in_word = true;
				offsets.push_back(offset);
			}
		}

		else
		{
			if(in_word)
			{
				in_word = false;
				filebuf[offset] = '\0';
			}
		}
		offset++;
	}while(ch!=EOF);

	fclose(fp);

	cout<<"offset number: "<<offsets.size()<<endl;
	Scheduler scheduler((void *)filebuf, nLen + 1, &offsets[0], offsets.size(), sizeof(int));

	scheduler.do_mapreduce();

	struct output output = scheduler.get_output();

	int key_num = scheduler.get_key_num();
	int total_num = 0;
	for(int i = 0; i < key_num; i++)
	{
			char *key_address = output.output_keys + output.key_index[i];
            char *val_address = output.output_vals + output.val_index[i];
			int number = *(int *)val_address;
			//cout<<key_address<<": "<<number<<endl;
			total_num+=number;
	}
	cout<<"total num of words: "<<total_num<<endl;

	delete[] filebuf;

	int a;
	std::cout<<"Enter any number to continue..."<<std::endl;
	std::cin>>a;
}