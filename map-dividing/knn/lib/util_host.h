#ifndef UTILHOST
#define UTILHOST

#include <fstream>

class Kernelfile
{
public:
	Kernelfile()
	{
		this->source_ = NULL;
		this->size_ = 0;
	}

	~Kernelfile()
	{
		free(source_);
	}

	bool open(const char *filename)
	{
		FILE *file;
		if(fopen_s(&file, filename, "r")!=0)
			return false;
		fseek(file, 0, SEEK_END);
		size_ = ftell(file) + 1;
		source_ = (char *)malloc(size_);
		fseek(file, 0, SEEK_SET);
		fread_s(source_, size_, 1, size_, file);
		//fread(source_, 1, size_, file);
		source_[size_-1] = '\0';
		fclose(file);
			return true;
	}
	
	char * source()
	{
		return source_;
	}

	int size()
	{
		return size_;
	}

private:
	char * source_;
	int size_;
};
#endif