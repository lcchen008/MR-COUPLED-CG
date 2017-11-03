CC=g++
LDFLAGS=-lOpenCL
AMD_HEADER=/opt/AMDAPP/include
FRAMEWORK_LIB=./lib
LIBS=/opt/AMDAPP/lib/x86
OBJS=lib/scheduler.o main.o

.PHONY: main

main:$(OBJS)
	$(CC) -o main $(OBJS) 

%.o:%.cpp
	$(CC) -c $< -o $*.o -I $(AMD_HEADER) -L $(LIBS) $(LDFLAGS)

clean:
	-rm main $(OBJS) 
