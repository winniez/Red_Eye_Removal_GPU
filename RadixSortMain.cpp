#include "CLRadixSort.hpp"

#include<iostream>
#include<fstream>
#include<assert.h>
#include <algorithm>
#include <vector>
#include <time.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */

using namespace std;


int main(void){

	cl_context Context;
	cl_command_queue CommandQueue;   // file de commandes
	cl_platform_id    cpPlatform; // OpenCL platform	
	cl_int status;


	// Check Platform
	status = clGetPlatformIDs(1, &cpPlatform, NULL);
	assert (status == CL_SUCCESS);

	// Get the devices
	cl_device_id device_id;
	status =clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	assert (status == CL_SUCCESS);

	// create context
	Context = clCreateContext(0, 1, &device_id, NULL, NULL, &status);
	assert (status == CL_SUCCESS);

	// create the command queue
	CommandQueue = clCreateCommandQueue(
                                      Context,
                                      device_id,
                                      CL_QUEUE_PROFILING_ENABLE,
                                      &status);
	assert (status == CL_SUCCESS);
	cout <<"OpenCL initializations OK !"<<endl<<endl;
    srand (time(NULL));
    int size = 256;
    int *ori = (int*) malloc(size * sizeof(int));
    int *sorted = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        ori[i] = ((rand()) % 20000); 
    }

	// declaration of a CLRadixSort object
	static CLRadixSort rs(Context, device_id, CommandQueue, size, ori);
	cout << "Radix="<<_RADIX<<endl;
	cout << "Max Int="<<(uint) _MAXINT <<endl;	

	// sort

	// test a non power of two size list
	//rs.Resize((1 << 20) -1);
	//rs.Resize(10);
	
	// cout << rs;
	//
	// cout <<"transpose"<<endl;
	// rs.Transpose();
	//
	// cout << rs;
	//
	// assert(1==2);
	
	cout << "sorting "<< rs.nkeys <<" keys"<<endl<<endl;
	rs.Sort();
	rs.RecupGPU();
	//
	cout << rs.histo_time<<" s in the histograms"<<endl;
	cout << rs.scan_time<<" s in the scanning"<<endl;
	cout << rs.reorder_time<<" s in the reordering"<<endl;
	cout << rs.transpose_time<<" s in the transposition"<<endl;
	//
	cout << rs.sort_time <<" s total GPU time (without memory transfers)"<<endl;
	// check the results (debugging)
	rs.Check();
    rs.CopyResults(sorted, size);

    for (int i=size-1; i>size-20; i--)
    {
        printf("%d\t", sorted[i]);
    }
	return 0;

}
