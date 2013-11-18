// OpenCV header
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "CLRadixSort.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <vector>

#include <CL/opencl.h>
#include "util.h"

#define BLOCKSIZE 16
#define GLOBALSIZE 1024

using namespace cv;
using namespace std;

cv::Mat grayImageFromFile(const char* filename)
{
    cv::Mat img = cv::imread(filename);
    if (!img.data)
    {
        printf("Could not load image file %s\n", filename);
        exit(1);
    }
    return img;
}


int main(int argc, char* argv[])
{
    // inputs
    const char* input_file = argv[1];
    const char* template_file = argv[2];

    Mat inputImage;
    Mat templateImage;

    // load image
    // inputImage = grayImageFromFile(input_file);
    // templateImage = grayImageFromFile(template_file);
    inputImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    templateImage = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    if ((!inputImage.data) || (!templateImage.data))
    {
        printf("Failed loading image.\n");
        return -1;
    }

    vector<Mat> image_bgr;
    vector<Mat> template_bgr;
    // split channel
    // indexing: 0 - b, 1 - g, 2 - r 
    split(inputImage, image_bgr);
    split(templateImage, template_bgr);

    // convert to float mat
    Mat img_r_float; 
    Mat temp_r_float;
    image_bgr[2].convertTo(img_r_float, CV_32FC1);
    template_bgr[2].convertTo(temp_r_float, CV_32FC1);

    // compute mean of template_r_float
    Scalar temp_r_mean = mean(temp_r_float);
    float temp_r_mean_float = temp_r_mean.val[0];
    printf("temp_r_mean_float: %f\n", temp_r_mean_float);


    unsigned int output_size = img_r_float.rows * img_r_float.cols;
    unsigned int input_size = output_size;
    unsigned int template_size = temp_r_float.rows * temp_r_float.cols;
    unsigned int template_half_height = templateImage.rows/2;
    unsigned int template_half_width = templateImage.cols/2;

    // print out the data array
    /*
    for (int i = 0; i < temp_r_float.cols; i++)
    {
        for (int j = 0; j < temp_r_float.rows; j++)
        {
            printf("%f\t", temp_r_float.at<float>(i,j));
        }
        printf("\n");
    }
    cout << "temp_r_float = "<< endl << " "  << temp_r_float << endl << endl;
    */

    // allocate host memory and convert image red channel into 1d array
    printf("input_size: %d, template_size: %d\n", input_size, template_size);
    float* h_output = (float*)malloc(input_size * sizeof(float));
    float* h_image = (float*) malloc(input_size * sizeof(float));
    float *h_template = (float*)malloc(template_size * sizeof(float));
    
    for (int i = 0; i < temp_r_float.cols; i++)
    {
        for (int j = 0; j < temp_r_float.rows; j++)
        {
            //printf("j:%d,i:%d,temp_r_float.cols:%d\n", j, i, temp_r_float.cols);
            h_template[j*temp_r_float.cols+i] = temp_r_float.at<float>(i,j);
        }
    }
    for (int i = 0; i < img_r_float.cols; i++)
    {
        for (int j = 0; j < img_r_float.rows; j++)
        {
            h_image[j*img_r_float.cols+i] = img_r_float.at<float>(i,j);
        }
    }
    //printf("host memory done\n");
    // opencl intializations
    cl_program clProgram;
    cl_kernel clKernel;
    cl_int errcode;
    cl_platform_id platform;
    cl_context clContext;
    cl_command_queue clCommandQueue;

    /*****************************************/ 
    /* Initialize OpenCL */
    /*****************************************/
    // Check Platform
    errcode = clGetPlatformIDs(1, &platform, NULL);
    if (errcode != CL_SUCCESS)
         FATAL("Failed to find an opencl platform!",errcode);

    // Get the devices
    cl_device_id device_id;
    errcode =clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (errcode != CL_SUCCESS)
        FATAL("Failed to create a device group!",errcode);

    // Create a compute context 
    if(!(clContext = clCreateContext(0, 1, &device_id, NULL, NULL, &errcode)))
        FATAL("Failed to create a compute context!",errcode);
 
    //Create a command-queue
    clCommandQueue = clCreateCommandQueue(clContext, device_id, CL_QUEUE_PROFILING_ENABLE, &errcode);
    OpenCL_CheckError(errcode, "clCreateCommandQueue");

    // Create memory buffers
    cl_mem d_inputImage;
    cl_mem d_output;
    cl_mem d_templateImage;
    
    // set up device memory
    d_inputImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size, h_image, &errcode);

    d_templateImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, template_size, h_template, &errcode);
    d_output = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size, NULL, &errcode);

   //  Create program and kernel
   // Load Program source
   char *source = OpenCL_LoadProgramSource("templatematch.cl");
   if(!source)
      FATAL("Error: Failed to load compute program from file!\n",0);

   // Create the compute program from the source buffer
   if(!(clProgram = clCreateProgramWithSource(clContext, 1, (const char **) & source, NULL, &errcode)))
        FATAL("Failed to create compute program!",errcode);

   // Build the program executable
   errcode = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
   if (errcode != CL_SUCCESS)
   {
        size_t len=2048;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

        printf("%s \n", buffer);
        FATAL("Failed: program build", errcode);
   }

   clKernel = clCreateKernel(clProgram, "template", &errcode);
   OpenCL_CheckError(errcode, "clCreateKernel"); 

    // set Kernels
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_output);
    errcode = errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&d_inputImage);
    errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_templateImage);
    errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&img_r_float.rows);
    errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&img_r_float.cols);
    errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void *)&template_half_height);
    errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void *)&temp_r_float.rows);
    errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void *)&template_half_width);
    errcode = clSetKernelArg(clKernel, 8, sizeof(int), (void *)&temp_r_float.cols);
    errcode = clSetKernelArg(clKernel, 9, sizeof(int), (void *)&template_size);
    errcode = clSetKernelArg(clKernel, 10, sizeof(float), (void *)&temp_r_mean_float);
    OpenCL_CheckError(errcode, "clSetKernelArg");

    // Launch OpenCL Kernel
    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = BLOCKSIZE;
    localWorkSize[1] = BLOCKSIZE;
    globalWorkSize[0] = GLOBALSIZE;
    globalWorkSize[1] = GLOBALSIZE;
    
    cl_event event;
    errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL,     globalWorkSize, localWorkSize, 0, NULL, &event);
    errcode =  clWaitForEvents(1, &event);
    OpenCL_CheckError(errcode, "clEnqueueNDRangeKernel");

    // Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQueue, d_output, CL_TRUE, 0,     output_size, h_output, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");

    // print some values
    for (int i = 0; i < 100; i++)
    {
        printf("%.2f\t", h_output[i]);
    }
    
    // Clean up
    free(h_output);
    free(h_template);
    free(h_image);
    
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_templateImage);

    clReleaseContext(clContext);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQueue);


    return 0;
}
