// OpenCV header
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "CLRadixSort.hpp"
#include "CLRadixSort.cpp"
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
#include "candidate.h"
#define BLOCKSIZE 32
#define GLOBALSIZE 1024

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // inputs
    const char* input_file = argv[1];
    const char* template_file = argv[2];

    Mat inputImage;
    Mat templateImage;

    // load image
    inputImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    templateImage = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    if ((!inputImage.data) || (!templateImage.data))
    {
        printf("Failed loading image.\n");
        return -1;
    }

    // convert to grayscale
    Mat gray_image, gray_template;
    cvtColor( inputImage, gray_image, CV_BGR2GRAY );
    cvtColor( templateImage, gray_template, CV_BGR2GRAY);
    // convert grayscale image to float 
    Mat gray_img_float, gray_template_float;
    gray_image.convertTo(gray_img_float, CV_32FC1);
    gray_template.convertTo(gray_template_float, CV_32FC1);
    /*
    for (int i=0; i<gray_img_float.cols; i++)
    {
        for (int j=0; j<gray_img_float.rows; j++)
        {
            printf("%d,%d,%.2f\t", i,j,gray_img_float.at<float>(i,j));
        }
    }*/

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
    //Scalar temp_r_mean = mean(temp_r_float);
    Scalar temp_gray_mean = mean(gray_template_float);
    float temp_gray_mean_float = temp_gray_mean.val[0];
    //float temp_r_mean_float = temp_r_mean.val[0];
    printf("temp_gray_mean_float: %f\n", temp_gray_mean_float);


    unsigned int output_size = img_r_float.rows * img_r_float.cols;
    unsigned int input_size = output_size;
    unsigned int template_size = temp_r_float.rows * temp_r_float.cols;
    unsigned int template_half_height = templateImage.rows/2;
    unsigned int template_half_width = templateImage.cols/2;

    // allocate host memory and convert image red channel into 1d array
    printf("input_size: %d, template_size: %d\n", input_size, template_size);
    float* h_output = (float*)malloc(input_size * sizeof(float));
    float* h_image = (float*) malloc(input_size * sizeof(float));
    float *h_template = (float*)malloc(template_size * sizeof(float));
    
    for (int i = 0; i < gray_template_float.cols; i++)
    {
        for (int j = 0; j < gray_template_float.rows; j++)
        {
            //printf("j:%d,i:%d,temp_r_float.cols:%d\n", j, i, temp_r_float.cols);
            h_template[j*gray_template_float.cols+i] = gray_template_float.at<float>(i,j);
            //printf("j:%d,i%d,%.2f\t", j,i, h_template[j*gray_template_float.cols+i]);
        }
    }
    for (int i = 0; i < gray_img_float.cols; i++)
    {
        for (int j = 0; j < gray_img_float.rows; j++)
        {
            h_image[j*gray_img_float.cols+i] = gray_img_float.at<float>(i,j);
            //printf("j:%d,i:%d,%.2f\t", j,i, h_image[j*gray_img_float.cols+i]);
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
    d_inputImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size*sizeof(float), h_image, &errcode);

    d_templateImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, template_size * sizeof(float), h_template, &errcode);
    d_output = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &errcode);

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
    size_t localSize = template_size;
    // set Kernels
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_output);
    errcode = errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&d_inputImage);
    errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_templateImage);
    errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&gray_img_float.rows);
    errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&gray_img_float.cols);
    errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void *)&template_half_height);
    errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void *)&gray_template_float.rows);
    errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void *)&template_half_width);
    errcode = clSetKernelArg(clKernel, 8, sizeof(int), (void *)&gray_template_float.cols);
    errcode = clSetKernelArg(clKernel, 9, sizeof(int), (void *)&template_size);
    errcode = clSetKernelArg(clKernel, 10, sizeof(float), (void *)&temp_gray_mean_float);
    //errcode = clSetKernelArg(clKernel, 11, localSize, NULL);
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
    errcode = clEnqueueReadBuffer(clCommandQueue, d_output, CL_TRUE, 0, output_size*sizeof(float), h_output, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");

    // Clean up
    //free(h_output);
    free(h_template);
    free(h_image);
    
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_templateImage);

    clReleaseContext(clContext);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQueue);
    // radix sort
    // Create a compute context
     if(!(clContext = clCreateContext(0, 1, &device_id, NULL, NULL, &errcode)))
         FATAL("Failed to create a compute context!",errcode);
     //Create a command-queue
     clCommandQueue = clCreateCommandQueue(clContext, device_id, CL_QUEUE_PROFILING_ENABLE, &errcode);
     OpenCL_CheckError(errcode, "clCreateCommandQueue");

    // convert float array into int array
    // the size of unsorted list passed into should be multiple of (_GROUPS * _ITEMS)

    int multiple = output_size / (_GROUPS * _ITEMS) + 1;
    int sort_size = multiple * (_GROUPS * _ITEMS);
    int *unsorted_ints = (int*)malloc(sort_size * sizeof(int));
    int *sorted_ints = (int*)malloc(sort_size * sizeof(int));
    for (int i = 0; i < output_size; i++)
    {
        if (h_output[i] < 0) unsorted_ints[i] = 0;
        else unsorted_ints[i] = (int) (h_output[i]*1000);
    }
    // feed the extra slots with 0 s
    for (int i = output_size; i < sort_size ; i++)
    {
        unsorted_ints[i] = 0;
    }
    // sorting
    static CLRadixSort rs(clContext, device_id, clCommandQueue, sort_size, unsorted_ints);
    cout << "sorting "<< rs.nkeys <<" keys"<<endl<<endl;
    rs.Sort();
    rs.RecupGPU();
    cout << rs.histo_time<<" s in the histograms"<<endl;
    cout << rs.scan_time<<" s in the scanning"<<endl;
    cout << rs.reorder_time<<" s in the reordering"<<endl;
    cout << rs.transpose_time<<" s in the transposition"<<endl;
    cout << rs.sort_time <<" s total GPU time (without memory transfers)"<<endl;
    
    rs.Check();
    rs.CopyResults(sorted_ints, sort_size);
    for (int i = sort_size-1; i > sort_size-50; i--)
    {
        printf("%d\t", sorted_ints[i]);
    }
   
    find_match_candidates(sorted_ints, unsorted_ints, sort_size, output_size, gray_img_float.cols, gray_img_float.rows);
    // get the index of the largest cross correlation value
    /*
    int candidate1, candidate2, candidate1_x, candidate1_y, candidate2_x, candidate2_y;
    candidate1 = sorted_ints[sort_size - 1];
    for (int i = 0; i < output_size; i++)
    {
        if (unsorted_ints[i] == candidate1)
        {
            candidate1_x = i % gray_img_float.cols;
            candidate1_y = (int) (i / gray_img_float.cols);
        }
    }
    printf("\ncandidate 1: %d, at %d, %d\n", candidate1, candidate1_x, candidate1_y);
    int top = sort_size - 2;
    while (sort_size > 0)
    {
        candidate2 = sorted_ints[top];
        for (int i = 0; i < output_size; i++)
        {
            if (unsorted_ints[i] == candidate2)
            {
                candidate2_x = i % gray_img_float.cols;
                candidate2_y = (int) (i / gray_img_float.cols);
            }
        }
        if (abs(candidate2_x - candidate1_x) + 
            abs(candidate2_y - candidate1_y) > 20)
        {
            break;
        }

        top--;
    }
    printf("candidate 2: %d, at %d, %d\n", candidate2, candidate2_x, candidate2_y);
    
    // remove redness
    for (int i = candidate1_x - template_half_width; i < candidate1_x + template_half_width; i++)
    {
        for (int j = candidate1_y - template_half_height; j < candidate1_y + template_half_height; j++)
        {
            inputImage.at<Vec3b>(i, j)[2] = (inputImage.at<Vec3b>(i, j)[0] + inputImage.at<Vec3b>(i, j)[1])/2;
        }
    }
    for (int i = candidate2_x - template_half_width; i < candidate2_x + template_half_width; i++)
    {
        for (int j = candidate2_y - template_half_height; j < candidate2_y + template_half_height; j++)
        {
            inputImage.at<Vec3b>(i, j)[2] = (inputImage.at<Vec3b>(i, j)[0] + inputImage.at<Vec3b>(i, j)[1])/2;
        }
    }
    // write
    imwrite("newresult.jpg", inputImage);
    */
    // get the coordination
    // clean up
    free(h_output);
    free(unsorted_ints);
    free(sorted_ints);
    return 0;
}
