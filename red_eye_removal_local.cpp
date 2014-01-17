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
#define BLOCKSIZE 16
#define GLOBALSIZE 1024

using namespace cv;
using namespace std;

void PrintEventInfo(cl_event evt)
{
   cl_int error;
   cl_ulong cl_start_time, cl_end_time, queued, submitted;

   error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cl_start_time, NULL);
   OpenCL_CheckError(error, "clGetEventProfilingInfo Error");

   error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cl_end_time, NULL);
   OpenCL_CheckError(error, "clGetEventProfilingInfo Error");

   error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
   OpenCL_CheckError(error, "clGetEventProfilingInfo Error");

   error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitted, NULL);
   OpenCL_CheckError(error, "clGetEventProfilingInfo Error");
   
   printf("submit->queued: %f ms\t", (submitted-queued)*1e-6);
   printf("queued->start: %f ms\t",  (cl_start_time-submitted)*1e-6);
   printf("start->end: %f ms\t",     (cl_end_time-cl_start_time)*1e-6);
    printf("\n");
}

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
    Mat img_r_float, img_g_float, img_b_float;
    Mat temp_r_float, temp_g_float, temp_b_float;
    image_bgr[0].convertTo(img_b_float, CV_32FC1);
    template_bgr[0].convertTo(temp_b_float, CV_32FC1);
    image_bgr[1].convertTo(img_g_float, CV_32FC1);
    template_bgr[1].convertTo(temp_g_float, CV_32FC1);
    image_bgr[2].convertTo(img_r_float, CV_32FC1);
    template_bgr[2].convertTo(temp_r_float, CV_32FC1);

    // compute mean of template_r_float
    //Scalar temp_gray_mean = mean(gray_template_float);
    //float temp_gray_mean_float = temp_gray_mean.val[0];
    Scalar temp_b_mean = mean(temp_b_float);
    float temp_b_mean_float = temp_b_mean.val[0];
    Scalar temp_g_mean = mean(temp_g_float);
    float temp_g_mean_float = temp_g_mean.val[0];
    Scalar temp_r_mean = mean(temp_r_float);
    float temp_r_mean_float = temp_r_mean.val[0];
    
    printf("temp_mean_float b: %f, g: %f, r: %f\n", temp_b_mean_float, temp_g_mean_float, temp_r_mean_float);
    unsigned int img_width = img_r_float.cols;
    unsigned int img_height = img_r_float.rows;
    unsigned int temp_width = temp_r_float.cols;
    unsigned int temp_height = temp_r_float.rows;
    unsigned int output_size = img_width * img_height;
    unsigned int input_size = output_size;
    unsigned int template_size = temp_width * temp_height;
    unsigned int template_half_height = temp_height/2;
    unsigned int template_half_width = temp_width/2;

    // allocate host memory and convert image red channel into 1d array
    printf("input_size: %d, template_size: %d\n", input_size, template_size);
    float* h_r_output = (float*)malloc(input_size * sizeof(float));
    float* h_g_output = (float*)malloc(input_size * sizeof(float));
    float* h_b_output = (float*)malloc(input_size * sizeof(float));
    float* h_r_image = (float*) malloc(input_size * sizeof(float));
    float* h_g_image = (float*) malloc(input_size * sizeof(float));
    float* h_b_image = (float*) malloc(input_size * sizeof(float));
    float *h_r_template = (float*)malloc(template_size * sizeof(float));
    float *h_g_template = (float*)malloc(template_size * sizeof(float));
    float *h_b_template = (float*)malloc(template_size * sizeof(float));
    float *h_response = (float*)malloc(input_size * sizeof(float));
    for (int i = 0; i < temp_r_float.cols; i++)
    {
        for (int j = 0; j < temp_r_float.rows; j++)
        {
            h_r_template[j*temp_r_float.cols+i] = temp_r_float.at<float>(j,i);
            h_b_template[j*temp_r_float.cols+i] = temp_b_float.at<float>(j,i);
            h_g_template[j*temp_r_float.cols+i] = temp_g_float.at<float>(j,i);
        }
    }
    for (int i = 0; i < img_r_float.cols; i++)
    {
        for (int j = 0; j < img_r_float.rows; j++)
        {
            h_r_image[j*img_r_float.cols+i] = img_r_float.at<float>(j,i);
            h_g_image[j*img_r_float.cols+i] = img_g_float.at<float>(j,i);
            h_b_image[j*img_r_float.cols+i] = img_b_float.at<float>(j,i);
        }
    }
    //printf("host memory done\n");
    // opencl intializations
    cl_program clProgram;
    cl_kernel clKernel, clKernel2;
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
    cl_mem d_r_inputImage, d_b_inputImage, d_g_inputImage;
    cl_mem d_r_output;
    cl_mem d_b_output;
    cl_mem d_g_output;
    cl_mem d_r_templateImage;
    cl_mem d_g_templateImage;
    cl_mem d_b_templateImage;
    cl_mem d_response;
    
    // set up device memory
    d_r_inputImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size*sizeof(float), h_r_image, &errcode);
    d_g_inputImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size*sizeof(float), h_g_image, &errcode);
    d_b_inputImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size*sizeof(float), h_b_image, &errcode);
    d_r_templateImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, template_size * sizeof(float), h_r_template, &errcode);
    d_g_templateImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, template_size * sizeof(float), h_g_template, &errcode);
    d_b_templateImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, template_size * sizeof(float), h_b_template, &errcode);
    d_r_output = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &errcode);
    d_g_output = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &errcode);
    d_b_output = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &errcode);
    d_response = clCreateBuffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &errcode);
    //  Create program and kernel
   // Load Program source
   char *source = OpenCL_LoadProgramSource("templatematch_3.cl");
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
    size_t localSize = 48*48*sizeof(float);
    // set Kernels for red channel
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_r_output);
    errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&d_r_inputImage);
    errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_r_templateImage);
    errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&img_height);
    errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&img_width);
    errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void *)&template_half_height);
    errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void *)&temp_height);
    errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void *)&template_half_width);
    errcode = clSetKernelArg(clKernel, 8, sizeof(int), (void *)&temp_width);
    errcode = clSetKernelArg(clKernel, 9, sizeof(int), (void *)&template_size);
    errcode = clSetKernelArg(clKernel, 10, sizeof(float), (void *)&temp_r_mean_float);
    errcode = clSetKernelArg(clKernel, 11, localSize, NULL);
    OpenCL_CheckError(errcode, "clSetKernelArg");

    // Launch OpenCL Kernel
    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = BLOCKSIZE;
    localWorkSize[1] = BLOCKSIZE;
    globalWorkSize[0] = GLOBALSIZE;
    globalWorkSize[1] = GLOBALSIZE;
    
    cl_event event_r;
    errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL,     globalWorkSize, localWorkSize, 0, NULL, &event_r);
    errcode =  clWaitForEvents(1, &event_r);
    OpenCL_CheckError(errcode, "clEnqueueNDRangeKernel");
    printf("Red channel cross correlation:\t");
    PrintEventInfo(event_r);
    // Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQueue, d_r_output, CL_TRUE, 0, output_size*sizeof(float), h_r_output, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");
    // set Kernels for green channel
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_g_output); 
    errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&d_g_inputImage);   
    errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_g_templateImage);
    errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&img_height);
    errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&img_width);
    errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void *)&template_half_height);
    errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void *)&temp_height);
    errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void *)&template_half_width);
    errcode = clSetKernelArg(clKernel, 8, sizeof(int), (void *)&temp_width);
    errcode = clSetKernelArg(clKernel, 9, sizeof(int), (void *)&template_size);
    errcode = clSetKernelArg(clKernel, 10, sizeof(float), (void *)&temp_g_mean_float);
    errcode = clSetKernelArg(clKernel, 11, localSize, NULL);
    OpenCL_CheckError(errcode, "clSetKernelArg");

    // Launch OpenCL Kernel
    cl_event event_g;
    errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL,     globalWorkSize, localWorkSize, 0, NULL, &event_g);
    errcode =  clWaitForEvents(1, &event_g);
    OpenCL_CheckError(errcode, "clEnqueueNDRangeKernel");
    printf("Green channel cross correlation:\t");
    PrintEventInfo(event_g);
    // Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQueue, d_g_output, CL_TRUE, 0, output_size*sizeof(float), h_g_output, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");

// set Kernels for blue channel
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_b_output);
    errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&d_b_inputImage);
    errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_b_templateImage);
    errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&img_height);
    errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&img_width);
    errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void *)&template_half_height);
    errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void *)&temp_height);
    errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void *)&template_half_width);
    errcode = clSetKernelArg(clKernel, 8, sizeof(int), (void *)&temp_width);
    errcode = clSetKernelArg(clKernel, 9, sizeof(int), (void *)&template_size);
    errcode = clSetKernelArg(clKernel, 10, sizeof(float), (void *)&temp_b_mean_float);
    errcode = clSetKernelArg(clKernel, 11, localSize, NULL);
    OpenCL_CheckError(errcode, "clSetKernelArg");

    // Launch OpenCL Kernel
    cl_event event_b;
    errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL,     globalWorkSize, localWorkSize, 0, NULL, &event_b);
    errcode =  clWaitForEvents(1, &event_b);
    OpenCL_CheckError(errcode, "clEnqueueNDRangeKernel");
    printf("Blue channel cross correlation:\t");
    PrintEventInfo(event_b);
    // Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQueue, d_b_output, CL_TRUE, 0, output_size*sizeof(float), h_b_output, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");
    clReleaseKernel(clKernel);
    
    clKernel2 =  clCreateKernel(clProgram, "combined_response", &errcode);
    OpenCL_CheckError(errcode, "clCreateKernel");
    // set Kernels for combined responese
    errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&d_response);
    errcode = clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&d_r_output );
    errcode = clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&d_g_output);
    errcode = clSetKernelArg(clKernel2, 3, sizeof(cl_mem), (void *)&d_b_output);
    errcode = clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&img_height);
    errcode = clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&img_width);
    OpenCL_CheckError(errcode, "clSetKernelArg");
    cl_event event_combine;
    errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel2, 2, NULL,     globalWorkSize, localWorkSize, 0, NULL, &event_combine);
    errcode =  clWaitForEvents(1, &event_combine);
    OpenCL_CheckError(errcode, "clEnqueueNDRangeKernel");
    printf("Combine response:\t");
    PrintEventInfo(event_combine);
    // Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQueue, d_response, CL_TRUE, 0, output_size*sizeof(float), h_response, 0, NULL, NULL);
    OpenCL_CheckError(errcode, "clEnqueueReadBuffer");
//Clean up
    //free(h_output);
    free(h_r_template);
    free(h_b_template);
    free(h_g_template);
    free(h_r_image);
    free(h_g_image);
    free(h_b_image);
    
    clReleaseMemObject(d_r_output);
    clReleaseMemObject(d_r_inputImage);
    clReleaseMemObject(d_r_templateImage);
    clReleaseMemObject(d_g_output);
    clReleaseMemObject(d_g_inputImage);
    clReleaseMemObject(d_g_templateImage);
    clReleaseMemObject(d_b_output);
    clReleaseMemObject(d_b_inputImage);
    clReleaseMemObject(d_b_templateImage);

    clReleaseContext(clContext);
    clReleaseKernel(clKernel2);
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
        if (h_response[i] < 0) unsorted_ints[i] = 0;// (int)(h_reponset[i]*(-1000));//0;
        else unsorted_ints[i] = (int) (h_response[i]*1000000);
    }
    // feed the extra slots with 0 s
    for (int i = output_size; i < sort_size ; i++)
    {
        unsorted_ints[i] = 0;
    }
    // sorting
    static CLRadixSort rs(clContext, device_id, clCommandQueue, sort_size, unsorted_ints);
    //cout << "sorting "<< rs.nkeys <<" keys"<<endl<<endl;
    rs.Sort();
    rs.RecupGPU();
    //cout << rs.histo_time<<" s in the histograms"<<endl;
    //cout << rs.scan_time<<" s in the scanning"<<endl;
    //cout << rs.reorder_time<<" s in the reordering"<<endl;
    //cout << rs.transpose_time<<" s in the transposition"<<endl;
    //cout << rs.sort_time <<" s total GPU time (without memory transfers)"<<endl;
    
    rs.Check();
    rs.CopyResults(sorted_ints, sort_size);
    /*
    for (int i = sort_size-1; i > sort_size-50; i--)
    {
        printf("%d\t", sorted_ints[i]);
    }
    */
    vector<Candidate> candidates = find_match_candidates(sorted_ints, unsorted_ints, sort_size, output_size, gray_img_float.cols, gray_img_float.rows);
    // get the index of the largest cross correlation value
    
    int candidate1, candidate2, candidate1_x, candidate1_y, candidate2_x, candidate2_y;
    candidate1_x = candidates[0].cox;
    candidate1_y = candidates[0].coy;
    candidate2_x = candidates[1].cox;
    candidate2_y = candidates[1].coy;
    /*candidate1 = sorted_ints[sort_size - 1];
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
    */
    // remove redness
    for (int i = candidate1_x - template_half_width; i < candidate1_x + template_half_width; i++)
    {
        for (int j = candidate1_y - template_half_height; j < candidate1_y + template_half_height; j++)
        {
            inputImage.at<Vec3b>(j, i)[2] = (inputImage.at<Vec3b>(j, i)[0] + inputImage.at<Vec3b>(j, i)[1])/2;
        }
    }
    for (int i = candidate2_x - template_half_width; i < candidate2_x + template_half_width; i++)
    {
        for (int j = candidate2_y - template_half_height; j < candidate2_y + template_half_height; j++)
        {
            inputImage.at<Vec3b>(j, i)[2] = (inputImage.at<Vec3b>(j, i)[0] + inputImage.at<Vec3b>(j, i)[1])/2;
        }
    }
    // write
    imwrite("newresult.jpg", inputImage);
    
    // get the coordination
    // clean up
    free(h_r_output);
    free(h_g_output); free(h_b_output); free(h_response);
    free(unsorted_ints);
    free(sorted_ints);
    return 0;
}
