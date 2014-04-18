#Red Eye Removal on GPU

##Overview
The project is proposed to correct red eye effect in a photograph.

The first phase is to match a red-eye template to an input photograph and sort the template responses to find eye regions. The template matching and sorting are computed on GPU.

The second phase is to remove some of the redness from each eye pixel to correct the red-eye effect.

#Dependency
OpenCV library and OpenCL platform.

##Run the program: 
./red_eye_removal red_eye_effect_5.jpg red_eye_effect_template_5.jpg

##Documentation
The bottleneck for GPU computing is copying data from host to GPU. For 2D template matching problem, each thread requires a datablock to do the computation.

We first implement a global memory version. In this version, each thread brings in data it needs. (CPU code: red_eye_removal_global.cpp, GPU code: templatematch.cl)

In template matching problem, neighboring threads (threads in a same block) do share data. In this case, we optimize using GPU local memory (memory shared within a block of threads, OpenCL local, CUAD shared). Each thread brings in part of data the entire block requires. The running time is cut of by 50% compared to the global memory version. (CPU code: red_eye_removal_local.cpp, GPU code: templatematch_3.cl) 

##Reference
http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/GpuComputing/master/IPython/RedEyeRemoval.ipynb
