/************************************************************************
OpenCL Exercise Utility
*************************************************************************/

#define FATAL(msg,err)\
do { \
   fprintf(stderr,"FATAL [%s:%d]:%s Error %d\n", __FILE__, __LINE__, msg,err);   \
   exit(-1); \
} while(0)


void OpenCL_CheckError(int error, char *msg) 
{
  if (error != CL_SUCCESS) {
      FATAL(msg, error); 
  }
}

char *OpenCL_LoadProgramSource(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;
	
    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;
	
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';
	
    return source;
}

int global = 1024, local = 16, intensity = 1;

void OpenCL_ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--global") == 0 || strcmp(argv[i], "-global") == 0) {
            global = atoi(argv[i+1]);
            i = i + 1;
        }
        if (strcmp(argv[i], "--local") == 0 || strcmp(argv[i], "-local") == 0) {
            local = atoi(argv[i+1]);
            i = i + 1;
        }
    }

  //printf("Global size %d, local size %d\n", global, local);
}


void InitArrayFloat(float *Data, int Size)
{
    int i;
    float Scale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < Size; ++i)
    {
        Data[i] = Scale * rand();
    }
}

void InitBinaryMaskArrayPercentile(int *Data, int Size, float percentile, float* actualPercentile)
{
    int i;
    float sum = 0;
    float Scale = 1.0f / (float) RAND_MAX;
    for (i = 0; i < Size; ++i)
    {
    	if (Scale*rand() < percentile)
	{Data[i] = 1;} 
	else
	{Data[i] = 0;}
	sum += Data[i];
    }
    *actualPercentile = sum / (float) Size;
}

void InitBinaryMaskArraySkipN(int *Data, int Size, int step)
{
    int i;
    for (i = 0; i < Size; ++i)
    {
    	if (i%step == 0) Data[i] = 0;
	else Data[i] = 1;
    }
}


void InitArrayInt(int *Data, int Size)
{
    int i;
    for (i = 0; i < Size; ++i)
    {
        Data[i] = rand();
    }
}

int CheckMatchFloat(float *ref, float *data, int len, float max_error)
{
    int comp = 0;
    int error_count = 0;
    for(int i = 0; i < len; ++i) {
        float diff = fabs((float)ref[i] - (float)data[i]);
        comp = (diff > max_error);
        error_count += comp;
    }
    return (error_count == 0) ? 1: 0;
}


int CheckMatchInt(int *ref, int *data, int len)
{
    int error_count = 0;
    for(int i = 0; i < len; ++i) {
        int diff = abs(ref[i] - data[i]);
	if (diff > 0)
           error_count += diff;
    }
    return (error_count == 0) ? 1: 0;
}
