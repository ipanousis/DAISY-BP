/*

  File: main.cpp

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : February/2012

*/

#include "main.h"
#include <stdio.h>
#include <sys/time.h>

using namespace kutility;

double getStd(double* observations, int length);
double timeDiff(struct timeval start, struct timeval end);
void displayTimes(daisy_params * daisy,time_params * times);
void writeInfofile(daisy_params * daisy, char * binaryfile);
void profileSpeed(short int cpuTransfer);
void runDaisy(char * filename, short int saveBinary);
void runMatcher();

int main( int argc, char **argv  )
{

  char* filename = NULL;

  short int saveBinary = 0;
  int counter = 1;

  // Get command line options
  if(argc > counter+1 && (!strcmp("-i", argv[counter]))){

    filename = argv[++counter];

    counter++;

    saveBinary = (argc > counter && !strcmp("-save",argv[counter]));


    runDaisy(filename,saveBinary);

  }
  else if(argc > counter && !strcmp("-profile", argv[counter])){

    // do the profiling across a range of inputs from 128x128 to 1536x1536

    counter++;

    saveBinary = (argc > counter && !strcmp("-save",argv[counter]));

    profileSpeed(saveBinary);

  }
  else if(argc > counter && !strcmp("-match", argv[counter])){

    /* do matching */
    counter++;

    runMatcher();

  }
  else{
    fprintf(stderr,"Pass image filename with argument -i <file>, or profile with -profile\n");
    return 1;
  }

  return 0;
}

void runMatcher(){

  time_params * times = (time_params*) malloc(sizeof(time_params));
  times->measureDeviceHostTransfers = 0;
  times->transPinned = 0;
  times->transRam = 0;
  times->displayRuntimes = 1;

  ocl_constructs * daisyCl = newOclConstructs(0,0,0);

  daisy_params * daisyTemplate = initDaisy("test-data/template.jpg",0);
  daisy_params * daisyTarget = initDaisy("test-data/frames/frame-0570s.png",0);

  initOcl(daisyTemplate, daisyCl);
  initOclMatch(daisyTemplate,daisyCl);

  daisyTarget->oclKernels = daisyTemplate->oclKernels;

  oclDaisy(daisyTemplate, daisyCl, times);
  oclDaisy(daisyTarget, daisyCl, times);

  oclMatchDaisy(daisyTemplate, daisyTarget, daisyCl, times);

  daisyCleanUp(daisyTemplate,daisyCl);
  daisyCleanUp(daisyTarget,daisyCl);

}

void runDaisy(char * filename, short int saveBinary){

  time_params times;
  times.measureDeviceHostTransfers = saveBinary;
  times.transPinned = 0;
  times.transRam = 0;
  times.displayRuntimes = 1;

  ocl_constructs * daisyCl = newOclConstructs(0,0,0);

  daisy_params * daisy = initDaisy(filename,saveBinary);

  initOcl(daisy,daisyCl);

  oclDaisy(daisy, daisyCl, &times);

  if(times.displayRuntimes)
    displayTimes(daisy,&times);

  if(saveBinary)
    saveToBinary(daisy);

  daisyCleanUp(daisy,daisyCl);
  
}

void profileSpeed(short int cpuTransfer){

    // initialise loop variables, input range numbers etc..
    struct tm * sysTime = NULL;                     

    time_t timeVal = 0;                            
    timeVal = time(NULL);                          
    sysTime = localtime(&timeVal);

    char * csvOutName = (char*)malloc(sizeof(char) * 200);
    char * nameTemplate = (char*)malloc(sizeof(char) * 100);
    if(cpuTransfer)
      strcpy(nameTemplate,"gdaisy-speeds-FAST-STANDARD-TRANSFER-%02d%02d-%02d%02d.csv");
    else
      strcpy(nameTemplate,"gdaisy-speeds-FAST-STANDARD-NOTRANSFER-%02d%02d-%02d%02d.csv");

    sprintf(csvOutName, nameTemplate, sysTime->tm_mon+1, sysTime->tm_mday, sysTime->tm_hour, sysTime->tm_min);

    FILE * csvOut = fopen(csvOutName,"w");

    /* Standard ranges QVGA,VGA,SVGA,XGA,SXGA,SXGA+,UXGA,QXGA*/
    int heights[8] = {320,640,800,1024,1280,1400,1600,2048};
    int widths[8] = {240,480,600,768,1024,1050,1200,1536};
    int total = 8;

    /* Without transfer ranges */
    //int heights[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    //int widths[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    //int total = 12;

    //#define BLOCK_SIZE 128
    //    int heights[8] = {BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE};
    //    int widths[8] = {BLOCK_SIZE,BLOCK_SIZE*2,BLOCK_SIZE*3,BLOCK_SIZE*4,BLOCK_SIZE*5,BLOCK_SIZE*6,BLOCK_SIZE*7,BLOCK_SIZE*8};
    //    int total = 8;    

    /* With transfer ranges */
    /*int heights[4] = {1152,1280,1408,1536};//{128,256,384,512};//,,1152,1280,1408,1536};
    int widths[4] = {1152,1280,1408,1536};//,640,768,896,1024,1152,1280,1408,1536};
    int total = 4;//12;*/

    // allocate the memory
    unsigned char * array = (unsigned char *)malloc(sizeof(unsigned char) * heights[total-1] * widths[total-1]);

    // generate random value input
    for(int i = 0; i < heights[total-1]*widths[total-1]; i++)
      array[i] = i % 255;

    fprintf(csvOut,"height,width,grad,convonly,convmiddlex,convgrad,\
transA,transB,transBhost,transBtopinned,transBtoram,\
whole,wholestd,dataTransfer,iterations,success\n");

    const char * templateRow = "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d\n";

    int height = 0;
    int width = 0;

    // initialise daisy and opencl
    daisy_params * daisy = newDaisyParams("", array, height, width, cpuTransfer);
    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_kernels * daisyKernels = daisy->oclKernels;

    initOcl(daisy,daisyCl);

    for(int w = 0; w < total; w++){

      int width = widths[w];
      int height = heights[w];

      printf("%dx%d\n",height,width);

      int iterations = 10;
      int success = 0;
      double * wholeTimes = (double*)malloc(sizeof(double) * iterations);

      time_params times;

      double t_grad = 0;
      double t_conv = 0;
      double t_convx = 0;
      double t_convGrad = 0;
      double t_transA = 0;
      double t_transB = 0;
      double t_transBhost = 0;
      double t_transPinned = 0;
      double t_transRam = 0;
      double t_whole = 0;

      times.measureDeviceHostTransfers = daisy->cpuTransfer;

      daisy = newDaisyParams("", array,height,width,daisy->cpuTransfer);
      daisy->oclKernels = daisyKernels;

      for(int i = 0; i < iterations; i++){

        times.transPinned = 0;
        times.transRam = 0;
      
        success |= oclDaisy(daisy, daisyCl, &times);
        daisyCleanUp(daisy, daisyCl);

        t_grad += timeDiff(times.startGrad, times.endGrad);
        t_conv += timeDiff(times.startConv, times.endConv);
        t_convx += timeDiff(times.startConvX, times.endConvX);
        t_convGrad += timeDiff(times.startConvGrad, times.endConvGrad);
        t_transA   += timeDiff(times.startTransGrad, times.endTransGrad);
        if(times.measureDeviceHostTransfers)
          t_transBhost += timeDiff(times.startTransDaisy, times.endTransDaisy);
        else
          t_transB += timeDiff(times.startTransDaisy, times.endTransDaisy);

        t_transPinned += times.transPinned;
        t_transRam += times.transRam;

        wholeTimes[i] = timeDiff(times.startFull, times.endFull);
        t_whole += wholeTimes[i];
        displayTimes(daisy,&times);
      }

      t_grad /= iterations;
      t_conv /= iterations;
      t_convx /= iterations;
      t_convGrad    /= iterations;
      t_transA      /= iterations;
      t_transBhost  /= iterations;
      t_transB      /= iterations;
      t_transPinned /= iterations;
      t_transRam /= iterations;
      t_whole       /= iterations;

      double wholeStd = getStd(wholeTimes,iterations);

      /*"height,width,grad,convonly,convmiddlex,convgrad,
         transA,transB,transBhost,transBtopinned,transBtoram,
         whole,wholestd,dataTransfer,iterations,success\n"*/
      fprintf(csvOut, templateRow, height, width, t_grad, t_conv, t_convx, t_convGrad, 
                  t_transA, t_transB, t_transBhost, t_transPinned, t_transRam, 
                  t_whole, wholeStd, times.measureDeviceHostTransfers, iterations, success);

    }
    
    // print name of output file
    fclose(csvOut);
    printf("Speed test results written to %s.\n", csvOutName);
    free(array);

}

double getStd(double * observations, int length){

  double mean = .0f;
  for(int i = 0; i < length; i++)
    mean += observations[i];
  mean /= length;
  double stdSum = .0f;
  for(int i = 0; i < length; i++)
    stdSum += pow(observations[i] - mean,2);

  return sqrt(stdSum / length);

}
