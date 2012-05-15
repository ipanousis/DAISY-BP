/*

  File: main.cpp

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : February/2012

*/

#include "main.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

using namespace kutility;

double getStd(double* observations, int length);
double timeDiff(struct timeval start, struct timeval end);
void displayTimes(daisy_params * daisy,time_params * times);

int main( int argc, char **argv  )
{
  struct timeval startTime,endTime;

  char* filename = NULL;
  uchar* srcArray = NULL;
  int width, height;

  int counter = 1;

  gettimeofday(&startTime,NULL);

  // Get command line options
  if(argc > counter+1 && (!strcmp("-i", argv[counter]) || !strcmp("--image", argv[counter]))){

    filename = argv[++counter];
    load_gray_image (filename, srcArray, height, width);
    counter++;
    
    if(height * width > 2048 * 2048){
      fprintf(stderr, "Apologies but this implementation is not yet ready to\
                       accept larger image sizes than 2048*2048.\n\
                       If you must process them, try to split your images in blocks or otherwise feel free to implement DAISY computation in parts :)");
      return 1;
    }

    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));

    daisy_params * daisy = newDaisyParams(srcArray, height, width, NO_GRADIENTS, REGION_PETALS_NO, SMOOTHINGS_NO);

    double start,end,diff;

    time_params times;
    times.measureDeviceHostTransfers = 1;
    times.transPinned = 0;
    times.transRam = 0;

    initOcl(daisyPrograms,daisyCl);

    daisy->oclPrograms = *daisyPrograms;

    oclDaisy(daisy, daisyCl, &times);

<<<<<<< HEAD
<<<<<<< HEAD
    //printf("Paired Offsets: %d\n",pairedOffsetsLength);
    //printf("Actual Pairs: %d\n",actualPairs);

    displayTimes(daisy,&times);

    string binaryfile = filename;
    binaryfile += ".bdaisy";
//    kutility::save_binary(binaryfile, daisy->descriptors, daisy->paddedHeight * daisy->paddedWidth, daisy->descriptorLength, 1, kutility::TYPE_FLOAT);
=======
    string binaryfile = filename;
=======
    string binaryfile = filename;
>>>>>>> 21b92db8a0cc84bfb791c594ad2e43976ecf9a97
    binaryfile += ".bgdaisy";

    printf("Saving binary as %s...\n",filename);
    unpadDescriptorArray(daisy);
    kutility::save_binary(binaryfile, daisy->descriptors, daisy->height * daisy->width, daisy->descriptorLength, 1, kutility::TYPE_FLOAT);
<<<<<<< HEAD
>>>>>>> 21b92db8a0cc84bfb791c594ad2e43976ecf9a97
=======
>>>>>>> 21b92db8a0cc84bfb791c594ad2e43976ecf9a97

    gettimeofday(&endTime,NULL);

    free(daisy->array);
  }
  else if(argc > counter+1 && !strcmp("-profile", argv[counter])){
    
    // do the profiling across a range of inputs from 128x128 to 1536x1536

    // initialise all the opencl stuff first outside loop
    daisy_params * daisy;
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));

    ocl_constructs * daisyCl = newOclConstructs(0,0,0);

    initOcl(daisyPrograms,daisyCl);

    // initialise loop variables, input range numbers etc..
    struct tm * sysTime = NULL;                     

    time_t timeVal = 0;                            
    timeVal = time(NULL);                          
    sysTime = localtime(&timeVal);

    char * csvOutName = (char*)malloc(sizeof(char) * 500);
    sprintf(csvOutName, "gdaisy-speed-tests-interop1-%02d%02d-%02d%02d.csv", sysTime->tm_mon+1, sysTime->tm_mday, sysTime->tm_hour, sysTime->tm_min);

    FILE * csvOut = fopen(csvOutName,"w");

    /* Standard ranges QVGA,VGA,SVGA,XGA,SXGA,SXGA+,UXGA,QXGA*/
    //int heights[8] = {320,640,800,1024,1280,1400,1600,2048};
    //int widths[8] = {240,480,600,768,1024,1050,1200,1536};
    //int total = 8;

    /* Without transfer ranges */
    //int heights[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    //int widths[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    //int total = 12;
//#define BLOCK_SIZE 128
//    int heights[8] = {BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE};
//    int widths[8] = {BLOCK_SIZE,BLOCK_SIZE*2,BLOCK_SIZE*3,BLOCK_SIZE*4,BLOCK_SIZE*5,BLOCK_SIZE*6,BLOCK_SIZE*7,BLOCK_SIZE*8};
//    int total = 8;    

    /* With transfer ranges */
/*    int heights[4] = {1152,1280,1408,1536};//{128,256,384,512};//,,1152,1280,1408,1536};
    int widths[4] = {1152,1280,1408,1536};//,640,768,896,1024,1152,1280,1408,1536};
    int total = 4;//12;*/
    int heights[1] = {1024};
    int widths[1] = {1024};
    int total = 1;
    // allocate the memory
    unsigned char * array = (unsigned char *)malloc(sizeof(unsigned char) * heights[total-1] * widths[total-1]);

    // generate random value input
    for(int i = 0; i < heights[total-1]*widths[total-1]; i++)
      array[i] = i % 255;

    fprintf(csvOut,"height,width,grad,convonly,convmiddlex,convgrad,transA,transB,transBhost,transBtopinned,transBtoram,whole,wholestd,dataTransfer,iterations,success\n");

    char* templateRow = "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d\n";

    for(int w = 0; w < total; w++){

      int width = widths[w];
      int height = heights[w];

      printf("%dx%d\n",height,width);

<<<<<<< HEAD
<<<<<<< HEAD
      int iterations = 10;
=======
      int iterations = 15;
>>>>>>> 21b92db8a0cc84bfb791c594ad2e43976ecf9a97
=======
      int iterations = 15;
>>>>>>> 21b92db8a0cc84bfb791c594ad2e43976ecf9a97
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

      times.measureDeviceHostTransfers = 0;

      daisy = newDaisyParams(array, height, width, NO_GRADIENTS, REGION_PETALS_NO, SMOOTHINGS_NO);
      daisy->oclPrograms = *daisyPrograms;

      for(int i = 0; i < iterations; i++){

        times.transPinned = 0;
        times.transRam = 0;
      
        success |= oclDaisy(daisy, daisyCl, &times);

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

//"height,width,grad,convonly,convmiddlex,convgrad,transA,transB,transBhost,transBtopinned,transBtoram,whole,wholestd,dataTransfer,iterations,success\n"
      fprintf(csvOut, templateRow, height, width, t_grad, t_conv, t_convx, t_convGrad, t_transA, t_transB, t_transBhost, t_transPinned, t_transRam, t_whole, wholeStd,
                      times.measureDeviceHostTransfers, iterations, success);

    }
    
    // print name of output file
    fclose(csvOut);
    printf("Speed test results written to %s.\n", csvOutName);
//    free(daisy->descriptors);
    free(array);
  }
  else{
    fprintf(stderr,"Pass image filename with argument -i <file>, or profile with -profile\n");
    return 1;
  }

  return 0;
}

double timeDiff(struct timeval start, struct timeval end){

  return (end.tv_sec+(end.tv_usec/1000000.0)) - (start.tv_sec+(start.tv_usec/1000000.0));

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

void displayTimes(daisy_params * daisy, time_params * times){

  printf("grad: %.4f sec\n",timeDiff(times->startGrad,times->endGrad));
  printf("conv: %.4f sec\n",timeDiff(times->startConv,times->endConv));
  printf("convx: %.4f sec\n",timeDiff(times->startConvX,times->endConvX));
  printf("transPinned: %.4f sec\n",times->transPinned);
  printf("transRam: %.4f sec\n",times->transRam);
  printf("daisyFull: %.4f sec\n",timeDiff(times->startFull,times->endFull));

}
