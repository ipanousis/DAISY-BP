#include "main.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

using namespace kutility;

double getStd(double* observations, int length);
double timeDiff(struct timeval start, struct timeval end);

int main( int argc, char **argv  )
{
  int counter = 1;
  struct timeval startTime,endTime;
  int width, height;
  uchar* srcArray = NULL;

  gettimeofday(&startTime,NULL);

  char* filename = NULL;

  // Get command line options
  if(argc > counter+1 && (!strcmp("-i", argv[counter]) || !strcmp("--image", argv[counter]))){

    filename = argv[++counter];
    load_gray_image (filename, srcArray, height, width);
    printf("HxW=%dx%d\n",height, width);
    counter++;
    
    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));
    daisy_params * daisy = newDaisyParams(srcArray, height, width, 8, 8, 3);

    double start,end,diff;

    time_params times;
    times.measureDeviceHostTransfers = 0;

    initOcl(daisyPrograms,daisyCl);

    daisy->oclPrograms = *daisyPrograms;

    oclDaisy(daisy, daisyCl, &times);

    //printf("Paired Offsets: %d\n",pairedOffsetsLength);
    //printf("Actual Pairs: %d\n",actualPairs);

    string binaryfile = filename;
    binaryfile += ".bdaisy";
    //kutility::save_binary(binaryfile, daisy->descriptors, daisy->paddedHeight * daisy->paddedWidth, daisy->descriptorLength, 1, kutility::TYPE_FLOAT);

    gettimeofday(&endTime,NULL);

    free(daisy->array);

    start = startTime.tv_sec+(startTime.tv_usec/1000000.0);
    end = endTime.tv_sec+(endTime.tv_usec/1000000.0);
    diff = end-start;
    printf("\nMain: %.3fs\n",diff);
  }
  else{
    
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
    sprintf(csvOutName, "gdaisy-speed-tests-%02d%02d-%02d%02d.csv", sysTime->tm_mon+1, sysTime->tm_mday, sysTime->tm_hour, sysTime->tm_min);

    FILE * csvOut = fopen(csvOutName,"w");

    int startWidth = 1024+128;
    //int startHeight = 128;
    int incrementWidth = 128;
    //int incrementHeight = 128;
    int finalWidth = 1536;
    //int finalHeight = 1536;

    // allocate the memory
    unsigned char * array = (unsigned char *)malloc(sizeof(unsigned char) * finalWidth * finalWidth);

    // generate random value input
    for(int i = 0; i < finalWidth * finalWidth; i++)
      array[i] = i % 255;

    fprintf(csvOut,"height,width,convgrad,transA,transB,transBhost,whole,wholestd,dataTransfer,iterations,success\n");

    char* templateRow = "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d\n";

    for(int width = startWidth; width < finalWidth + incrementWidth; width+=incrementWidth){

      printf("%dx%d\n",width,width);

      int iterations = 25;
      int success = 0;
      double * wholeTimes = (double*)malloc(sizeof(double) * iterations);

      time_params times;

      double t_convGrad = 0;
      double t_transA = 0;
      double t_transB = 0;
      double t_transBhost = 0;
      double t_whole = 0;

      times.measureDeviceHostTransfers = 1;

      daisy = newDaisyParams(array, width, width, 8, 8, 3);
      daisy->oclPrograms = *daisyPrograms;

      for(int i = 0; i < iterations; i++){
      
        success |= oclDaisy(daisy, daisyCl, &times);

        t_convGrad += timeDiff(times.startConvGrad, times.endConvGrad);
        t_transA   += timeDiff(times.startTransGrad, times.endTransGrad);
        if(times.measureDeviceHostTransfers)
          t_transBhost += timeDiff(times.startTransDaisy, times.endTransDaisy);
        else
          t_transB += timeDiff(times.startTransDaisy, times.endTransDaisy);

        wholeTimes[i] = timeDiff(times.startFull, times.endFull);
        t_whole += wholeTimes[i];

      }

      t_convGrad    /= iterations;
      t_transA      /= iterations;
      t_transBhost  /= iterations;
      t_transB      /= iterations;
      t_whole       /= iterations;

      double wholeStd = getStd(wholeTimes,iterations);

      fprintf(csvOut, templateRow, width, width, t_convGrad, t_transA, t_transB, t_transBhost, t_whole, wholeStd,
                      times.measureDeviceHostTransfers, iterations, success);

      if(width == finalWidth) break;

      daisy = newDaisyParams(array, width + incrementWidth, width, 8, 8, 3);
      daisy->oclPrograms = *daisyPrograms;

      t_convGrad = 0;
      t_transA = 0;
      t_transB = 0;
      t_transBhost = 0;
      t_whole = 0;

      for(int i = 0; i < iterations; i++){
      
        success |= oclDaisy(daisy, daisyCl, &times);

        t_convGrad += timeDiff(times.startConvGrad, times.endConvGrad);
        t_transA   += timeDiff(times.startTransGrad, times.endTransGrad);
        if(times.measureDeviceHostTransfers)
          t_transBhost += timeDiff(times.startTransDaisy, times.endTransDaisy);
        else
          t_transB += timeDiff(times.startTransDaisy, times.endTransDaisy);

        wholeTimes[i] = timeDiff(times.startFull, times.endFull);
        t_whole += wholeTimes[i];
      }

      t_convGrad    /= iterations;
      t_transA      /= iterations;
      t_transBhost  /= iterations;
      t_transB      /= iterations;
      t_whole       /= iterations;
      
      wholeStd = getStd(wholeTimes,iterations);

      fprintf(csvOut, templateRow, width + incrementWidth, width, t_convGrad, t_transA, t_transB, t_transBhost, t_whole, wholeStd,
                      times.measureDeviceHostTransfers, iterations, success);

      printf("wholestd = %.4f\n",wholeStd);

    }
    
    // print name of output file
    fclose(csvOut);
    printf("Speed test results written to %s.\n", csvOutName);
    free(daisy->descriptors);
    free(array);
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
