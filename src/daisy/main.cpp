#include "main.h"
#include <stdio.h>
#include <sys/time.h>

using namespace kutility;

int main( int argc, char **argv  )
{
  int counter = 1;
  struct timeval startTime,endTime;
  int width, height;
  uchar* srcArray = NULL;
  //uchar* othArray = NULL;
  gettimeofday(&startTime,NULL);
  /*
  double opy = -1;
  double opx = -1;
  int opo =  0;

  int rad   = 15;
  int radq  =  3;
  int thq   =  8;
  int histq =  8;

  //int nrm_type = NRM_PARTIAL;

  int orientation_resolution = 18;*/

  char* filename = NULL;

  // Get command line options
  if(argc > counter+1 && (!strcmp("-i", argv[counter]) || !strcmp("--image", argv[counter])))
  {
    filename = argv[++counter];
    // im = load_byte_image(filename,w,h);
    load_gray_image (filename, srcArray, height, width);
    //load_gray_image (filename, othArray, height, width);
    printf("HxW=%dx%d\n",height, width);
    counter++;
  }
  
  ocl_constructs * daisyCl = newOclConstructs(0,0,0);
  //ocl_constructs * daisyOcl = newOclConstructs(0,0,0);
  daisy_params * daisy = newDaisyParams(srcArray, height, width, 8, 8, 3);

  double start,end,diff;

  //initOcl(daisy, daisyCl);
  //initOcl(daisy, daisyOcl);

  //oclDaisy(daisy, daisyCl);
  //oclDaisy(daisy, daisyOcl);

  float * petalOffsetsS1 = generatePetalOffsets(2.5,daisy->petalsNo);
  float * petalOffsetsS2 = generatePetalOffsets(5,daisy->petalsNo);
  float * petalOffsetsS3 = generatePetalOffsets(7.5,daisy->petalsNo);

  int i;
  for(i = 0; i < daisy->petalsNo; i++)
    printf("p%d y,x - %f,%f\n",i,petalOffsetsS3[i*2],petalOffsetsS3[i*2+1]);
/*int* generateTranspositionOffsets(int windowHeight, int windowWidth,
                                  float*  petalOffsets,
                                  int     petalsNo,
                                  int*    pairedOffsetsLength)*/
  int pairedOffsetsLength;
  int actualPairs;
  int * transpositionOffsets = generateTranspositionOffsets(16,16,petalOffsetsS3,daisy->petalsNo,&pairedOffsetsLength,&actualPairs);

  //for(i = 0; i < 200; i++)
  //  printf("Pair %d: (%d,%d,%d,%d)\n",i,transpositionOffsets[i*4],transpositionOffsets[i*4+1],transpositionOffsets[i*4+2],transpositionOffsets[i*4+3]);

  printf("Paired Offsets: %d\n",pairedOffsetsLength);
  printf("Actual Pairs: %d\n",actualPairs);
  gettimeofday(&endTime,NULL);

  free(daisy->array);

  start = startTime.tv_sec+(startTime.tv_usec/1000000.0);
  end = endTime.tv_sec+(endTime.tv_usec/1000000.0);
  diff = end-start;
  printf("\nMain: %.3fs\n",diff);

  return 0;
}
