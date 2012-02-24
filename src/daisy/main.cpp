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

  initOcl(daisy, daisyCl);
  //initOcl(daisy, daisyOcl);

  oclDaisy(daisy, daisyCl);
  //oclDaisy(daisy, daisyOcl);

  //printf("Paired Offsets: %d\n",pairedOffsetsLength);
  //printf("Actual Pairs: %d\n",actualPairs);

  //kutility::save(filename, m_dense_descriptors, m_h*m_w, m_descriptor_size );
  string binaryfile = filename;
  binaryfile += ".bdaisy";
//   kutility::save_binary(filename, m_dense_descriptors, m_h*m_w, m_descriptor_size, 1, kutility::TYPE_FLOAT );
  kutility::save_binary(binaryfile, daisy->descriptors, height * width, daisy->descriptorLength, 1, kutility::TYPE_FLOAT);

  gettimeofday(&endTime,NULL);

  free(daisy->array);

  start = startTime.tv_sec+(startTime.tv_usec/1000000.0);
  end = endTime.tv_sec+(endTime.tv_usec/1000000.0);
  diff = end-start;
  printf("\nMain: %.3fs\n",diff);

  return 0;
}
