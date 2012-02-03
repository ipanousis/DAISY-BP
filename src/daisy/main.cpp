#include "main.h"
#include <stdio.h>

using namespace kutility;

int main( int argc, char **argv  )
{
  int counter = 1;

  int width, height;
  uchar* srcArray = NULL;

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
    printf("HxW=%dx%d\n",height, width);
    counter++;
  }
  
  ocl_constructs * daisyCl = newOclConstructs(0,0,0);
  daisy_params * daisy = newDaisyParams(srcArray, height, width, 8, 3);
  initOcl(daisy, daisyCl);
  oclDaisy(daisy, daisyCl);

  return 0;
}
