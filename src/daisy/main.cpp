#include "main.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "cv.h"
#include "highgui.h"

using namespace kutility;

double getStd(double* observations, int length);
double timeDiff(struct timeval start, struct timeval end);
void displayFlow(int width, int height);

float * flowArray = NULL;
float * flowArrayTemp = NULL;
int flowHalo;

int main( int argc, char **argv  )
{
  struct timeval startTime,endTime;
  struct timeval endDaisyTime,endSearchTime;

  time_params times;
  times.measureDeviceHostTransfers = 0;

  gettimeofday(&startTime,NULL);

  float sigmas[3] = {2.5,5,7.5};

  int counter = 1;

  // Get command line options
  if(argc > counter+1 && (!strcmp("-i", argv[counter]) || !strcmp("--image", argv[counter]))){

    string filename = argv[++counter];

    uchar* srcArray = NULL;
    int width, height;

    load_gray_image (filename, srcArray, height, width);
    printf("Image HeightxWidth=%dx%d\n",height, width);
    counter++;
    
    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));

    daisy_params * daisy = newDaisyParams(srcArray, height, width, 8, 8, 3, sigmas);

    initOcl(daisyPrograms,daisyCl);

    daisy->oclPrograms = daisyPrograms;

    oclDaisy(daisy, daisyCl, &times);

/*
    //printf("Paired Offsets: %d\n",pairedOffsetsLength);
    //printf("Actual Pairs: %d\n",actualPairs);

    //int oclDaisySearch(ocl_constructs * ocl, daisy_params * daisyRef, daisy_params * daisyTar)

    string binaryfile = filename;
    binaryfile += ".bdaisy";
    kutility::save_binary(binaryfile, daisy->descriptors, daisy->paddedHeight * daisy->paddedWidth, daisy->descriptorLength, 1, kutility::TYPE_FLOAT);*/

    gettimeofday(&endTime,NULL);

    free(daisy->array);

    double start = startTime.tv_sec+(startTime.tv_usec/1000000.0);
    double end = endTime.tv_sec+(endTime.tv_usec/1000000.0);
    printf("\nMain: %.3fs\n",end-start);
  }else if(argc > counter+1 && !strcmp("-s1", argv[counter]) && !strcmp("-s2",argv[counter+2])){

    string filename1 = argv[(++counter)++];
    string filename2 = argv[++counter];

    uchar* srcArrayR = NULL;
    uchar* srcArrayT = NULL;
    int width, height;

    load_gray_image (filename1, srcArrayR, height, width);
    load_gray_image (filename2, srcArrayT, height, width);

    printf("Image HeightxWidth=%dx%d\n",height, width);
    counter++;
    
    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));

    daisy_params * daisyR = newDaisyParams(srcArrayR, height, width, 8, 8, 3, sigmas);
    daisy_params * daisyT = newDaisyParams(srcArrayT, height, width, 8, 8, 3, sigmas);

    initOcl(daisyPrograms,daisyCl);

    daisyR->oclPrograms = daisyPrograms;
    daisyT->oclPrograms = daisyPrograms;

    oclDaisy(daisyR, daisyCl, &times);
    oclDaisy(daisyT, daisyCl, &times);

    int swidth, sheight;

    int error = oclDaisySearch(daisyCl, daisyR, daisyT, flowArray, &flowHalo, &swidth, &sheight);

    gettimeofday(&endTime,NULL);

    free(daisyR->array);
    free(daisyT->array);
    free(flowArray);

    double start = startTime.tv_sec+(startTime.tv_usec/1000000.0);
    double end = endTime.tv_sec+(endTime.tv_usec/1000000.0);
    printf("\nMain: %.3fs\n",end-start);
  }else if(argc > counter+1 && !strcmp("-ff", argv[counter])){
    
    char * frameTemplate = "basket-sequence/basket%04d.jpg";

    uchar* srcArrayR = NULL;
    uchar* srcArrayT = NULL;

    int swidth, sheight;
    int width = 512;
    int height = 480;
    int numFrames = 875;

    ocl_constructs * daisyCl = newOclConstructs(0,0,0);
    ocl_daisy_programs * daisyPrograms = (ocl_daisy_programs*)malloc(sizeof(ocl_daisy_programs));

    initOcl(daisyPrograms,daisyCl);

    char * frameName = (char*)malloc(sizeof(char)*100);

    int frameNum = 0;

    sprintf(frameName,frameTemplate,frameNum++);
    load_gray_image(frameName, srcArrayR, height, width);

    sprintf(frameName,frameTemplate,frameNum++);
    load_gray_image(frameName, srcArrayT, height, width);

    daisy_params * daisyR = newDaisyParams(srcArrayR, height, width, 8, 8, 3, sigmas);
    daisy_params * daisyT = newDaisyParams(srcArrayT, height, width, 8, 8, 3, sigmas);
    daisy_params * daisyTemp;

    daisyR->oclPrograms = daisyPrograms;
    daisyT->oclPrograms = daisyPrograms;

    oclDaisy(daisyR, daisyCl, &times);
    oclDaisy(daisyT, daisyCl, &times);

    //pthread_create(&glutThread, NULL, (void*)startGlutMainLoop, (void*)NULL);

    flowArray = (float*)malloc(sizeof(float) * width * height * 2);
    flowArrayTemp = (float*)malloc(sizeof(float) * width * height * 2);

    cvNamedWindow("DAISY Flow",1);

    for(int currentFrame = frameNum; currentFrame < numFrames; currentFrame++){

      daisyTemp = daisyR;
      daisyR = daisyT;
      daisyT = daisyTemp;

      gettimeofday(&startTime,NULL);

      sprintf(frameName,frameTemplate,currentFrame);
      load_gray_image(frameName, daisyT->array, height, width);

      oclDaisy(daisyT, daisyCl, &times);

      gettimeofday(&endDaisyTime,NULL);

      oclDaisySearch(daisyCl, daisyR, daisyT, flowArray, &flowHalo, &swidth, &sheight);

      gettimeofday(&endSearchTime,NULL);

      displayFlow(swidth, sheight);

      gettimeofday(&endTime,NULL);

      float daisyFps = 1.0 / timeDiff(startTime,endDaisyTime);
      float searchFps = 1.0 / timeDiff(endDaisyTime,endSearchTime);
      float totalFps = 1.0 / timeDiff(startTime,endTime);

      printf("FPS= %.1f, DAISY= %.1f, SEARCH= %.1f\n",totalFps,daisyFps,searchFps);
    }


  }else{
    
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

    /* Standard ranges QVGA,VGA,SVGA,XGA,SXGA,SXGA+,UXGA,QXGA*/
    int heights[8] = {320,640,800,1024,1280,1400,1600,2048};
    int widths[8] = {240,480,600,768,1024,1050,1200,1536};
    int total = 8;

    /* Without transfer ranges */
    /*int heights[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    int widths[12] = {128,256,384,512,640,768,896,1024,1152,1280,1408,1536};
    int total = 12;*/
    
    /* With transfer ranges */
    /*int heights[4] = {128,256,384,512};//,640,768,896,1024,1152,1280,1408,1536};
    int widths[4] = {128,256,384,512};//,640,768,896,1024,1152,1280,1408,1536};
    int total = 4;//12;*/

    // allocate the memory
    unsigned char * array = (unsigned char *)malloc(sizeof(unsigned char) * heights[total-1] * widths[total-1]);

    // generate random value input
    for(int i = 0; i < heights[total-1]*widths[total-1]; i++)
      array[i] = i % 255;

    fprintf(csvOut,"height,width,convgrad,transA,transB,transBhost,whole,wholestd,dataTransfer,iterations,success\n");

    char* templateRow = "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d\n";

    for(int w = 0; w < total; w++){

      int width = widths[w];
      int height = heights[w];

      printf("%dx%d\n",height,width);

      int iterations = 25;
      int success = 0;
      double * wholeTimes = (double*)malloc(sizeof(double) * iterations);

      time_params times;

      double t_convGrad = 0;
      double t_transA = 0;
      double t_transB = 0;
      double t_transBhost = 0;
      double t_whole = 0;

      times.measureDeviceHostTransfers = 0;

      daisy = newDaisyParams(array, height, width, 8, 8, 3, NULL);
      daisy->oclPrograms = daisyPrograms;

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

      fprintf(csvOut, templateRow, height, width, t_convGrad, t_transA, t_transB, t_transBhost, t_whole, wholeStd,
                      times.measureDeviceHostTransfers, iterations, success);

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

void displayFlow(int width, int height){

  int row;

  #pragma omp parallel for private(row)
  for(row = 0; row < height; row++)
    memcpy(flowArrayTemp + row * width, flowArray + row * width, width * sizeof(float));

  IplImage * img = cvCreateImage(cvSize(width, height), 8, 3);
  
  #pragma omp parallel for private(row)
  for(row = 0; row < height; row++){

    for(int col = 0; col < width; col++){

      
      img->imageData[row*width*3+col*3] = (fabs(flowArrayTemp[row * width + col]) / 5.0) * 255;
      img->imageData[row*width*3+col*3+1] = (fabs(flowArrayTemp[width * height + row * width + col]) / 5.0) * 255;
      img->imageData[row*width*3+col*3+2] = (fabs(flowArrayTemp[row * width + col]) / 5.0) * 255;
      //rgbRow[col*3+1] = (fabs(flowArrayTemp[row * width + col]) / 40.0) * 255;
      //rgbRow[col*3+2] = (fabs(flowArrayTemp[width * height + row * width + col]) / 40.0) * 255;

    }

  }
  cvShowImage("DAISY Flow", img);
  cvWaitKey(0);

}

/*
void* redisplay(int value){

  if(!active) return NULL;
  
  glutPostRedisplay();

  glutTimerFunc(10, (void*)redisplay, 0);

  return NULL;

}

void startGlutMainLoop(void* params){

  // Initialise GLUT
  int argc = 0;
  glutInit(&argc, NULL);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(SIZE, SIZE); //Set the window size

  // Create the window
  displayWindow = glutCreateWindow("DAISY Flow");
  initRendering(); // init rendering

  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(resize);
  glutTimerFunc(20, (void*)redisplay, 0);

  glutMainLoop();

  return NULL;
}

void display(){

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);

  glLoadIdentity();

  glBegin(GL_LINE_STRIP);
  glColor3f(1.0f, 1.0f, 1.0f);

  glVertex3f(-2.0f, 2.0f, -5.0f);
  glVertex3f( 2.0f, 2.0f, -5.0f);
  glVertex3f( 2.0f,-2.0f, -5.0f);
  glVertex3f(-2.0f,-2.0f, -5.0f);
  glVertex3f(-2.0f, 2.0f, -5.0f);
  glEnd();

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  /*if(computeMT){
    for(drawingrow = 0; drawingrow < mandel->dimensions.y; drawingrow++){
      if(readyrows[drawingrow]){
        glVertexPointer(3, GL_FLOAT, 6 * sizeof(GLfloat), &(vertexarray[drawingrow * mandel->dimensions.x * 6]));
        glColorPointer(3, GL_FLOAT, 6 * sizeof(GLfloat), &(vertexarray[drawingrow * mandel->dimensions.x * 6 + 3]));
        glDrawArrays(GL_POINTS, 0, mandel->dimensions.x);
      }
    }
  }
  else if(computeGPU){*/
/*
  glVertexPointer(3, GL_FLOAT, 3 * sizeof(GLfloat), &(vertexarray[0]));
  glColorPointer(3, GL_FLOAT, 3 * sizeof(GLfloat), &(colorarray[0]));
  glDrawArrays(GL_POINTS, 0, mandel->dimensions.x * mandel->dimensions.y);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glutSwapBuffers(); // Sends 3D scene
  glFlush();

}

void keyboard(){

}

void resize(){


}*/
