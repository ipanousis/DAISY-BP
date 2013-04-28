
#ifndef POINT_STRUCT
#define POINT_STRUCT
typedef struct {

  float x;
  float y;

} point;
#endif

#define min(a,b) (a > b ? b : a)
#define max(a,b) (a > b ? a : b)

#ifndef TRANSFORM_STRUCT
#define TRANSFORM_STRUCT
typedef struct {

  float th;
  float s;
  float tx;
  float ty;

} transform;
#endif

