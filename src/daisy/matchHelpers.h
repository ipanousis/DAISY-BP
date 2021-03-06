#include "general.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

point estimateObjectCentre(point * templatePoints, int * templateMatches, 
                           int * targetMatches, int corrsNo,
                           point targetSize, point templateSize, 
                           transform * trans);

transform * minimise2dProjection(point * templatePoints, int * templateMatches, 
                                 int * targetMatches, int corrsNo,
                                 point targetSize, point templateSize, 
                                 float * projectionErrors);

void projectTargetSeeds(point * seedTemplatePoints, point * seedTargetPoints, int seedsNo,
                        point * templatePoints, int * templateMatches,
                        int * matches, int matchesNo, point coarseTargetSize, transform * t);

void projectPoint(point p, transform t, point * to);
