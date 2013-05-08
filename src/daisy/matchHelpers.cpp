#include "matchHelpers.h"
#include <stdio.h>


transform get2dProjection(point source1, point source2, point target1, point target2){

  float thI = atan2((source2.y - source1.y),(source2.x - source1.x));
  float thW = atan2((target2.y - target1.y),(target2.x - target1.x));

  float th = thW - thI;

  float stop = target1.x - target2.x;
  float sbot = (source1.x-source2.x) * cos(th) - (source1.y - source2.y) * sin(th);

  float s;
  if(sbot == 0) s = 1;
  else if(stop == 0) s = fabs((target1.y-target2.y) / (source1.y-source2.y));
  else s = stop / sbot;

  float tx = target1.x - source1.x * s * cos(th) + source1.y * s * sin(th);
  float ty = target1.y - source1.x * s * sin(th) - source1.y * s * cos(th);

  return {th, s, tx, ty};

}


point estimateObjectCentre(point * templatePoints, int * templateMatches, int * targetMatches, int corrsNo,
                           point targetSize, point templateSize, transform * t){

  point centre = {0,0};
  point * centres = (point*)malloc(sizeof(point) * corrsNo * (corrsNo-1));
  int centresNo = 0;

  // for every pair of correspondences in the templateMatches/targetMatches set
  for(int c1 = 0; c1 < corrsNo; c1++){

    point source1 = templatePoints[templateMatches[c1]];
    point target1 = { (float)(targetMatches[c1] % static_cast<int>(targetSize.x)), targetMatches[c1] / targetSize.x };

    for(int c2 = 0; c2 < corrsNo; c2++){

      point source2 = templatePoints[templateMatches[c2]];
      point target2 = { (float)(targetMatches[c2] % static_cast<int>(targetSize.x)), targetMatches[c2] / targetSize.x };
      // compute 2d projection
      transform trans = get2dProjection(source1, source2, target1, target2);
      // project template centre with it
      centres[centresNo++] = projectPoint({templateSize.x / 2, templateSize.y / 2}, trans);
      centre.x += centres[centresNo-1].x;
      centre.y += centres[centresNo-1].y;
      
    }
  }
  
  // get mean centre
  centre.x /= centresNo;
  centre.y /= centresNo;

  // get distances of others from mean centre
  float * distances = (float*) malloc(sizeof(float) * centresNo);
  float meanDistance = 0;
  for(int c = 0; c < centresNo; c++){
    distances[c] = sqrt(pow(centres[c].x - centre.x,2)+pow(centres[c].y-centre.y,2));
    meanDistance += distances[c];
  }
  meanDistance /= centresNo;

  // get std
  float stdDistance = 0;
  for(int c = 0; c < centresNo; c++)
    stdDistance += pow(distances[c] - meanDistance,2);
  stdDistance /= centresNo;

  // divide distances by std
  // from ones closer than STD_THRESH get a new centre mean
  centre.x = 0;
  centre.y = 0;
  float STD_THRESH = 0.019;
  int cNo = 0;
  for(int c = 0; c < centresNo; c++){
    if((distances[c]-meanDistance) / stdDistance < STD_THRESH){
      centre.x += centres[c].x;
      centre.y += centres[c].y;
      cNo++;
    }
  }
  centre.x /= cNo;
  centre.y /= cNo;

  *t = get2dProjection({ templateSize.x / 2, templateSize.y / 2}, templatePoints[templateMatches[0]],
                       centre, { (float)(targetMatches[0] % static_cast<int>(targetSize.x)), targetMatches[0] / targetSize.x });

  return centre;

}

void projectTargetSeeds(point * seedTemplatePoints, point * seedTargetPoints, int seedsNo,
                        point * templatePoints, int * templateMatches,
                        int * matches, int matchesNo, point targetSize, transform * t){

  for(int s = 0; s < seedsNo; s++){

    // project template seed with overall projection
    point initialSeed = projectPoint(seedTemplatePoints[s], *t);

    // find 2 closest correspondences in target
    float min1 = 9999;
    float min2 = 9999;
    int argmin1 = -1;
    int argmin2 = -1;
    for(int t = 0; t < matchesNo; t++){
      float dist = sqrt(pow(initialSeed.x - 
                       (matches[t] % static_cast<int>(targetSize.x)),2) + 
                        pow(initialSeed.y - floor(matches[t] / targetSize.x),2));
      if(dist < min1){
        min2 = min1;
        min1 = dist;
        argmin2 = argmin1;
        argmin1 = t;
      }
      else if(dist < min2){
        min2 = dist;
        argmin2 = t;
      }
    }

    if(argmin1 < 0 || argmin2 < 0)
      printf("HELPPPPPP!\n");

    // get projection of the closest points
    transform localt = get2dProjection(templatePoints[templateMatches[argmin1]], templatePoints[templateMatches[argmin2]],
                   { (float)(matches[argmin1] % static_cast<int>(targetSize.x)), floorf(matches[argmin1] / targetSize.x)},
                   { (float)(matches[argmin2] % static_cast<int>(targetSize.x)), floorf(matches[argmin2] / targetSize.x)});

    // re-project template seed
    seedTargetPoints[s] = projectPoint(seedTemplatePoints[s], localt);

  }

}

point projectPoint(point p, transform t){

  return { (float)(p.x * t.s * cos(t.th) - p.y * t.s * sin(t.th) + t.tx),
           (float)(p.x * t.s * sin(t.th) + p.y * t.s * cos(t.th) + t.ty)};

}

transform * minimise2dProjection(point * templatePoints, int * templateMatches, int * targetMatches, int corrsNo,
                                 point targetSize, point templateSize, float * projectionErrors){

  float minErrorSum = 9999;
  float * errors = (float*) malloc(sizeof(float) * corrsNo);
  transform * minTransform = (transform*)malloc(sizeof(transform));

  // for every pair of correspondences in the templateMatches/targetMatches set
  for(int c1 = 0; c1 < corrsNo; c1++){

    point source1 = templatePoints[templateMatches[c1]];
    point target1 = { (float)(targetMatches[c1] % static_cast<int>(targetSize.x)), floorf(targetMatches[c1] / targetSize.x) };

    for(int c2 = 0; c2 < corrsNo; c2++){

      point source2 = templatePoints[templateMatches[c2]];
      point target2 = { (float)(targetMatches[c2] % static_cast<int>(targetSize.x)), floorf(targetMatches[c2] / targetSize.x) };
      // compute 2d projection
      transform trans = get2dProjection(source1, source2, target1, target2);
      // measure errors
      float errorSum = 0;
      for(int c = 0; c < corrsNo; c++){
        point p = projectPoint(templatePoints[templateMatches[c]], trans);
        errors[c] =   sqrt(pow(targetMatches[c] % static_cast<int>(targetSize.x) - p.x,2) + 
                           pow(floor(targetMatches[c] / targetSize.x) - p.y,2))
                       / sqrt(pow(targetMatches[c] % static_cast<int>(targetSize.x) - templatePoints[templateMatches[c]].x,2)
                            + pow(floor(targetMatches[c] / targetSize.x) - templatePoints[templateMatches[c]].y,2));
        errorSum += errors[c];
      }
      
      if(errorSum < minErrorSum){

        minErrorSum = errorSum;
        *minTransform = trans;
        memcpy(projectionErrors, errors, corrsNo * sizeof(float));

      }
    }
  }

  return minTransform;
}



