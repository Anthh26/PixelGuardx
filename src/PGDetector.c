#include "PGDetector.h"
/* No CoreML / UIKit dependencies: exercise parsing and NMS in ordinary C tests. */
#define PG_CANDIDATE_CAP 160
static double min_d(double a, double b){return a < b ? a : b;}
static double max_d(double a, double b){return a > b ? a : b;}
static int finite_d(double v){return v == v && v > -1.0e8 && v < 1.0e8;}
int pg_is_sensitive_class(int id){
  /* v3.4 label ordering: BUTTOCKS_EXPOSED=2, FEMALE_BREAST_EXPOSED=3,
     FEMALE_GENITALIA_EXPOSED=4, ANUS_EXPOSED=6, MALE_GENITALIA_EXPOSED=14. */
  return id == 2 || id == 3 || id == 4 || id == 6 || id == 14;
}
static double overlap(PGBox a,PGBox b){
  double x1=max_d(a.x,b.x), y1=max_d(a.y,b.y);
  double x2=min_d(a.x+a.width,b.x+b.width),y2=min_d(a.y+a.height,b.y+b.height);
  double inter=max_d(0,x2-x1)*max_d(0,y2-y1);
  double uni=a.width*a.height+b.width*b.height-inter;
  return uni>0 ? inter/uni : 0;
}
int pg_decode_yolo(PGTensorReader reader, void *ctx, int channels, int proposals,
                   double model_w, double model_h, double threshold,
                   PGBox *boxes, int capacity){
  if(!reader || !boxes || capacity<=0 || channels!=PG_CHANNELS ||
     proposals<=0 || proposals>15000 || !finite_d(model_w) || !finite_d(model_h) ||
     model_w<32 || model_h<32 || !finite_d(threshold) || threshold<=0 || threshold>1)
    return 0;
  if(capacity>PG_MAX_RESULTS)capacity=PG_MAX_RESULTS;
  PGBox candidates[PG_CANDIDATE_CAP];
  int candidate_count=0;
  for(int p=0;p<proposals;p++){
    double best=threshold; int best_id=-1;
    for(int k=0;k<PG_CLASS_COUNT;k++){
      double value=reader(ctx,4+k,p);
      if(finite_d(value) && value>best && value<=1.0001){best=value;best_id=k;}
    }
    if(best_id<0 || !pg_is_sensitive_class(best_id))continue;
    double cx=reader(ctx,0,p),cy=reader(ctx,1,p),w=reader(ctx,2,p),h=reader(ctx,3,p);
    if(!finite_d(cx)||!finite_d(cy)||!finite_d(w)||!finite_d(h)||w<=0||h<=0)continue;
    /* Native YOLO v8 logits are pixel positions on the square, scale-filled input. */
    double x1=(cx-w*0.5)/model_w,x2=(cx+w*0.5)/model_w;
    double ty=(cy-h*0.5)/model_h, by=(cy+h*0.5)/model_h;
    /* Convert from top-left origin to Core Image's bottom-left origin. */
    double y1=1.0-by,y2=1.0-ty;
    x1=max_d(0,min_d(1,x1));x2=max_d(0,min_d(1,x2));
    y1=max_d(0,min_d(1,y1));y2=max_d(0,min_d(1,y2));
    if(x2-x1 < .002 || y2-y1 < .002)continue;
    PGBox b={x1,y1,x2-x1,y2-y1,best,best_id};
    /* Keep the top scoring candidates instead of the first proposals. */
    int i=0;
    while(i<candidate_count && candidates[i].score>=b.score)i++;
    if(i>=PG_CANDIDATE_CAP)continue;
    int n=candidate_count<PG_CANDIDATE_CAP?candidate_count+1:candidate_count;
    for(int j=n-1;j>i;j--)candidates[j]=candidates[j-1];
    candidates[i]=b;candidate_count=n;
  }
  int out=0;
  for(int i=0;i<candidate_count && out<capacity;i++){
    PGBox b=candidates[i];int reject=0;
    for(int j=0;j<out;j++){
      if(boxes[j].class_id==b.class_id && overlap(boxes[j],b)>0.45){reject=1;break;}
    }
    if(!reject)boxes[out++]=b;
  }
  return out;
}

int pg_decode_nms(PGTensorReader coordinates, void *coords_ctx,
                  PGTensorReader confidence, void *conf_ctx,
                  int conf_channels, int proposals, double threshold,
                  PGBox *boxes, int capacity){
  if (!coordinates || !confidence || !boxes || capacity <= 0 ||
      conf_channels < PG_CLASS_COUNT || conf_channels > 256 ||
      proposals <= 0 || proposals > 15000 ||
      !finite_d(threshold) || threshold <= 0 || threshold > 1) return 0;
  if(capacity > PG_MAX_RESULTS) capacity = PG_MAX_RESULTS;
  int out=0;
  /* Embedded Core ML NMS has already suppressed overlapping candidates. */
  for(int p=0;p<proposals && out<capacity;p++){
    double best=threshold;int best_id=-1;
    for(int k=0;k<PG_CLASS_COUNT;k++){
      double value=confidence(conf_ctx,k,p);
      if(finite_d(value) && value>best && value<=1.0001){best=value;best_id=k;}
    }
    if(best_id<0 || !pg_is_sensitive_class(best_id))continue;
    double cx=coordinates(coords_ctx,0,p), cy=coordinates(coords_ctx,1,p);
    double w=coordinates(coords_ctx,2,p), h=coordinates(coords_ctx,3,p);
    if(!finite_d(cx)||!finite_d(cy)||!finite_d(w)||!finite_d(h)||w<=0||h<=0)continue;
    double x1=max_d(0,min_d(1,cx-w/2.0));
    double x2=max_d(0,min_d(1,cx+w/2.0));
    double y1=max_d(0,min_d(1,1.0-(cy+h/2.0)));
    double y2=max_d(0,min_d(1,1.0-(cy-h/2.0)));
    if(x2-x1<.002||y2-y1<.002)continue;
    PGBox b={x1,y1,x2-x1,y2-y1,best,best_id};
    boxes[out++]=b;
  }
  return out;
}
