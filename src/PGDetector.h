#ifndef PG_DETECTOR_H
#define PG_DETECTOR_H
/* NudeNet v3.4 YOLOv8 label ordering: 18 classes. Results in normalized
 * bottom-left-origin image coordinates. Pure C for standalone unit tests. */
#define PG_CLASS_COUNT 18
#define PG_CHANNELS 22
#define PG_MAX_RESULTS 40
typedef struct { double x,y,width,height,score; int class_id; } PGBox;
typedef double (*PGTensorReader)(void *ctx, int channel, int proposal);
int pg_decode_yolo(PGTensorReader reader, void *ctx, int channels, int proposals,
                   double model_w, double model_h, double threshold,
                   PGBox *boxes, int capacity);
/* Core ML NMS pipeline can return separate coordinates[N,4] and
 * confidence[N,numClasses] matrices. Coordinates are normalized xywh with
 * top-left convention; rows must match; 18 real classes (possibly padded). */
int pg_decode_nms(PGTensorReader coordinates, void *coords_ctx,
                  PGTensorReader confidence, void *conf_ctx,
                  int conf_channels, int proposals, double threshold,
                  PGBox *boxes, int capacity);
int pg_is_sensitive_class(int class_id);
#endif
