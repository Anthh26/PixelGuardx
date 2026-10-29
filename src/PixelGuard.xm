// PixelGuard 0.8 — experimental, Photos-only. Expand eligible image sizes and orientations.
// On-device NudeNet via Vision/Core ML. This is not a guaranteed privacy barrier.
#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreML/CoreML.h>
#import <CoreImage/CoreImage.h>
#import <objc/runtime.h>
#import <rootless.h>
#import <math.h>
#import <stdint.h>
extern "C" {
#import "PGDetector.h"
}

static char PGBypassKey, PGTokenKey, PGCachedKey, PGCleanKey;
static volatile int PGOutstanding = 0;
static const int PGMaxOutstanding = 3;

static NSString *PGDisabledPath(void) {
    return ROOT_PATH_NS(@"/Library/Application Support/PixelGuard/disabled");
}
static NSString *PGModelPath(void) {
    return ROOT_PATH_NS(@"/Library/Application Support/PixelGuard/NudeNet.mlmodelc");
}
static BOOL PGIsEnabled(void) {
    // Installed ENABLED. Create this file to disable temporarily.
    return ![[NSFileManager defaultManager] fileExistsAtPath:PGDisabledPath()];
}
static BOOL PGHasModel(void) {
    BOOL directory = NO;
    return [[NSFileManager defaultManager] fileExistsAtPath:PGModelPath() isDirectory:&directory] && directory;
}
static dispatch_queue_t PGQueue(void) {
    static dispatch_queue_t queue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ queue = dispatch_queue_create("com.anth.pixelguard.inference", DISPATCH_QUEUE_SERIAL); });
    return queue;
}
static VNCoreMLModel *PGVisionModel(void) {
    static VNCoreMLModel *result;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSURL *url = [NSURL fileURLWithPath:PGModelPath()];
        NSError *error = nil;
        MLModelConfiguration *config = [MLModelConfiguration new];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model) result = [VNCoreMLModel modelForMLModel:model error:&error];
        if (!result) NSLog(@"[PixelGuard] CoreML load error: %@", error);
    });
    return result;
}

typedef struct {
    const void *bytes;
    int dtype;
    NSInteger channels, proposals;
    NSInteger channelStride, proposalStride;
} PGTensor;

// Handles [N,C], [C,N], [1,N,C], [1,C,N] exported NMS feature arrays.
static BOOL PGTensorFromArray(MLMultiArray *array, NSInteger expectedChannels,
                              BOOL allowPadding, PGTensor *out) {
    if (!array || !out) return NO;
    NSArray<NSNumber *> *shape = array.shape;
    NSArray<NSNumber *> *strides = array.strides;
    NSInteger channelAxis = -1, proposalAxis = -1, count = shape.count;
    if (count != 2 && count != 3) return NO;
    if (count == 3 && shape[0].integerValue != 1) return NO;
    NSInteger start = count == 3 ? 1 : 0;
    for (NSInteger c = start; c < count; c++) {
        NSInteger n = shape[c].integerValue;
        BOOL accepted = allowPadding ? (n == expectedChannels || n == 80) : (n == expectedChannels);
        if (accepted) { channelAxis = c; proposalAxis = (c == start ? start+1 : start); break; }
    }
    if (channelAxis < 0 || proposalAxis >= count) return NO;
    NSInteger proposals = shape[proposalAxis].integerValue;
    if (proposals < 1 || proposals > 15000 || !array.dataPointer) return NO;
    MLMultiArrayDataType dtype = array.dataType;
    if (dtype != MLMultiArrayDataTypeFloat32 && dtype != MLMultiArrayDataTypeFloat16 &&
        dtype != MLMultiArrayDataTypeDouble) return NO;
    *out = (PGTensor){array.dataPointer, (int)dtype, shape[channelAxis].integerValue,
                      proposals, strides[channelAxis].integerValue, strides[proposalAxis].integerValue};
    return YES;
}
static double PGTensorRead(void *context, int channel, int proposal) {
    PGTensor *tensor = (PGTensor *)context;
    if (!tensor || !tensor->bytes || channel < 0 || channel >= tensor->channels ||
        proposal < 0 || proposal >= tensor->proposals) return 0;
    NSInteger index = channel * tensor->channelStride + proposal * tensor->proposalStride;
    if (tensor->dtype == MLMultiArrayDataTypeFloat32) return ((const float *)tensor->bytes)[index];
    if (tensor->dtype == MLMultiArrayDataTypeDouble) return ((const double *)tensor->bytes)[index];
    if (tensor->dtype == MLMultiArrayDataTypeFloat16) return (double)((const __fp16 *)tensor->bytes)[index];
    return 0;
}
static BOOL PGEligibleFrame(UIImage *image) {
    if (!image || !image.CGImage) return NO;
    size_t width = CGImageGetWidth(image.CGImage), height = CGImageGetHeight(image.CGImage);
    // Small grid thumbnails and typical 48 MP images need to be eligible.
    // Very large unsupported images remain covered by the fail-closed view hook.
    return width >= 96 && height >= 96 && width <= 12000 && height <= 12000 &&
           ((double)width * (double)height) <= 80000000.0;
}
// Short UIKit animated UIImage sequences are scanned before they can play.
static BOOL PGEligibleImage(UIImage *image) {
    if (!image) return NO;
    NSArray<UIImage *> *frames=image.images;
    if (frames.count) {
        if (frames.count > 48) return NO;
        for (UIImage *frame in frames) if (!PGEligibleFrame(frame)) return NO;
        return YES;
    }
    return PGEligibleFrame(image);
}
static CIContext *PGContext(void) {
    static CIContext *ctx;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ ctx = [CIContext contextWithOptions:nil]; });
    return ctx;
}
static UIImage *PGPlaceholder(void) {
    static UIImage *placeholder;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        UIGraphicsBeginImageContextWithOptions(CGSizeMake(2, 2), YES, 1);
        [[UIColor blackColor] setFill];
        UIRectFill(CGRectMake(0, 0, 2, 2));
        placeholder = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
    });
    return placeholder;
}
static int PGAddRecognizedBoxes(NSArray<VNObservation *> *observations, PGBox *boxes) {
    const NSArray<NSString *> *sensitive = @[@"BUTTOCKS_EXPOSED", @"FEMALE_BREAST_EXPOSED",
      @"FEMALE_GENITALIA_EXPOSED", @"ANUS_EXPOSED", @"MALE_GENITALIA_EXPOSED"];
    int n = 0;
    for (VNObservation *observation in observations) {
        if (![observation isKindOfClass:[VNRecognizedObjectObservation class]]) continue;
        VNRecognizedObjectObservation *object = (VNRecognizedObjectObservation *)observation;
        VNClassificationObservation *top = object.labels.firstObject;
        if (!top || top.confidence < .55 || ![sensitive containsObject:top.identifier] ||
            n == PG_MAX_RESULTS) continue;
        CGRect r = object.boundingBox; // Vision boxes are bottom-left normalized.
        if (r.size.width < .002 || r.size.height < .002) continue;
        boxes[n++] = (PGBox){r.origin.x, r.origin.y, r.size.width, r.size.height,
                             top.confidence, 0};
    }
    return n;
}
static int PGAddNMSBoxes(NSArray<VNObservation *> *observations, PGBox *boxes) {
    MLMultiArray *coordinates = nil, *confidence = nil;
    for (VNObservation *observation in observations) {
        if (![observation isKindOfClass:[VNCoreMLFeatureValueObservation class]]) continue;
        VNCoreMLFeatureValueObservation *feature = (VNCoreMLFeatureValueObservation *)observation;
        if ([feature.featureName isEqualToString:@"coordinates"])
            coordinates = feature.featureValue.multiArrayValue;
        else if ([feature.featureName isEqualToString:@"confidence"])
            confidence = feature.featureValue.multiArrayValue;
    }
    PGTensor c, p;
    if (!PGTensorFromArray(coordinates, 4, NO, &c) ||
        !PGTensorFromArray(confidence, PG_CLASS_COUNT, YES, &p) ||
        c.proposals != p.proposals) return 0;
    return pg_decode_nms(PGTensorRead, &c, PGTensorRead, &p, (int)p.channels,
                         (int)p.proposals, .55, boxes, PG_MAX_RESULTS);
}
// Upright rasterization gives Vision and Core Image the same coordinate system.
static UIImage *PGUprightImage(UIImage *image) {
    if (!image || !image.CGImage) return nil;
    CGFloat pixelWidth = (CGFloat)CGImageGetWidth(image.CGImage);
    CGFloat pixelHeight = (CGFloat)CGImageGetHeight(image.CGImage);
    if (image.imageOrientation == UIImageOrientationUp &&
        MAX(pixelWidth, pixelHeight) <= 2560.0) return image;
    // Bound the processed render to around 2560 px on its longest edge.
    // Otherwise a 48 MP zoom image allocates very large inference/compositing buffers.
    CGFloat pointsScale = MAX(image.scale, 1.0);
    CGFloat displayWidth = image.size.width * pointsScale;
    CGFloat displayHeight = image.size.height * pointsScale;
    CGFloat shrink = MIN(1.0, 2560.0 / MAX(displayWidth, displayHeight));
    CGSize size = CGSizeMake(MAX(1, displayWidth*shrink), MAX(1, displayHeight*shrink));
    UIGraphicsBeginImageContextWithOptions(size, NO, 1.0);
    [image drawInRect:CGRectMake(0, 0, size.width, size.height)];
    UIImage *upright = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return upright;
}
static UIImage *PGProcessFrame(UIImage *source) {
    if (!PGEligibleFrame(source) || !PGIsEnabled()) return nil;
    source = PGUprightImage(source);
    if (!source || !source.CGImage) return nil;
    VNCoreMLModel *model = PGVisionModel();
    if (!model) return nil;
    VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:model];
    request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:source.CGImage options:@{}];
    NSError *error = nil;
    if (![handler performRequests:@[request] error:&error]) {
        NSLog(@"[PixelGuard] inference error: %@", error);
        return nil;
    }
    PGBox boxes[PG_MAX_RESULTS];
    int n = PGAddRecognizedBoxes(request.results, boxes);
    if (!n) n = PGAddNMSBoxes(request.results, boxes);
    // No exposed regions: successfully scanned image is safe to display.
    if (n <= 0) return source;

    CGFloat width = CGImageGetWidth(source.CGImage), height = CGImageGetHeight(source.CGImage);
    CIImage *original = [CIImage imageWithCGImage:source.CGImage];
    CIFilter *pixelFilter = [CIFilter filterWithName:@"CIPixellate"];
    [pixelFilter setValue:original forKey:kCIInputImageKey];
    [pixelFilter setValue:@(MAX(14.0, MIN(width,height)/42.0)) forKey:kCIInputScaleKey];
    CIImage *pixelated = pixelFilter.outputImage;
    if (!pixelated) return nil;
    CIImage *merged = original;
    int applied = 0;
    for (int i=0; i<n; i++) {
        PGBox b=boxes[i];
        // PGBox origin is bottom-left normalized, matching Core Image.
        CGFloat x0 = MAX(0, b.x - b.width*.12) * width;
        CGFloat y0 = MAX(0, b.y - b.height*.12) * height;
        CGFloat x1 = MIN(1, b.x + b.width*1.12) * width;
        CGFloat y1 = MIN(1, b.y + b.height*1.12) * height;
        CGRect rect = CGRectMake(x0, y0, x1-x0, y1-y0);
        if (rect.size.width <= 2 || rect.size.height <= 2) continue;
        CIImage *piece = [pixelated imageByCroppingToRect:rect];
        merged = [piece imageByCompositingOverImage:merged];
        applied++;
    }
    if (!applied) return source;
    CGImageRef cgImage = [PGContext() createCGImage:merged fromRect:original.extent];
    if (!cgImage) return nil;
    UIImage *result = [UIImage imageWithCGImage:cgImage scale:source.scale orientation:UIImageOrientationUp];
    CGImageRelease(cgImage);
    return result;
}

static UIImage *PGProcess(UIImage *source) {
    if (!PGIsEnabled() || !PGEligibleImage(source)) return nil;
    NSArray<UIImage *> *frames=source.images;
    if (!frames.count) return PGProcessFrame(source);
    NSMutableArray<UIImage *> *processed=[NSMutableArray arrayWithCapacity:frames.count];
    for (UIImage *frame in frames) {
        UIImage *output = PGProcessFrame(frame);
        if (!output) return nil; // Any failed frame keeps the GIF concealed.
        [processed addObject:output];
    }
    return [UIImage animatedImageWithImages:processed duration:MAX(0.1, source.duration)];
}
static BOOL PGSubmit(UIImageView *view, UIImage *source, NSObject *token) {
    if (__sync_add_and_fetch(&PGOutstanding, 1) > PGMaxOutstanding) {
        __sync_sub_and_fetch(&PGOutstanding, 1);
        return NO; // Keep placeholder; retry while this view is current.
    }
    __weak UIImageView *weakView = view;
    dispatch_async(PGQueue(), ^{
        @autoreleasepool {
            UIImage *output = nil;
            @try { output = PGProcess(source); }
            @catch (NSException *exception) { NSLog(@"[PixelGuard] exception: %@", exception); }
            dispatch_async(dispatch_get_main_queue(), ^{
                UIImageView *strongView = weakView;
                if (strongView && objc_getAssociatedObject(strongView, &PGTokenKey) == token) {
                    objc_setAssociatedObject(strongView, &PGTokenKey, nil, OBJC_ASSOCIATION_ASSIGN);
                    objc_setAssociatedObject(strongView, &PGBypassKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                    if (output && PGIsEnabled()) {
                        if (output == source) objc_setAssociatedObject(source, &PGCleanKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                        else objc_setAssociatedObject(source, &PGCachedKey, output, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                    }
                    strongView.image = (output && PGIsEnabled()) ? output : PGPlaceholder();
                    objc_setAssociatedObject(strongView, &PGBypassKey, nil, OBJC_ASSOCIATION_ASSIGN);
                }
                __sync_sub_and_fetch(&PGOutstanding, 1);
            });
        }
    });
    return YES;
}

// Retry scans deferred by concurrency cap; stale reused views are ignored.
static void PGRetry(UIImageView *view, UIImage *source, NSObject *token, NSUInteger attempt) {
    if (attempt >= 30) return; // Fail closed if continuously overloaded.
    __weak UIImageView *weakView = view;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.20 * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
        UIImageView *strongView=weakView;
        if (strongView && objc_getAssociatedObject(strongView, &PGTokenKey) == token) {
            if (!PGSubmit(strongView, source, token)) PGRetry(strongView, source, token, attempt+1);
        }
    });
}
%hook UIImageView
// UIKit sometimes starts animations through animationImages, not image.
// Replace raw frames with a covered placeholder until the shared image pipeline
// has produced a fully censored animation.
- (void)setAnimationImages:(NSArray<UIImage *> *)images {
    if (objc_getAssociatedObject(self, &PGBypassKey) || !PGIsEnabled() || !images.count) {
        %orig; return;
    }
    %orig(nil); // Original animation frames must never play during scanning.
    if (images.count > 48) { self.image = PGPlaceholder(); return; }
    NSTimeInterval duration = self.animationDuration;
    if (duration <= 0) duration = (NSTimeInterval)images.count / 12.0;
    UIImage *animated = [UIImage animatedImageWithImages:images duration:duration];
    self.image = animated ?: PGPlaceholder();
}
- (void)startAnimating {
    if (PGIsEnabled() && objc_getAssociatedObject(self, &PGTokenKey)) return;
    %orig;
}
- (void)setImage:(UIImage *)image {
    if (objc_getAssociatedObject(self, &PGBypassKey)) { %orig; return; }
    // Invalidate any pending work when Photos reuses a view or sets a new image.
    objc_setAssociatedObject(self, &PGTokenKey, nil, OBJC_ASSOCIATION_ASSIGN);
    if (!PGIsEnabled()) { %orig; return; }
    if (image.images.count && !PGEligibleImage(image)) { %orig(PGPlaceholder()); return; }
    if (!PGEligibleImage(image)) {
        // A large unsupported/CI-backed photo must not silently bypass the filter.
        // Small toolbar icons remain untouched.
        if (image && image.size.width >= 96 && image.size.height >= 96) {
            %orig(PGPlaceholder());
        } else { %orig; }
        return;
    }
    if (!PGHasModel()) { %orig(PGPlaceholder()); return; }
    UIImage *alreadyCensored=objc_getAssociatedObject(image, &PGCachedKey);
    if (alreadyCensored) { %orig(alreadyCensored); return; }
    if (objc_getAssociatedObject(image, &PGCleanKey)) { %orig; return; }
    NSObject *token = [NSObject new];
    objc_setAssociatedObject(self, &PGTokenKey, token, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    if (!PGSubmit(self, image, token)) {
        PGRetry(self, image, token, 0);
    }
    // Prevent a flash of the original image while inference is queued.
    // Photos may still use other rendering layers not covered by this hook.
    %orig(PGPlaceholder());
}
%end
