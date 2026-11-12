#ifndef S2_KEYPOINTS_H
#define S2_KEYPOINTS_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gradient & Structure Tensor computation
void compute_sobel_gradients(const ImageGray* src, ImageGray* ix, ImageGray* iy);

// Corner response detectors
ImageGray compute_harris_response(const ImageGray* src, float k_param, int window_size, float sigma);
ImageGray compute_shi_tomasi_response(const ImageGray* src, int window_size, float sigma);
ImageGray compute_fast_response(const ImageGray* src, float threshold, int n_contiguous);

// Non-Maximum Suppression (NMS) and Top-K Keypoint Selection
KeypointList non_maximum_suppression(const ImageGray* response, int nms_window,
                                    float threshold_rel, int border_margin, int max_keypoints);

// Stage 2 execution interface
KeypointList run_stage2_keypoints(const ImageGray* blurred, const char* detector,
                                 int max_keypoints, ImageGray* out_response);

#ifdef __cplusplus
}
#endif

#endif // S2_KEYPOINTS_H
