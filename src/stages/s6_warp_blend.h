#ifndef S6_WARP_BLEND_H
#define S6_WARP_BLEND_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bounding box of warped panorama
typedef struct {
    int x_min, y_min;
    int x_max, y_max;
    int width, height;
} CanvasBounds;

// Compute canvas bounds by projecting corners of image B via H
CanvasBounds compute_canvas_bounds(const ImageRGB* img_a, const ImageRGB* img_b, const Homography* H);

// Inverse warping and blending
ImageRGB run_stage6_warp_blend(const ImageRGB* img_a, const ImageRGB* img_b,
                               const Homography* H, const char* interp, const char* blend);

#ifdef __cplusplus
}
#endif

#endif // S6_WARP_BLEND_H
