#ifndef S5_HOMOGRAPHY_H
#define S5_HOMOGRAPHY_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Solve 4-point (or N-point) DLT homography mapping src -> dst
bool solve_dlt_homography(const float* src_pts, const float* dst_pts, int n, Homography* out_h);

// RANSAC homography estimation
bool estimate_homography_ransac(const KeypointList* kp_a, const KeypointList* kp_b,
                                const MatchList* matches, int max_iterations,
                                float inlier_threshold, Homography* out_h,
                                bool* out_inlier_mask, int* out_inlier_count);

#ifdef __cplusplus
}
#endif

#endif // S5_HOMOGRAPHY_H
