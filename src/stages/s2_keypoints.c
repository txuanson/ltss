/*
 * Author: Tran Xuan Son (Student ID: 19127321) - Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * Stage 2: Keypoint Detection (Sobel Gradients, Harris Corner Response, 7x7 NMS, Top-K)
 * External Sources: C. Harris and M. Stephens, "A Combined Corner and Edge Detector", 1988.
 */

#include "s2_keypoints.h"
#include "s1_preprocess.h"

// 16-pixel Bresenham circle offsets
static const int FAST_DX[16] = { 0,  1,  2,  3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
static const int FAST_DY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3,  3,  2,  1,  0, -1, -2, -3 };

void compute_sobel_gradients(const ImageGray* src, ImageGray* ix, ImageGray* iy) {
    int w = src->width;
    int h = src->height;

    // Sobel X: [-1 0 1] horizontal, [1 2 1]^T vertical
    // Sobel Y: [1 2 1] horizontal, [-1 0 1]^T vertical
    for (int y = 0; y < h; ++y) {
        int ym1 = y > 0 ? y - 1 : 0;
        int yp1 = y < h - 1 ? y + 1 : h - 1;
        for (int x = 0; x < w; ++x) {
            int xm1 = x > 0 ? x - 1 : 0;
            int xp1 = x < w - 1 ? x + 1 : w - 1;

            float p00 = src->data[ym1 * w + xm1];
            float p01 = src->data[ym1 * w + x];
            float p02 = src->data[ym1 * w + xp1];

            float p10 = src->data[y * w + xm1];
            float p12 = src->data[y * w + xp1];

            float p20 = src->data[yp1 * w + xm1];
            float p21 = src->data[yp1 * w + x];
            float p22 = src->data[yp1 * w + xp1];

            float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
            float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

            ix->data[y * w + x] = gx * 0.125f;
            iy->data[y * w + x] = gy * 0.125f;
        }
    }
}

static void compute_structure_tensor(const ImageGray* src, int window_size, float sigma,
                                     ImageGray* sxx, ImageGray* syy, ImageGray* sxy) {
    int w = src->width;
    int h = src->height;
    int total = w * h;

    ImageGray ix = create_image_gray(w, h);
    ImageGray iy = create_image_gray(w, h);
    compute_sobel_gradients(src, &ix, &iy);

    ImageGray ixx = create_image_gray(w, h);
    ImageGray iyy = create_image_gray(w, h);
    ImageGray temp_ixy = create_image_gray(w, h);

    for (int i = 0; i < total; ++i) {
        float x = ix.data[i];
        float y = iy.data[i];
        ixx.data[i] = x * x;
        iyy.data[i] = y * y;
        temp_ixy.data[i] = x * y;
    }

    free_image_gray(&ix);
    free_image_gray(&iy);

    float* k1d = make_gaussian_kernel_1d(window_size, sigma);
    *sxx = convolve_separable_2d(&ixx, k1d, window_size);
    *syy = convolve_separable_2d(&iyy, k1d, window_size);
    *sxy = convolve_separable_2d(&temp_ixy, k1d, window_size);

    free(k1d);
    free_image_gray(&ixx);
    free_image_gray(&iyy);
    free_image_gray(&temp_ixy);
}

ImageGray compute_harris_response(const ImageGray* src, float k_param, int window_size, float sigma) {
    int w = src->width;
    int h = src->height;
    int total = w * h;

    ImageGray sxx, syy, sxy;
    compute_structure_tensor(src, window_size, sigma, &sxx, &syy, &sxy);

    ImageGray resp = create_image_gray(w, h);
    for (int i = 0; i < total; ++i) {
        float xx = sxx.data[i];
        float yy = syy.data[i];
        float xy = sxy.data[i];
        float det = xx * yy - xy * xy;
        float trace = xx + yy;
        resp.data[i] = det - k_param * (trace * trace);
    }

    free_image_gray(&sxx);
    free_image_gray(&syy);
    free_image_gray(&sxy);
    return resp;
}

ImageGray compute_shi_tomasi_response(const ImageGray* src, int window_size, float sigma) {
    int w = src->width;
    int h = src->height;
    int total = w * h;

    ImageGray sxx, syy, sxy;
    compute_structure_tensor(src, window_size, sigma, &sxx, &syy, &sxy);

    ImageGray resp = create_image_gray(w, h);
    for (int i = 0; i < total; ++i) {
        float xx = sxx.data[i];
        float yy = syy.data[i];
        float xy = sxy.data[i];
        float trace = xx + yy;
        float diff = xx - yy;
        float disc = sqrtf(fmaxf(0.0f, diff * diff + 4.0f * xy * xy));
        resp.data[i] = 0.5f * (trace - disc); // Minimum eigenvalue
    }

    free_image_gray(&sxx);
    free_image_gray(&syy);
    free_image_gray(&sxy);
    return resp;
}

ImageGray compute_fast_response(const ImageGray* src, float threshold, int n_contiguous) {
    int w = src->width;
    int h = src->height;
    ImageGray resp = create_image_gray(w, h);

    for (int y = 3; y < h - 3; ++y) {
        for (int x = 3; x < w - 3; ++x) {
            float center = src->data[y * w + x];

            // Quick rejection: cardinal points 0, 4, 8, 12
            float p0  = src->data[(y + FAST_DY[0])  * w + (x + FAST_DX[0])];
            float p4  = src->data[(y + FAST_DY[4])  * w + (x + FAST_DX[4])];
            float p8  = src->data[(y + FAST_DY[8])  * w + (x + FAST_DX[8])];
            float p12 = src->data[(y + FAST_DY[12]) * w + (x + FAST_DX[12])];

            int cbright = (p0 - center > threshold) + (p4 - center > threshold) +
                          (p8 - center > threshold) + (p12 - center > threshold);
            int cdark   = (center - p0 > threshold) + (center - p4 > threshold) +
                          (center - p8 > threshold) + (center - p12 > threshold);

            if (cbright < 2 && cdark < 2) continue;

            // Full circular arc test
            int bright_run = 0, max_bright = 0;
            int dark_run = 0, max_dark = 0;
            float sum_bright = 0.0f, sum_dark = 0.0f;

            for (int k = 0; k < 16 + n_contiguous - 1; ++k) {
                int idx = k % 16;
                float val = src->data[(y + FAST_DY[idx]) * w + (x + FAST_DX[idx])];
                float diff = val - center;

                if (diff > threshold) {
                    bright_run++;
                    if (bright_run > max_bright) max_bright = bright_run;
                    if (k < 16) sum_bright += (diff - threshold);
                } else {
                    bright_run = 0;
                }

                if (diff < -threshold) {
                    dark_run++;
                    if (dark_run > max_dark) max_dark = dark_run;
                    if (k < 16) sum_dark += (-diff - threshold);
                } else {
                    dark_run = 0;
                }
            }

            if (max_bright >= n_contiguous || max_dark >= n_contiguous) {
                resp.data[y * w + x] = sum_bright > sum_dark ? sum_bright : sum_dark;
            }
        }
    }
    return resp;
}

// Comparator for qsort descending by score
static int compare_keypoints_desc(const void* a, const void* b) {
    const Keypoint* ka = (const Keypoint*)a;
    const Keypoint* kb = (const Keypoint*)b;
    if (kb->score > ka->score) return 1;
    if (kb->score < ka->score) return -1;
    return 0;
}

KeypointList non_maximum_suppression(const ImageGray* response, int nms_window,
                                    float threshold_rel, int border_margin, int max_keypoints) {
    int w = response->width;
    int h = response->height;
    int radius = nms_window / 2;

    float max_score = 0.0f;
    for (int i = 0; i < w * h; ++i) {
        if (response->data[i] > max_score) max_score = response->data[i];
    }
    float abs_thresh = max_score * threshold_rel;

    KeypointList candidates = create_keypoint_list(2048);

    for (int y = border_margin; y < h - border_margin; ++y) {
        for (int x = border_margin; x < w - border_margin; ++x) {
            float val = response->data[y * w + x];
            if (val <= abs_thresh) continue;

            bool is_max = true;
            for (int dy = -radius; dy <= radius && is_max; ++dy) {
                int sy = y + dy;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = x + dx;
                    if (response->data[sy * w + sx] > val) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                Keypoint kp;
                kp.x = (float)x;
                kp.y = (float)y;
                kp.score = val;
                kp.angle = 0.0f;
                add_keypoint(&candidates, kp);
            }
        }
    }

    // Sort descending by response score
    if (candidates.count > 1) {
        qsort(candidates.points, candidates.count, sizeof(Keypoint), compare_keypoints_desc);
    }

    if (max_keypoints > 0 && candidates.count > max_keypoints) {
        candidates.count = max_keypoints;
    }

    return candidates;
}

KeypointList run_stage2_keypoints(const ImageGray* blurred, const char* detector,
                                 int max_keypoints, ImageGray* out_response) {
    ImageGray resp;
    if (strcmp(detector, "harris") == 0) {
        resp = compute_harris_response(blurred, 0.04f, 5, 1.5f);
    } else if (strcmp(detector, "shi_tomasi") == 0) {
        resp = compute_shi_tomasi_response(blurred, 5, 1.5f);
    } else if (strcmp(detector, "fast") == 0) {
        resp = compute_fast_response(blurred, 20.0f, 9);
    } else {
        resp = compute_harris_response(blurred, 0.04f, 5, 1.5f);
    }

    KeypointList kps = non_maximum_suppression(&resp, 7, 0.01f, 16, max_keypoints);
    if (out_response != NULL) {
        *out_response = resp;
    } else {
        free_image_gray(&resp);
    }
    return kps;
}
