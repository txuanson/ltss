#ifndef S1_PREPROCESS_H
#define S1_PREPROCESS_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Convert RGB to float grayscale: Y = 0.299*R + 0.587*G + 0.114*B
ImageGray rgb_to_grayscale(const ImageRGB* rgb);

// Separable 1D Gaussian kernel generation
float* make_gaussian_kernel_1d(int kernel_size, float sigma);

// Convolve 2D image with separable 1D filter (edge replication mode)
ImageGray convolve_separable_2d(const ImageGray* src, const float* kernel_1d, int ksize);

// Full 2D convolution (for comparison)
ImageGray convolve_full_2d(const ImageGray* src, int ksize, float sigma);

// Run Stage 1 on an RGB image
ImageGray run_stage1_preprocess(const ImageRGB* img, const char* method, int ksize, float sigma);

#ifdef __cplusplus
}
#endif

#endif // S1_PREPROCESS_H
