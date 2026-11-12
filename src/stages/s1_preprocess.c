#include "s1_preprocess.h"

ImageGray rgb_to_grayscale(const ImageRGB* rgb) {
    int w = rgb->width;
    int h = rgb->height;
    ImageGray gray = create_image_gray(w, h);
    int total = w * h;
    for (int i = 0; i < total; ++i) {
        float r = (float)rgb->data[i * 3 + 0];
        float g = (float)rgb->data[i * 3 + 1];
        float b = (float)rgb->data[i * 3 + 2];
        gray.data[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return gray;
}

float* make_gaussian_kernel_1d(int kernel_size, float sigma) {
    if (kernel_size % 2 == 0) kernel_size += 1;
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    int radius = kernel_size / 2;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        int x = i - radius;
        float val = expf(-(float)(x * x) / (2.0f * sigma * sigma));
        kernel[i] = val;
        sum += val;
    }
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }
    return kernel;
}

ImageGray convolve_separable_2d(const ImageGray* src, const float* kernel_1d, int ksize) {
    int w = src->width;
    int h = src->height;
    int radius = ksize / 2;

    // Buffer for intermediate horizontal pass
    float* temp = (float*)malloc(w * h * sizeof(float));

    // 1. Horizontal pass with edge clamping
    for (int y = 0; y < h; ++y) {
        int row_offset = y * w;
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int sx = x + (k - radius);
                if (sx < 0) sx = 0;
                else if (sx >= w) sx = w - 1;
                sum += kernel_1d[k] * src->data[row_offset + sx];
            }
            temp[row_offset + x] = sum;
        }
    }

    // 2. Vertical pass
    ImageGray out = create_image_gray(w, h);
    for (int y = 0; y < h; ++y) {
        int row_offset = y * w;
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int sy = y + (k - radius);
                if (sy < 0) sy = 0;
                else if (sy >= h) sy = h - 1;
                sum += kernel_1d[k] * temp[sy * w + x];
            }
            out.data[row_offset + x] = sum;
        }
    }

    free(temp);
    return out;
}

ImageGray run_stage1_preprocess(const ImageRGB* img, const char* method, int ksize, float sigma) {
    ImageGray gray = rgb_to_grayscale(img);
    if (strcmp(method, "none") == 0 || ksize <= 1) {
        return gray;
    }
    float* k1d = make_gaussian_kernel_1d(ksize, sigma);
    ImageGray blurred = convolve_separable_2d(&gray, k1d, ksize);
    free(k1d);
    free_image_gray(&gray);
    return blurred;
}
