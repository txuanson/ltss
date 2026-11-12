#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// RGB uint8 image buffer
typedef struct {
    int width;
    int height;
    int channels; // Typically 3 (RGB)
    uint8_t* data;
} ImageRGB;

// Grayscale float32 image buffer
typedef struct {
    int width;
    int height;
    float* data;
} ImageGray;

// 2D Keypoint with corner response score
typedef struct {
    float x;
    float y;
    float score;
    float angle; // Orientation in radians (for ORB)
} Keypoint;

// Keypoint list
typedef struct {
    int count;
    int capacity;
    Keypoint* points;
} KeypointList;

// 256-bit binary descriptor (32 uint8 bytes)
typedef struct {
    uint8_t bytes[32];
} Descriptor256;

// Keypoint match pair
typedef struct {
    int idx_a;
    int idx_b;
    float distance;
} Match;

// Match list
typedef struct {
    int count;
    int capacity;
    Match* matches;
} MatchList;

// 3x3 projective transformation matrix (Homography)
typedef struct {
    double m[3][3];
} Homography;

// Memory helpers
ImageRGB create_image_rgb(int width, int height, int channels);
void free_image_rgb(ImageRGB* img);

ImageGray create_image_gray(int width, int height);
void free_image_gray(ImageGray* img);

KeypointList create_keypoint_list(int capacity);
void free_keypoint_list(KeypointList* list);
void add_keypoint(KeypointList* list, Keypoint kp);

MatchList create_match_list(int capacity);
void free_match_list(MatchList* list);
void add_match(MatchList* list, Match m);

// Helper for saving a grayscale float image to PNG for debugging
void save_gray_png(const ImageGray* img, const char* filepath);
void save_heatmap_png(const ImageGray* resp, const char* filepath);
void draw_keypoints_png(const ImageRGB* src, const KeypointList* kps, const char* filepath);
void draw_matches_png(const ImageRGB* img_a, const KeypointList* kp_a,
                      const ImageRGB* img_b, const KeypointList* kp_b,
                      const MatchList* matches, const bool* inlier_mask,
                      const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
