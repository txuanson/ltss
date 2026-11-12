/*
 * Author: Tran Xuan Son (Student ID: 19127321)
 * Group: Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * External Sources: stb_image and stb_image_write by Sean Barrett (Public Domain / MIT).
 */

#include "common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

extern "C" {

ImageRGB load_image(const char* filepath) {
    int w = 0, h = 0, channels = 0;
    uint8_t* data = stbi_load(filepath, &w, &h, &channels, 3);
    if (!data) {
        fprintf(stderr, "Error: Could not load image from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    ImageRGB img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data = data;
    return img;
}

void save_image_png(const ImageRGB* img, const char* filepath) {
    if (!img || !img->data) return;
    stbi_write_png(filepath, img->width, img->height, img->channels, img->data, img->width * img->channels);
}

}
