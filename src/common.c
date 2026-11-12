#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

ImageRGB create_image_rgb(int width, int height, int channels) {
    ImageRGB img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = (uint8_t*)calloc(width * height * channels, sizeof(uint8_t));
    return img;
}

void free_image_rgb(ImageRGB* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

ImageGray create_image_gray(int width, int height) {
    ImageGray img;
    img.width = width;
    img.height = height;
    img.data = (float*)calloc(width * height, sizeof(float));
    return img;
}

void free_image_gray(ImageGray* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

KeypointList create_keypoint_list(int capacity) {
    KeypointList list;
    list.count = 0;
    list.capacity = capacity > 0 ? capacity : 128;
    list.points = (Keypoint*)malloc(list.capacity * sizeof(Keypoint));
    return list;
}

void free_keypoint_list(KeypointList* list) {
    if (list && list->points) {
        free(list->points);
        list->points = NULL;
        list->count = 0;
        list->capacity = 0;
    }
}

void add_keypoint(KeypointList* list, Keypoint kp) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->points = (Keypoint*)realloc(list->points, list->capacity * sizeof(Keypoint));
    }
    list->points[list->count++] = kp;
}

MatchList create_match_list(int capacity) {
    MatchList list;
    list.count = 0;
    list.capacity = capacity > 0 ? capacity : 128;
    list.matches = (Match*)malloc(list.capacity * sizeof(Match));
    return list;
}

void free_match_list(MatchList* list) {
    if (list && list->matches) {
        free(list->matches);
        list->matches = NULL;
        list->count = 0;
        list->capacity = 0;
    }
}

void add_match(MatchList* list, Match m) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->matches = (Match*)realloc(list->matches, list->capacity * sizeof(Match));
    }
    list->matches[list->count++] = m;
}

void save_gray_png(const ImageGray* img, const char* filepath) {
    int total = img->width * img->height;
    uint8_t* u8 = (uint8_t*)malloc(total);
    for (int i = 0; i < total; ++i) {
        float val = img->data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        u8[i] = (uint8_t)val;
    }
    stbi_write_png(filepath, img->width, img->height, 1, u8, img->width);
    free(u8);
}

void save_heatmap_png(const ImageGray* resp, const char* filepath) {
    int w = resp->width;
    int h = resp->height;
    int total = w * h;
    float max_val = 0.0f;
    for (int i = 0; i < total; ++i) {
        if (resp->data[i] > max_val) max_val = resp->data[i];
    }

    uint8_t* rgb = (uint8_t*)malloc(total * 3);
    for (int i = 0; i < total; ++i) {
        float val = resp->data[i];
        float norm = 0.0f;
        if (val > 0.0f && max_val > 1e-12f) {
            norm = powf(val / max_val, 0.35f); // Gamma boost
        }
        float r = norm * 3.0f;
        float g = norm * 3.0f - 1.0f;
        float b = norm * 3.0f - 2.0f;
        if (r < 0.0f) r = 0.0f; else if (r > 1.0f) r = 1.0f;
        if (g < 0.0f) g = 0.0f; else if (g > 1.0f) g = 1.0f;
        if (b < 0.0f) b = 0.0f; else if (b > 1.0f) b = 1.0f;

        rgb[i * 3 + 0] = (uint8_t)(r * 255.0f);
        rgb[i * 3 + 1] = (uint8_t)(g * 255.0f);
        rgb[i * 3 + 2] = (uint8_t)(b * 255.0f);
    }
    stbi_write_png(filepath, w, h, 3, rgb, w * 3);
    free(rgb);
}

void draw_keypoints_png(const ImageRGB* src, const KeypointList* kps, const char* filepath) {
    int w = src->width;
    int h = src->height;
    ImageRGB out = create_image_rgb(w, h, 3);
    memcpy(out.data, src->data, w * h * 3);

    for (int k = 0; k < kps->count; ++k) {
        int cx = (int)roundf(kps->points[k].x);
        int cy = (int)roundf(kps->points[k].y);
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (abs(dx) + abs(dy) <= 3) {
                    int px = cx + dx;
                    int py = cy + dy;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        int idx = (py * w + px) * 3;
                        out.data[idx + 0] = 0;   // R
                        out.data[idx + 1] = 255; // G
                        out.data[idx + 2] = 0;   // B
                    }
                }
            }
        }
    }
    stbi_write_png(filepath, w, h, 3, out.data, w * 3);
    free_image_rgb(&out);
}

// Simple Bresenham line drawing for visual matches
static void draw_line(uint8_t* data, int stride, int width, int height,
                      int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;

    while (1) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int idx = y0 * stride + x0 * 3;
            data[idx + 0] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = err;
        if (e2 > -dx) { err -= dy; x0 += sx; }
        if (e2 < dy)  { err += dx; y0 += sy; }
    }
}

void draw_matches_png(const ImageRGB* img_a, const KeypointList* kp_a,
                      const ImageRGB* img_b, const KeypointList* kp_b,
                      const MatchList* matches, const bool* inlier_mask,
                      const char* filepath) {
    int ha = img_a->height, wa = img_a->width;
    int hb = img_b->height, wb = img_b->width;
    int out_h = ha > hb ? ha : hb;
    int out_w = wa + wb;
    int stride = out_w * 3;

    uint8_t* canvas = (uint8_t*)calloc(out_w * out_h * 3, sizeof(uint8_t));

    // Blit img_a
    for (int y = 0; y < ha; ++y) {
        memcpy(canvas + y * stride, img_a->data + y * wa * 3, wa * 3);
    }
    // Blit img_b
    for (int y = 0; y < hb; ++y) {
        memcpy(canvas + y * stride + wa * 3, img_b->data + y * wb * 3, wb * 3);
    }

    // Draw matches
    for (int i = 0; i < matches->count; ++i) {
        int ia = matches->matches[i].idx_a;
        int ib = matches->matches[i].idx_b;
        if (ia < 0 || ia >= kp_a->count || ib < 0 || ib >= kp_b->count) continue;

        int xa = (int)roundf(kp_a->points[ia].x);
        int ya = (int)roundf(kp_a->points[ia].y);
        int xb = (int)roundf(kp_b->points[ib].x) + wa;
        int yb = (int)roundf(kp_b->points[ib].y);

        uint8_t r = 220, g = 50, b = 50; // Red for outlier / generic
        if (inlier_mask != NULL && inlier_mask[i]) {
            r = 0; g = 230; b = 70; // Green for inliers
        } else if (inlier_mask == NULL) {
            r = (uint8_t)((i * 73) % 200 + 55);
            g = (uint8_t)((i * 151) % 200 + 55);
            b = (uint8_t)((i * 223) % 200 + 55);
        }

        draw_line(canvas, stride, out_w, out_h, xa, ya, xb, yb, r, g, b);
    }

    stbi_write_png(filepath, out_w, out_h, 3, canvas, stride);
    free(canvas);
}
