/*
 * Author: Tran Xuan Son (Student ID: 19127321) - Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * Stage 6: Warping & Blending (Inverse Homography, Bilinear Interpolation, Distance Feathering Blend)
 * External Sources: Matthew Brown & David G. Lowe, "Automatic Panoramic Image Stitching", IJCV 2007.
 */

#include "s6_warp_blend.h"

static void invert_homography(const Homography* H, double H_inv[3][3]) {
    double A[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) A[r][c] = H->m[r][c];
    }
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    double invdet = 1.0 / det;

    H_inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * invdet;
    H_inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * invdet;
    H_inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invdet;

    H_inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * invdet;
    H_inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invdet;
    H_inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * invdet;

    H_inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * invdet;
    H_inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * invdet;
    H_inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * invdet;
}

CanvasBounds compute_canvas_bounds(const ImageRGB* img_a, const ImageRGB* img_b, const Homography* H) {
    int wa = img_a->width, ha = img_a->height;
    int wb = img_b->width, hb = img_b->height;

    double corners_b[4][2] = {
        {0, 0}, {wb - 1.0, 0}, {wb - 1.0, hb - 1.0}, {0, hb - 1.0}
    };

    double min_x = 0.0, max_x = wa - 1.0;
    double min_y = 0.0, max_y = ha - 1.0;

    for (int i = 0; i < 4; ++i) {
        double bx = corners_b[i][0];
        double by = corners_b[i][1];
        double denom = H->m[2][0] * bx + H->m[2][1] * by + H->m[2][2];
        double px = (H->m[0][0] * bx + H->m[0][1] * by + H->m[0][2]) / denom;
        double py = (H->m[1][0] * bx + H->m[1][1] * by + H->m[1][2]) / denom;

        if (px < min_x) min_x = px;
        if (px > max_x) max_x = px;
        if (py < min_y) min_y = py;
        if (py > max_y) max_y = py;
    }

    CanvasBounds cb;
    cb.x_min = (int)floor(min_x);
    cb.y_min = (int)floor(min_y);
    cb.x_max = (int)ceil(max_x);
    cb.y_max = (int)ceil(max_y);
    cb.width = cb.x_max - cb.x_min + 1;
    cb.height = cb.y_max - cb.y_min + 1;
    return cb;
}

static inline float sample_feather_weight(int x, int y, int w, int h) {
    float wx = 1.0f - fabsf((float)x / (float)(w - 1) * 2.0f - 1.0f);
    float wy = 1.0f - fabsf((float)y / (float)(h - 1) * 2.0f - 1.0f);
    return fmaxf(1e-4f, wx * wy);
}

ImageRGB run_stage6_warp_blend(const ImageRGB* img_a, const ImageRGB* img_b,
                               const Homography* H, const char* interp, const char* blend) {
    CanvasBounds cb = compute_canvas_bounds(img_a, img_b, H);
    int out_w = cb.width;
    int out_h = cb.height;
    int off_x = -cb.x_min;
    int off_y = -cb.y_min;

    ImageRGB pano = create_image_rgb(out_w, out_h, 3);
    float* weight_a = (float*)calloc(out_w * out_h, sizeof(float));
    float* weight_b = (float*)calloc(out_w * out_h, sizeof(float));
    float* rgb_b_accum = (float*)calloc(out_w * out_h * 3, sizeof(float));

    int wa = img_a->width, ha = img_a->height;
    int wb = img_b->width, hb = img_b->height;

    // 1. Place Image A weights
    for (int y = 0; y < ha; ++y) {
        int py = y + off_y;
        for (int x = 0; x < wa; ++x) {
            int px = x + off_x;
            weight_a[py * out_w + px] = sample_feather_weight(x, y, wa, ha);
        }
    }

    // 2. Inverse warp Image B onto canvas
    double H_inv[3][3];
    invert_homography(H, H_inv);

    bool use_bilinear = (strcmp(interp, "bilinear") == 0);

    for (int cy = 0; cy < out_h; ++cy) {
        double world_y = cy + cb.y_min;
        for (int cx = 0; cx < out_w; ++cx) {
            double world_x = cx + cb.x_min;

            double denom = H_inv[2][0] * world_x + H_inv[2][1] * world_y + H_inv[2][2];
            if (denom <= 1e-6) continue;

            double src_x = (H_inv[0][0] * world_x + H_inv[0][1] * world_y + H_inv[0][2]) / denom;
            double src_y = (H_inv[1][0] * world_x + H_inv[1][1] * world_y + H_inv[1][2]) / denom;

            if (use_bilinear) {
                if (src_x >= 0.0 && src_x <= wb - 1.0 && src_y >= 0.0 && src_y <= hb - 1.0) {
                    int x0 = (int)floor(src_x);
                    int y0 = (int)floor(src_y);
                    int x1 = x0 + 1 < wb ? x0 + 1 : x0;
                    int y1 = y0 + 1 < hb ? y0 + 1 : y0;

                    float alpha = (float)(src_x - x0);
                    float beta  = (float)(src_y - y0);

                    float w00 = (1.0f - alpha) * (1.0f - beta);
                    float w10 = alpha * (1.0f - beta);
                    float w01 = (1.0f - alpha) * beta;
                    float w11 = alpha * beta;

                    int idx = (cy * out_w + cx) * 3;
                    for (int c = 0; c < 3; ++c) {
                        float v00 = img_b->data[(y0 * wb + x0) * 3 + c];
                        float v10 = img_b->data[(y0 * wb + x1) * 3 + c];
                        float v01 = img_b->data[(y1 * wb + x0) * 3 + c];
                        float v11 = img_b->data[(y1 * wb + x1) * 3 + c];
                        rgb_b_accum[idx + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                    }

                    weight_b[cy * out_w + cx] = sample_feather_weight((int)round(src_x), (int)round(src_y), wb, hb);
                }
            } else {
                // Nearest neighbor
                int rx = (int)round(src_x);
                int ry = (int)round(src_y);
                if (rx >= 0 && rx < wb && ry >= 0 && ry < hb) {
                    int idx = (cy * out_w + cx) * 3;
                    int src_idx = (ry * wb + rx) * 3;
                    rgb_b_accum[idx + 0] = img_b->data[src_idx + 0];
                    rgb_b_accum[idx + 1] = img_b->data[src_idx + 1];
                    rgb_b_accum[idx + 2] = img_b->data[src_idx + 2];
                    weight_b[cy * out_w + cx] = sample_feather_weight(rx, ry, wb, hb);
                }
            }
        }
    }

    // 3. Blend
    bool is_overwrite = (strcmp(blend, "overwrite") == 0);

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            int p_idx = y * out_w + x;
            float wa_val = weight_a[p_idx];
            float wb_val = weight_b[p_idx];

            float r_a = 0.0f, g_a = 0.0f, b_a = 0.0f;
            int local_ax = x - off_x;
            int local_ay = y - off_y;
            if (local_ax >= 0 && local_ax < wa && local_ay >= 0 && local_ay < ha) {
                int a_idx = (local_ay * wa + local_ax) * 3;
                r_a = img_a->data[a_idx + 0];
                g_a = img_a->data[a_idx + 1];
                b_a = img_a->data[a_idx + 2];
            }

            float r_b = rgb_b_accum[p_idx * 3 + 0];
            float g_b = rgb_b_accum[p_idx * 3 + 1];
            float b_b = rgb_b_accum[p_idx * 3 + 2];

            if (is_overwrite) {
                if (wa_val > 0.0f) {
                    pano.data[p_idx * 3 + 0] = (uint8_t)r_a;
                    pano.data[p_idx * 3 + 1] = (uint8_t)g_a;
                    pano.data[p_idx * 3 + 2] = (uint8_t)b_a;
                } else if (wb_val > 0.0f) {
                    pano.data[p_idx * 3 + 0] = (uint8_t)r_b;
                    pano.data[p_idx * 3 + 1] = (uint8_t)g_b;
                    pano.data[p_idx * 3 + 2] = (uint8_t)b_b;
                }
            } else {
                // Distance feathering alpha blend
                float total_w = wa_val + wb_val;
                if (total_w > 1e-6f) {
                    float a_weight = wa_val / total_w;
                    float b_weight = wb_val / total_w;

                    float final_r = a_weight * r_a + b_weight * r_b;
                    float final_g = a_weight * g_a + b_weight * g_b;
                    float final_b = a_weight * b_a + b_weight * b_b;

                    if (final_r < 0.0f) final_r = 0.0f; else if (final_r > 255.0f) final_r = 255.0f;
                    if (final_g < 0.0f) final_g = 0.0f; else if (final_g > 255.0f) final_g = 255.0f;
                    if (final_b < 0.0f) final_b = 0.0f; else if (final_b > 255.0f) final_b = 255.0f;

                    pano.data[p_idx * 3 + 0] = (uint8_t)final_r;
                    pano.data[p_idx * 3 + 1] = (uint8_t)final_g;
                    pano.data[p_idx * 3 + 2] = (uint8_t)final_b;
                }
            }
        }
    }

    free(weight_a);
    free(weight_b);
    free(rgb_b_accum);
    return pano;
}
