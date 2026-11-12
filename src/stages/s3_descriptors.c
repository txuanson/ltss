#include "s3_descriptors.h"

// Box-Muller transform for pseudo-random Gaussian sampling
static float rand_gauss(float sigma) {
    static int have_spare = 0;
    static float spare;
    if (have_spare) {
        have_spare = 0;
        return spare * sigma;
    }
    have_spare = 1;
    float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float v = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float r = sqrtf(-2.0f * logf(u));
    float theta = 2.0f * (float)M_PI * v;
    spare = r * sinf(theta);
    return (r * cosf(theta)) * sigma;
}

BriefPair* generate_brief_pattern(int n_bits, int patch_size, unsigned int seed) {
    srand(seed);
    BriefPair* pattern = (BriefPair*)malloc(n_bits * sizeof(BriefPair));
    float sigma = (float)patch_size / 5.0f;
    float max_r = (float)((patch_size / 2 - 1) / 1.41421356);

    for (int i = 0; i < n_bits; ++i) {
        float x1 = roundf(rand_gauss(sigma));
        float y1 = roundf(rand_gauss(sigma));
        float x2 = roundf(rand_gauss(sigma));
        float y2 = roundf(rand_gauss(sigma));

        if (x1 < -max_r) { x1 = -max_r; }
        if (x1 > max_r) { x1 = max_r; }
        if (y1 < -max_r) { y1 = -max_r; }
        if (y1 > max_r) { y1 = max_r; }
        if (x2 < -max_r) { x2 = -max_r; }
        if (x2 > max_r) { x2 = max_r; }
        if (y2 < -max_r) { y2 = -max_r; }
        if (y2 > max_r) { y2 = max_r; }

        pattern[i].dx1 = x1;
        pattern[i].dy1 = y1;
        pattern[i].dx2 = x2;
        pattern[i].dy2 = y2;
    }
    return pattern;
}

void compute_ic_orientations(const ImageGray* src, KeypointList* kps, int patch_radius) {
    int w = src->width;
    int h = src->height;
    int r2 = patch_radius * patch_radius;

    for (int k = 0; k < kps->count; ++k) {
        int cx = (int)roundf(kps->points[k].x);
        int cy = (int)roundf(kps->points[k].y);

        float m10 = 0.0f;
        float m01 = 0.0f;

        for (int dy = -patch_radius; dy <= patch_radius; ++dy) {
            int sy = cy + dy;
            if (sy < 0 || sy >= h) continue;
            int dy2 = dy * dy;

            for (int dx = -patch_radius; dx <= patch_radius; ++dx) {
                if (dx * dx + dy2 > r2) continue;
                int sx = cx + dx;
                if (sx < 0 || sx >= w) continue;

                float intensity = src->data[sy * w + sx];
                m10 += (float)dx * intensity;
                m01 += (float)dy * intensity;
            }
        }
        kps->points[k].angle = atan2f(m01, m10);
    }
}

Descriptor256* extract_binary_descriptors(const ImageGray* src, const KeypointList* kps,
                                          const BriefPair* pattern, int n_bits, bool use_orientation) {
    int w = src->width;
    int h = src->height;
    Descriptor256* desc = (Descriptor256*)calloc(kps->count, sizeof(Descriptor256));

    for (int k = 0; k < kps->count; ++k) {
        int cx = (int)roundf(kps->points[k].x);
        int cy = (int)roundf(kps->points[k].y);
        float angle = use_orientation ? kps->points[k].angle : 0.0f;
        float cos_t = cosf(angle);
        float sin_t = sinf(angle);

        for (int b = 0; b < n_bits; ++b) {
            float dx1 = pattern[b].dx1;
            float dy1 = pattern[b].dy1;
            float dx2 = pattern[b].dx2;
            float dy2 = pattern[b].dy2;

            int rx1, ry1, rx2, ry2;
            if (use_orientation) {
                rx1 = (int)roundf(cos_t * dx1 - sin_t * dy1);
                ry1 = (int)roundf(sin_t * dx1 + cos_t * dy1);
                rx2 = (int)roundf(cos_t * dx2 - sin_t * dy2);
                ry2 = (int)roundf(sin_t * dx2 + cos_t * dy2);
            } else {
                rx1 = (int)dx1;
                ry1 = (int)dy1;
                rx2 = (int)dx2;
                ry2 = (int)dy2;
            }

            int p1x = cx + rx1; if (p1x < 0) p1x = 0; else if (p1x >= w) p1x = w - 1;
            int p1y = cy + ry1; if (p1y < 0) p1y = 0; else if (p1y >= h) p1y = h - 1;
            int p2x = cx + rx2; if (p2x < 0) p2x = 0; else if (p2x >= w) p2x = w - 1;
            int p2y = cy + ry2; if (p2y < 0) p2y = 0; else if (p2y >= h) p2y = h - 1;

            float val1 = src->data[p1y * w + p1x];
            float val2 = src->data[p2y * w + p2x];

            if (val1 < val2) {
                int byte_idx = b / 8;
                int bit_idx = 7 - (b % 8);
                desc[k].bytes[byte_idx] |= (uint8_t)(1 << bit_idx);
            }
        }
    }
    return desc;
}
