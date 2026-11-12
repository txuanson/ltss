#include "s5_homography.h"

// Hartley coordinate normalization
static void normalize_points(const float* pts, int n, double* norm_pts, double T[3][3]) {
    double cx = 0.0, cy = 0.0;
    for (int i = 0; i < n; ++i) {
        cx += pts[i * 2 + 0];
        cy += pts[i * 2 + 1];
    }
    cx /= n;
    cy /= n;

    double mean_dist = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = pts[i * 2 + 0] - cx;
        double dy = pts[i * 2 + 1] - cy;
        mean_dist += sqrt(dx * dx + dy * dy);
    }
    mean_dist /= n;
    double scale = mean_dist > 1e-8 ? sqrt(2.0) / mean_dist : 1.0;

    T[0][0] = scale; T[0][1] = 0.0;   T[0][2] = -scale * cx;
    T[1][0] = 0.0;   T[1][1] = scale; T[1][2] = -scale * cy;
    T[2][0] = 0.0;   T[2][1] = 0.0;   T[2][2] = 1.0;

    for (int i = 0; i < n; ++i) {
        norm_pts[i * 2 + 0] = (pts[i * 2 + 0] - cx) * scale;
        norm_pts[i * 2 + 1] = (pts[i * 2 + 1] - cy) * scale;
    }
}

// Invert 3x3 matrix
static bool invert_3x3(const double A[3][3], double inv[3][3]) {
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (fabs(det) < 1e-12) return false;
    double invdet = 1.0 / det;

    inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * invdet;
    inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * invdet;
    inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invdet;

    inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * invdet;
    inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invdet;
    inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * invdet;

    inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * invdet;
    inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * invdet;
    inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * invdet;
    return true;
}

// Multiply 3x3 matrices: C = A @ B
static void matmul_3x3(const double A[3][3], const double B[3][3], double C[3][3]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
        }
    }
}

// Direct 4-point linear system solution: H[2][2] = 1, solve 8x8 system via Gaussian elimination
static bool solve_dlt_4pt(const double* src, const double* dst, double H[3][3]) {
    double M[8][9];
    for (int i = 0; i < 4; ++i) {
        double x = src[i * 2 + 0];
        double y = src[i * 2 + 1];
        double u = dst[i * 2 + 0];
        double v = dst[i * 2 + 1];

        // u = (h0*x + h1*y + h2) / (h6*x + h7*y + 1)
        // h0*x + h1*y + h2 - u*x*h6 - u*y*h7 = u
        M[2 * i + 0][0] = x;
        M[2 * i + 0][1] = y;
        M[2 * i + 0][2] = 1.0;
        M[2 * i + 0][3] = 0.0;
        M[2 * i + 0][4] = 0.0;
        M[2 * i + 0][5] = 0.0;
        M[2 * i + 0][6] = -u * x;
        M[2 * i + 0][7] = -u * y;
        M[2 * i + 0][8] = u;

        // v = (h3*x + h4*y + h5) / (h6*x + h7*y + 1)
        // h3*x + h4*y + h5 - v*x*h6 - v*y*h7 = v
        M[2 * i + 1][0] = 0.0;
        M[2 * i + 1][1] = 0.0;
        M[2 * i + 1][2] = 0.0;
        M[2 * i + 1][3] = x;
        M[2 * i + 1][4] = y;
        M[2 * i + 1][5] = 1.0;
        M[2 * i + 1][6] = -v * x;
        M[2 * i + 1][7] = -v * y;
        M[2 * i + 1][8] = v;
    }

    // Gaussian elimination with partial pivoting
    for (int i = 0; i < 8; ++i) {
        int max_r = i;
        double max_val = fabs(M[i][i]);
        for (int r = i + 1; r < 8; ++r) {
            if (fabs(M[r][i]) > max_val) {
                max_val = fabs(M[r][i]);
                max_r = r;
            }
        }
        if (max_val < 1e-12) return false;

        if (max_r != i) {
            for (int c = 0; c <= 8; ++c) {
                double tmp = M[i][c];
                M[i][c] = M[max_r][c];
                M[max_r][c] = tmp;
            }
        }

        double pivot = M[i][i];
        for (int c = i; c <= 8; ++c) {
            M[i][c] /= pivot;
        }

        for (int r = 0; r < 8; ++r) {
            if (r == i) continue;
            double factor = M[r][i];
            for (int c = i; c <= 8; ++c) {
                M[r][c] -= factor * M[i][c];
            }
        }
    }

    H[0][0] = M[0][8]; H[0][1] = M[1][8]; H[0][2] = M[2][8];
    H[1][0] = M[3][8]; H[1][1] = M[4][8]; H[1][2] = M[5][8];
    H[2][0] = M[6][8]; H[2][1] = M[7][8]; H[2][2] = 1.0;
    return true;
}

bool solve_dlt_homography(const float* src_pts, const float* dst_pts, int n, Homography* out_h) {
    if (n < 4) return false;

    double* norm_src = (double*)malloc(n * 2 * sizeof(double));
    double* norm_dst = (double*)malloc(n * 2 * sizeof(double));
    double T_src[3][3], T_dst[3][3], T_dst_inv[3][3];

    normalize_points(src_pts, n, norm_src, T_src);
    normalize_points(dst_pts, n, norm_dst, T_dst);

    if (!invert_3x3(T_dst, T_dst_inv)) {
        free(norm_src); free(norm_dst);
        return false;
    }

    double H_norm[3][3];
    bool ok = solve_dlt_4pt(norm_src, norm_dst, H_norm);
    free(norm_src); free(norm_dst);
    if (!ok) return false;

    // Denormalize: H = inv(T_dst) @ H_norm @ T_src
    double tmp[3][3];
    matmul_3x3(T_dst_inv, H_norm, tmp);
    matmul_3x3(tmp, T_src, out_h->m);

    if (fabs(out_h->m[2][2]) > 1e-8) {
        double d = 1.0 / out_h->m[2][2];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) out_h->m[r][c] *= d;
        }
    }
    return true;
}

static bool is_degenerate_4pt(const float* pts) {
    // Check if any 3 points are collinear
    int triples[4][3] = {{0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}};
    for (int t = 0; t < 4; ++t) {
        int i = triples[t][0], j = triples[t][1], k = triples[t][2];
        float dx1 = pts[j * 2 + 0] - pts[i * 2 + 0];
        float dy1 = pts[j * 2 + 1] - pts[i * 2 + 1];
        float dx2 = pts[k * 2 + 0] - pts[i * 2 + 0];
        float dy2 = pts[k * 2 + 1] - pts[i * 2 + 1];
        float area2 = fabsf(dx1 * dy2 - dy1 * dx2);
        if (area2 < 1e-2f) return true;
    }
    return false;
}

bool estimate_homography_ransac(const KeypointList* kp_a, const KeypointList* kp_b,
                                const MatchList* matches, int max_iterations,
                                float inlier_threshold, Homography* out_h,
                                bool* out_inlier_mask, int* out_inlier_count) {
    int m = matches->count;
    if (m < 4) return false;

    float* pts_a = (float*)malloc(m * 2 * sizeof(float));
    float* pts_b = (float*)malloc(m * 2 * sizeof(float));
    for (int i = 0; i < m; ++i) {
        pts_a[i * 2 + 0] = kp_a->points[matches->matches[i].idx_a].x;
        pts_a[i * 2 + 1] = kp_a->points[matches->matches[i].idx_a].y;
        pts_b[i * 2 + 0] = kp_b->points[matches->matches[i].idx_b].x;
        pts_b[i * 2 + 1] = kp_b->points[matches->matches[i].idx_b].y;
    }

    int best_inliers = 0;
    Homography best_h;
    bool* best_mask = (bool*)calloc(m, sizeof(bool));
    bool* cur_mask  = (bool*)calloc(m, sizeof(bool));

    srand(42);
    for (int it = 0; it < max_iterations; ++it) {
        // Sample 4 distinct random matches
        int sample[4];
        for (int i = 0; i < 4; ++i) {
            while (1) {
                int r = rand() % m;
                bool exists = false;
                for (int j = 0; j < i; ++j) {
                    if (sample[j] == r) { exists = true; break; }
                }
                if (!exists) { sample[i] = r; break; }
            }
        }

        float s_pts_b[8], s_pts_a[8];
        for (int i = 0; i < 4; ++i) {
            s_pts_b[i * 2 + 0] = pts_b[sample[i] * 2 + 0];
            s_pts_b[i * 2 + 1] = pts_b[sample[i] * 2 + 1];
            s_pts_a[i * 2 + 0] = pts_a[sample[i] * 2 + 0];
            s_pts_a[i * 2 + 1] = pts_a[sample[i] * 2 + 1];
        }

        if (is_degenerate_4pt(s_pts_b) || is_degenerate_4pt(s_pts_a)) continue;

        Homography h_cand;
        if (!solve_dlt_homography(s_pts_b, s_pts_a, 4, &h_cand)) continue;

        // Count inliers
        int inliers = 0;
        for (int i = 0; i < m; ++i) {
            double bx = pts_b[i * 2 + 0];
            double by = pts_b[i * 2 + 1];
            double denom = h_cand.m[2][0] * bx + h_cand.m[2][1] * by + h_cand.m[2][2];
            if (denom <= 1e-6) {
                cur_mask[i] = false;
                continue;
            }
            double proj_x = (h_cand.m[0][0] * bx + h_cand.m[0][1] * by + h_cand.m[0][2]) / denom;
            double proj_y = (h_cand.m[1][0] * bx + h_cand.m[1][1] * by + h_cand.m[1][2]) / denom;

            double dx = proj_x - pts_a[i * 2 + 0];
            double dy = proj_y - pts_a[i * 2 + 1];
            double err = sqrt(dx * dx + dy * dy);

            if (err < inlier_threshold) {
                cur_mask[i] = true;
                inliers++;
            } else {
                cur_mask[i] = false;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_h = h_cand;
            memcpy(best_mask, cur_mask, m * sizeof(bool));
        }
    }

    *out_h = best_h;
    *out_inlier_count = best_inliers;
    if (out_inlier_mask != NULL) {
        memcpy(out_inlier_mask, best_mask, m * sizeof(bool));
    }

    free(pts_a);
    free(pts_b);
    free(best_mask);
    free(cur_mask);
    return best_inliers >= 4;
}
