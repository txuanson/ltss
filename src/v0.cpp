/*
 * Author: Tran Xuan Son (Student ID: 19127321)
 * Group: Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * Purpose: Sequential V0 Reference Implementation (Single-Threaded CPU).
 *          Strictly pure C/C++ without multithreading (no OpenMP, no std::thread)
 *          and without handwritten SIMD intrinsics for a fair baseline.
 */

#include "versions.h"
#include "stages/s1_preprocess.h"
#include "stages/s2_keypoints.h"
#include "stages/s3_descriptors.h"
#include "stages/s4_matching.h"
#include "stages/s5_homography.h"
#include "stages/s6_warp_blend.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

static void ensure_dir_v0(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
}

int run_v0_sequential(const ImageRGB& img_a, const ImageRGB& img_b,
                       const PipelineConfig& config, PipelineSummary& summary,
                       ImageRGB& final_pano) {
    int runs = config.runs;
    bool save_intermediate = config.save_intermediate;
    std::string out_dir = config.out_dir;

    ensure_dir_v0("data");
    ensure_dir_v0("data/output");
    ensure_dir_v0(out_dir);

    std::string sub_s1 = out_dir + "/stage1_preprocess";
    std::string sub_s2 = out_dir + "/stage2_keypoints";
    std::string sub_s3 = out_dir + "/stage3_descriptors";
    std::string sub_s4 = out_dir + "/stage4_matching";
    std::string sub_s5 = out_dir + "/stage5_homography";
    std::string sub_s6 = out_dir + "/stage6_warp_blend";

    if (save_intermediate) {
        ensure_dir_v0(sub_s1.c_str());
        ensure_dir_v0(sub_s2.c_str());
        ensure_dir_v0(sub_s3.c_str());
        ensure_dir_v0(sub_s4.c_str());
        ensure_dir_v0(sub_s5.c_str());
        ensure_dir_v0(sub_s6.c_str());
    }

    BriefPair* pattern = generate_brief_pattern(256, 31, 42);

    // 1. Untimed warm-up run per Section 4.2
    {
        ImageGray b_a = run_stage1_preprocess(&img_a, "separable", 5, 1.4f);
        ImageGray b_b = run_stage1_preprocess(&img_b, "separable", 5, 1.4f);
        KeypointList k_a = run_stage2_keypoints(&b_a, "harris", 1500, nullptr);
        KeypointList k_b = run_stage2_keypoints(&b_b, "harris", 1500, nullptr);
        compute_ic_orientations(&b_a, &k_a, 15);
        compute_ic_orientations(&b_b, &k_b, 15);
        Descriptor256* d_a = extract_binary_descriptors(&b_a, &k_a, pattern, 256, true);
        Descriptor256* d_b = extract_binary_descriptors(&b_b, &k_b, pattern, 256, true);
        MatchList m = match_descriptors_hamming(d_a, k_a.count, d_b, k_b.count, 0.75f, true);
        Homography H_dummy;
        bool* mask = (bool*)calloc(m.count, sizeof(bool));
        int inliers = 0;
        estimate_homography_ransac(&k_a, &k_b, &m, 2000, 3.5f, &H_dummy, mask, &inliers);
        ImageRGB pano = run_stage6_warp_blend(&img_a, &img_b, &H_dummy, "bilinear", "feathering");

        free_image_rgb(&pano);
        free(mask);
        free_match_list(&m);
        free(d_a);
        free(d_b);
        free_keypoint_list(&k_a);
        free_keypoint_list(&k_b);
        free_image_gray(&b_a);
        free_image_gray(&b_b);
    }

    // 2. Timed runs
    std::vector<StageTimings> all_timings;
    all_timings.reserve(runs);

    int final_inliers = 0;
    int final_matches = 0;
    int num_kps_a = 0, num_kps_b = 0;

    for (int r = 0; r < runs; ++r) {
        bool is_last = (r == runs - 1);
        StageTimings st;

        auto t_e2e_start = std::chrono::high_resolution_clock::now();

        // Stage 1: Preprocessing
        auto t1_start = std::chrono::high_resolution_clock::now();
        ImageGray blur_a = run_stage1_preprocess(&img_a, "separable", 5, 1.4f);
        ImageGray blur_b = run_stage1_preprocess(&img_b, "separable", 5, 1.4f);
        auto t1_end = std::chrono::high_resolution_clock::now();
        st.t_stage1_ms = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();

        if (is_last && save_intermediate) {
            save_gray_png(&blur_a, (sub_s1 + "/blur_A.png").c_str());
            save_gray_png(&blur_b, (sub_s1 + "/blur_B.png").c_str());
        }

        // Stage 2: Keypoints
        auto t2_start = std::chrono::high_resolution_clock::now();
        ImageGray resp_a = {0, 0, nullptr}, resp_b = {0, 0, nullptr};
        KeypointList kp_a = run_stage2_keypoints(&blur_a, "harris", 1500, is_last && save_intermediate ? &resp_a : nullptr);
        KeypointList kp_b = run_stage2_keypoints(&blur_b, "harris", 1500, is_last && save_intermediate ? &resp_b : nullptr);
        auto t2_end = std::chrono::high_resolution_clock::now();
        st.t_stage2_ms = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();
        num_kps_a = kp_a.count;
        num_kps_b = kp_b.count;

        if (is_last && save_intermediate) {
            save_heatmap_png(&resp_a, (sub_s2 + "/response_heatmap_A.png").c_str());
            save_heatmap_png(&resp_b, (sub_s2 + "/response_heatmap_B.png").c_str());
            draw_keypoints_png(&img_a, &kp_a, (sub_s2 + "/keypoints_A.png").c_str());
            draw_keypoints_png(&img_b, &kp_b, (sub_s2 + "/keypoints_B.png").c_str());
            free_image_gray(&resp_a);
            free_image_gray(&resp_b);
        }

        // Stage 3: Descriptors
        auto t3_start = std::chrono::high_resolution_clock::now();
        compute_ic_orientations(&blur_a, &kp_a, 15);
        compute_ic_orientations(&blur_b, &kp_b, 15);
        Descriptor256* desc_a = extract_binary_descriptors(&blur_a, &kp_a, pattern, 256, true);
        Descriptor256* desc_b = extract_binary_descriptors(&blur_b, &kp_b, pattern, 256, true);
        auto t3_end = std::chrono::high_resolution_clock::now();
        st.t_stage3_ms = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();

        // Stage 4: Matching
        auto t4_start = std::chrono::high_resolution_clock::now();
        MatchList matches = match_descriptors_hamming(desc_a, kp_a.count, desc_b, kp_b.count, 0.75f, true);
        auto t4_end = std::chrono::high_resolution_clock::now();
        st.t_stage4_ms = std::chrono::duration<double, std::milli>(t4_end - t4_start).count();
        final_matches = matches.count;

        if (is_last && save_intermediate) {
            draw_matches_png(&img_a, &kp_a, &img_b, &kp_b, &matches, nullptr, (sub_s4 + "/filtered_matches.png").c_str());
        }

        // Stage 5: Homography
        auto t5_start = std::chrono::high_resolution_clock::now();
        Homography H;
        bool* inlier_mask = (bool*)calloc(matches.count, sizeof(bool));
        int inlier_cnt = 0;
        estimate_homography_ransac(&kp_a, &kp_b, &matches, 2000, 3.5f, &H, inlier_mask, &inlier_cnt);
        auto t5_end = std::chrono::high_resolution_clock::now();
        st.t_stage5_ms = std::chrono::duration<double, std::milli>(t5_end - t5_start).count();
        final_inliers = inlier_cnt;

        if (is_last && save_intermediate) {
            draw_matches_png(&img_a, &kp_a, &img_b, &kp_b, &matches, inlier_mask, (sub_s5 + "/inlier_matches.png").c_str());
            FILE* f_h = fopen((sub_s5 + "/homography.json").c_str(), "w");
            if (f_h) {
                fprintf(f_h, "{\n  \"H\": [\n");
                for (int row = 0; row < 3; ++row) {
                    fprintf(f_h, "    [%.8f, %.8f, %.8f]%s\n", H.m[row][0], H.m[row][1], H.m[row][2], row < 2 ? "," : "");
                }
                fprintf(f_h, "  ]\n}\n");
                fclose(f_h);
            }
        }

        // Stage 6: Warping & Blending
        auto t6_start = std::chrono::high_resolution_clock::now();
        if (is_last) {
            final_pano = run_stage6_warp_blend(&img_a, &img_b, &H, "bilinear", "feathering");
        } else {
            ImageRGB temp_pano = run_stage6_warp_blend(&img_a, &img_b, &H, "bilinear", "feathering");
            free_image_rgb(&temp_pano);
        }
        auto t6_end = std::chrono::high_resolution_clock::now();
        st.t_stage6_ms = std::chrono::duration<double, std::milli>(t6_end - t6_start).count();

        auto t_e2e_end = std::chrono::high_resolution_clock::now();
        st.t_kernel_total_ms = st.t_stage1_ms + st.t_stage2_ms + st.t_stage3_ms +
                               st.t_stage4_ms + st.t_stage5_ms + st.t_stage6_ms;
        st.t_e2e_ms = std::chrono::duration<double, std::milli>(t_e2e_end - t_e2e_start).count();

        all_timings.push_back(st);

        // Free per-run allocations
        free_image_gray(&blur_a);
        free_image_gray(&blur_b);
        free_keypoint_list(&kp_a);
        free_keypoint_list(&kp_b);
        free(desc_a);
        free(desc_b);
        free_match_list(&matches);
        free(inlier_mask);
    }

    free(pattern);

    // Compute averages and minimums
    StageTimings mean_t{}, min_t{};
    min_t.t_stage1_ms = min_t.t_stage2_ms = min_t.t_stage3_ms = min_t.t_stage4_ms =
        min_t.t_stage5_ms = min_t.t_stage6_ms = min_t.t_kernel_total_ms = min_t.t_e2e_ms = 1e9;

    for (const auto& t : all_timings) {
        mean_t.t_stage1_ms += t.t_stage1_ms;
        mean_t.t_stage2_ms += t.t_stage2_ms;
        mean_t.t_stage3_ms += t.t_stage3_ms;
        mean_t.t_stage4_ms += t.t_stage4_ms;
        mean_t.t_stage5_ms += t.t_stage5_ms;
        mean_t.t_stage6_ms += t.t_stage6_ms;
        mean_t.t_kernel_total_ms += t.t_kernel_total_ms;
        mean_t.t_e2e_ms += t.t_e2e_ms;

        min_t.t_stage1_ms = std::min(min_t.t_stage1_ms, t.t_stage1_ms);
        min_t.t_stage2_ms = std::min(min_t.t_stage2_ms, t.t_stage2_ms);
        min_t.t_stage3_ms = std::min(min_t.t_stage3_ms, t.t_stage3_ms);
        min_t.t_stage4_ms = std::min(min_t.t_stage4_ms, t.t_stage4_ms);
        min_t.t_stage5_ms = std::min(min_t.t_stage5_ms, t.t_stage5_ms);
        min_t.t_stage6_ms = std::min(min_t.t_stage6_ms, t.t_stage6_ms);
        min_t.t_kernel_total_ms = std::min(min_t.t_kernel_total_ms, t.t_kernel_total_ms);
        min_t.t_e2e_ms = std::min(min_t.t_e2e_ms, t.t_e2e_ms);
    }

    mean_t.t_stage1_ms /= runs;
    mean_t.t_stage2_ms /= runs;
    mean_t.t_stage3_ms /= runs;
    mean_t.t_stage4_ms /= runs;
    mean_t.t_stage5_ms /= runs;
    mean_t.t_stage6_ms /= runs;
    mean_t.t_kernel_total_ms /= runs;
    mean_t.t_e2e_ms /= runs;

    summary.runs = runs;
    summary.mean_time = mean_t;
    summary.min_time = min_t;
    summary.kps_a = num_kps_a;
    summary.kps_b = num_kps_b;
    summary.matches = final_matches;
    summary.inliers = final_inliers;
    summary.pano_width = final_pano.width;
    summary.pano_height = final_pano.height;

    return 0;
}
