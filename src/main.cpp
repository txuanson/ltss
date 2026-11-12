#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

extern "C" {
#include "common.h"
#include "stages/s1_preprocess.h"
#include "stages/s2_keypoints.h"
#include "stages/s3_descriptors.h"
#include "stages/s4_matching.h"
#include "stages/s5_homography.h"
#include "stages/s6_warp_blend.h"
}

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <sys/stat.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#define MKDIR(d) mkdir(d, 0755)
#endif

static void ensure_dir(const char* dir) {
    MKDIR(dir);
}

ImageRGB load_image(const char* filepath) {
    int w, h, channels;
    uint8_t* data = stbi_load(filepath, &w, &h, &channels, 3);
    if (!data) {
        std::cerr << "Error: Could not load image from " << filepath << std::endl;
        exit(1);
    }
    ImageRGB img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data = data;
    return img;
}

struct RunTiming {
    double t_stage1_ms{0.0};
    double t_stage2_ms{0.0};
    double t_stage3_ms{0.0};
    double t_stage4_ms{0.0};
    double t_stage5_ms{0.0};
    double t_stage6_ms{0.0};
    double t_kernel_total_ms{0.0};
    double t_e2e_ms{0.0}; // excluding disk I/O
};

int main(int argc, char** argv) {
    std::string left_path = "data/input_resized/small/1.jpg";
    std::string right_path = "data/input_resized/small/2.jpg";
    std::string out_path = "data/output/c_panorama.png";
    std::string out_dir = "data/output/c_run";
    int version = 0;
    int runs = 5; // Default 5 runs per Section 4.1 & 4.2
    std::string blur_method = "gaussian_sep";
    std::string detector = "harris";
    std::string interp = "bilinear";
    std::string blend = "feathering";
    bool save_intermediate = true;
    std::string timing_json_path = "";

    // Parse command line arguments
    int pos_idx = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            version = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--output-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--detector" && i + 1 < argc) {
            detector = argv[++i];
        } else if (arg == "--blend" && i + 1 < argc) {
            blend = argv[++i];
        } else if (arg == "--no-intermediate") {
            save_intermediate = false;
        } else if (arg == "--timing-json" && i + 1 < argc) {
            timing_json_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N]\n";
            return 0;
        } else if (arg[0] != '-') {
            if (pos_idx == 0) left_path = arg;
            else if (pos_idx == 1) right_path = arg;
            else if (pos_idx == 2) out_path = arg;
            pos_idx++;
        }
    }

    std::cout << "==================================================================\n";
    std::cout << "CSC14120 Panorama Image Stitching (Sequential V" << version << " - Reference)\n";
    std::cout << "  Left Image (A)  : " << left_path << "\n";
    std::cout << "  Right Image (B) : " << right_path << "\n";
    std::cout << "  Output File     : " << out_path << "\n";
    std::cout << "  Version Selected: V" << version << "\n";
    std::cout << "  Timed Runs (N)  : " << runs << " (with initial warm-up run)\n";
    std::cout << "==================================================================\n";

    // 1. Image loading from disk (EXCLUDED from timing per Section 4.2)
    ImageRGB img_a = load_image(left_path.c_str());
    ImageRGB img_b = load_image(right_path.c_str());
    std::cout << "Loaded Image A: " << img_a.width << "x" << img_a.height 
              << " | Image B: " << img_b.width << "x" << img_b.height << std::endl;

    ensure_dir("data");
    ensure_dir("data/output");
    ensure_dir(out_dir.c_str());

    std::string sub_s1 = out_dir + "/stage1_preprocess";
    std::string sub_s2 = out_dir + "/stage2_keypoints";
    std::string sub_s3 = out_dir + "/stage3_descriptors";
    std::string sub_s4 = out_dir + "/stage4_matching";
    std::string sub_s5 = out_dir + "/stage5_homography";
    std::string sub_s6 = out_dir + "/stage6_warp_blend";

    if (save_intermediate) {
        ensure_dir(sub_s1.c_str());
        ensure_dir(sub_s2.c_str());
        ensure_dir(sub_s3.c_str());
        ensure_dir(sub_s4.c_str());
        ensure_dir(sub_s5.c_str());
        ensure_dir(sub_s6.c_str());
    }

    BriefPair* pattern = generate_brief_pattern(256, 31, 42);

    // Warm-up run (untimed, per Section 4.2: "Warm-up: run the pipeline once before timing")
    std::cout << "Executing 1 untimed warm-up run..." << std::endl;
    {
        ImageGray b_a = run_stage1_preprocess(&img_a, blur_method.c_str(), 5, 1.4f);
        ImageGray b_b = run_stage1_preprocess(&img_b, blur_method.c_str(), 5, 1.4f);
        KeypointList kp_a = run_stage2_keypoints(&b_a, detector.c_str(), 1500, nullptr);
        KeypointList kp_b = run_stage2_keypoints(&b_b, detector.c_str(), 1500, nullptr);
        compute_ic_orientations(&b_a, &kp_a, 15);
        compute_ic_orientations(&b_b, &kp_b, 15);
        Descriptor256* d_a = extract_binary_descriptors(&b_a, &kp_a, pattern, 256, true);
        Descriptor256* d_b = extract_binary_descriptors(&b_b, &kp_b, pattern, 256, true);
        MatchList m = match_descriptors_hamming(d_a, kp_a.count, d_b, kp_b.count, 0.75f, true);
        Homography H;
        bool* mask = (bool*)calloc(m.count, sizeof(bool));
        int in_cnt = 0;
        bool h_ok = estimate_homography_ransac(&kp_a, &kp_b, &m, 2000, 3.5f, &H, mask, &in_cnt);
        if (h_ok) {
            ImageRGB pano = run_stage6_warp_blend(&img_a, &img_b, &H, interp.c_str(), blend.c_str());
            free_image_rgb(&pano);
        }
        free_image_gray(&b_a);
        free_image_gray(&b_b);
        free_keypoint_list(&kp_a);
        free_keypoint_list(&kp_b);
        free(d_a);
        free(d_b);
        free_match_list(&m);
        free(mask);
    }

    std::vector<RunTiming> all_timings;
    all_timings.reserve(runs);

    ImageRGB final_panorama{0, 0, 0, nullptr};
    int final_inliers = 0;
    int final_matches = 0;
    int num_kps_a = 0, num_kps_b = 0;

    for (int r = 0; r < runs; ++r) {
        bool is_last = (r == runs - 1);
        RunTiming rt;

        auto t_e2e_start = std::chrono::high_resolution_clock::now();

        // Stage 1: Preprocessing
        auto t1_start = std::chrono::high_resolution_clock::now();
        ImageGray blur_a = run_stage1_preprocess(&img_a, blur_method.c_str(), 5, 1.4f);
        ImageGray blur_b = run_stage1_preprocess(&img_b, blur_method.c_str(), 5, 1.4f);
        auto t1_end = std::chrono::high_resolution_clock::now();
        rt.t_stage1_ms = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();

        if (is_last && save_intermediate) {
            save_gray_png(&blur_a, (sub_s1 + "/blur_A.png").c_str());
            save_gray_png(&blur_b, (sub_s1 + "/blur_B.png").c_str());
        }

        // Stage 2: Keypoints
        auto t2_start = std::chrono::high_resolution_clock::now();
        ImageGray resp_a = {0}, resp_b = {0};
        KeypointList kp_a = run_stage2_keypoints(&blur_a, detector.c_str(), 1500, is_last && save_intermediate ? &resp_a : nullptr);
        KeypointList kp_b = run_stage2_keypoints(&blur_b, detector.c_str(), 1500, is_last && save_intermediate ? &resp_b : nullptr);
        auto t2_end = std::chrono::high_resolution_clock::now();
        rt.t_stage2_ms = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();
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
        rt.t_stage3_ms = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();

        // Stage 4: Matching
        auto t4_start = std::chrono::high_resolution_clock::now();
        MatchList matches = match_descriptors_hamming(desc_a, kp_a.count, desc_b, kp_b.count, 0.75f, true);
        auto t4_end = std::chrono::high_resolution_clock::now();
        rt.t_stage4_ms = std::chrono::duration<double, std::milli>(t4_end - t4_start).count();
        final_matches = matches.count;

        if (is_last && save_intermediate) {
            draw_matches_png(&img_a, &kp_a, &img_b, &kp_b, &matches, nullptr, (sub_s4 + "/filtered_matches.png").c_str());
        }

        // Stage 5: Homography
        auto t5_start = std::chrono::high_resolution_clock::now();
        Homography H;
        bool* inlier_mask = (bool*)calloc(matches.count, sizeof(bool));
        int inlier_cnt = 0;
        bool h_ok = estimate_homography_ransac(&kp_a, &kp_b, &matches, 2000, 3.5f, &H, inlier_mask, &inlier_cnt);
        auto t5_end = std::chrono::high_resolution_clock::now();
        rt.t_stage5_ms = std::chrono::duration<double, std::milli>(t5_end - t5_start).count();
        final_inliers = inlier_cnt;

        if (is_last && save_intermediate) {
            draw_matches_png(&img_a, &kp_a, &img_b, &kp_b, &matches, inlier_mask, (sub_s5 + "/inlier_matches.png").c_str());
            FILE* f_h = fopen((sub_s5 + "/homography.json").c_str(), "w");
            if (f_h) {
                fprintf(f_h, "{\n  \"H\": [\n");
                for (int row = 0; row < 3; ++row) {
                    fprintf(f_h, "    [%.8f, %.8f, %.8f]%s\n", H.m[row][0], H.m[row][1], H.m[row][2], row < 2 ? "," : "");
                }
                fprintf(f_h, "  ],\n  \"num_inliers\": %d,\n  \"total_matches\": %d\n}\n", inlier_cnt, matches.count);
                fclose(f_h);
            }
        }

        // Stage 6: Warping & Blending
        auto t6_start = std::chrono::high_resolution_clock::now();
        if (final_panorama.data) free_image_rgb(&final_panorama);
        if (h_ok) {
            final_panorama = run_stage6_warp_blend(&img_a, &img_b, &H, interp.c_str(), blend.c_str());
        } else {
            std::cerr << "Warning: Homography estimation failed!\n";
            final_panorama = create_image_rgb(1, 1, 3);
        }
        auto t6_end = std::chrono::high_resolution_clock::now();
        rt.t_stage6_ms = std::chrono::duration<double, std::milli>(t6_end - t6_start).count();

        auto t_e2e_end = std::chrono::high_resolution_clock::now();
        rt.t_kernel_total_ms = rt.t_stage1_ms + rt.t_stage2_ms + rt.t_stage3_ms +
                               rt.t_stage4_ms + rt.t_stage5_ms + rt.t_stage6_ms;
        rt.t_e2e_ms = std::chrono::duration<double, std::milli>(t_e2e_end - t_e2e_start).count();

        all_timings.push_back(rt);

        if (is_last) {
            if (save_intermediate) {
                stbi_write_png((sub_s6 + "/panorama.png").c_str(), final_panorama.width, final_panorama.height, 3, final_panorama.data, final_panorama.width * 3);
            }
            // Save to disk outside timed region
            stbi_write_png(out_path.c_str(), final_panorama.width, final_panorama.height, 3, final_panorama.data, final_panorama.width * 3);
        }

        // Cleanup per-run allocations
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
    free_image_rgb(&img_a);
    free_image_rgb(&img_b);

    // Compute averages and minimums across timed runs
    RunTiming mean_t{}, min_t{};
    min_t.t_stage1_ms = min_t.t_stage2_ms = min_t.t_stage3_ms = min_t.t_stage4_ms = min_t.t_stage5_ms = min_t.t_stage6_ms = min_t.t_kernel_total_ms = min_t.t_e2e_ms = 1e9;

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

    std::cout << "\n" << std::string(68, '-') << "\n";
    std::cout << "PERFORMANCE REPORT (Section 4.2 Specification Requirements)\n";
    std::cout << "  Platform / Version   : V0 Sequential (CPU single-thread)\n";
    std::cout << "  Timing Mechanism     : std::chrono::high_resolution_clock\n";
    std::cout << "  Timed Runs Averaged  : " << runs << "\n";
    std::cout << "  Disk I/O Status      : EXCLUDED from all measurements\n";
    std::cout << std::string(68, '-') << "\n";
    std::cout << std::left << std::setw(28) << "Stage"
              << " | " << std::right << std::setw(10) << "Mean (ms)"
              << " | " << std::right << std::setw(10) << "Min (ms)"
              << " | " << std::right << std::setw(8) << "Share (%)" << "\n";
    std::cout << std::string(68, '-') << "\n";

    auto print_row = [&](const std::string& name, double mean_val, double min_val) {
        double share = mean_t.t_e2e_ms > 0 ? (mean_val / mean_t.t_e2e_ms * 100.0) : 0.0;
        std::cout << std::left << std::setw(28) << name
                  << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << mean_val
                  << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << min_val
                  << " | " << std::right << std::setw(7) << std::fixed << std::setprecision(1) << share << "%\n";
    };

    print_row("Stage 1: Preprocessing", mean_t.t_stage1_ms, min_t.t_stage1_ms);
    print_row("Stage 2: Keypoint Detection", mean_t.t_stage2_ms, min_t.t_stage2_ms);
    print_row("Stage 3: Description", mean_t.t_stage3_ms, min_t.t_stage3_ms);
    print_row("Stage 4: Matching", mean_t.t_stage4_ms, min_t.t_stage4_ms);
    print_row("Stage 5: Homography (RANSAC)", mean_t.t_stage5_ms, min_t.t_stage5_ms);
    print_row("Stage 6: Warping & Blending", mean_t.t_stage6_ms, min_t.t_stage6_ms);
    std::cout << std::string(68, '-') << "\n";
    print_row("TOTAL KERNEL / STAGES TIME", mean_t.t_kernel_total_ms, min_t.t_kernel_total_ms);
    print_row("END-TO-END PIPELINE TIME", mean_t.t_e2e_ms, min_t.t_e2e_ms);
    std::cout << std::string(68, '-') << "\n";
    std::cout << "Keypoints Detected (A / B) : " << num_kps_a << " / " << num_kps_b << "\n";
    std::cout << "Matches (Ratio + CrossCheck): " << final_matches << "\n";
    std::cout << "RANSAC Inliers             : " << final_inliers << " / " << final_matches 
              << " (" << std::fixed << std::setprecision(1) << (final_matches > 0 ? (double)final_inliers / final_matches * 100.0 : 0.0) << "%)\n";
    std::cout << "Stitched Panorama Size     : " << final_panorama.width << "x" << final_panorama.height 
              << " saved to " << out_path << "\n";
    std::cout << std::string(68, '=') << std::endl;

    if (!timing_json_path.empty()) {
        FILE* fj = fopen(timing_json_path.c_str(), "w");
        if (fj) {
            fprintf(fj, "{\n");
            fprintf(fj, "  \"version\": %d,\n", version);
            fprintf(fj, "  \"runs\": %d,\n", runs);
            fprintf(fj, "  \"img_a_width\": %d,\n", img_a.width);
            fprintf(fj, "  \"img_a_height\": %d,\n", img_a.height);
            fprintf(fj, "  \"kps_a\": %d,\n", num_kps_a);
            fprintf(fj, "  \"kps_b\": %d,\n", num_kps_b);
            fprintf(fj, "  \"matches\": %d,\n", final_matches);
            fprintf(fj, "  \"inliers\": %d,\n", final_inliers);
            fprintf(fj, "  \"pano_width\": %d,\n", final_panorama.width);
            fprintf(fj, "  \"pano_height\": %d,\n", final_panorama.height);
            fprintf(fj, "  \"stages\": {\n");
            fprintf(fj, "    \"stage1_preprocess\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", mean_t.t_stage1_ms, min_t.t_stage1_ms);
            fprintf(fj, "    \"stage2_keypoints\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", mean_t.t_stage2_ms, min_t.t_stage2_ms);
            fprintf(fj, "    \"stage3_descriptors\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", mean_t.t_stage3_ms, min_t.t_stage3_ms);
            fprintf(fj, "    \"stage4_matching\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", mean_t.t_stage4_ms, min_t.t_stage4_ms);
            fprintf(fj, "    \"stage5_homography\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", mean_t.t_stage5_ms, min_t.t_stage5_ms);
            fprintf(fj, "    \"stage6_warp_blend\": {\"mean_ms\": %.3f, \"min_ms\": %.3f}\n", mean_t.t_stage6_ms, min_t.t_stage6_ms);
            fprintf(fj, "  },\n");
            fprintf(fj, "  \"total_kernel_mean_ms\": %.3f,\n", mean_t.t_kernel_total_ms);
            fprintf(fj, "  \"total_kernel_min_ms\": %.3f,\n", min_t.t_kernel_total_ms);
            fprintf(fj, "  \"e2e_mean_ms\": %.3f,\n", mean_t.t_e2e_ms);
            fprintf(fj, "  \"e2e_min_ms\": %.3f\n", min_t.t_e2e_ms);
            fprintf(fj, "}\n");
            fclose(fj);
        }
    }

    free_image_rgb(&final_panorama);
    return 0;
}
