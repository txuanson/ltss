/*
 * Author: Tran Xuan Son (Student ID: 19127321)
 * Group: Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * Purpose: Top-level driver / CLI harness for Panorama Image Stitching.
 *          Supports Section 4.1 CLI specification and Section 4.2 timing standards.
 */

#include "common.h"
#include "versions.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <left_image> <right_image> <output_image> --version <0|1|2> [--runs N] [--timing-json <path>]\n";
        return 1;
    }

    PipelineConfig config;
    int pos_idx = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            config.version = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            config.runs = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.out_dir = argv[++i];
            config.save_intermediate = true;
        } else if (arg == "--save-intermediate") {
            config.save_intermediate = true;
        } else if (arg == "--no-intermediate") {
            config.save_intermediate = false;
        } else if (arg == "--timing-json" && i + 1 < argc) {
            config.timing_json_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <left_image> <right_image> <output_image> --version <0|1|2> [--runs N]\n";
            return 0;
        } else if (arg[0] != '-') {
            if (pos_idx == 0) config.left_path = arg;
            else if (pos_idx == 1) config.right_path = arg;
            else if (pos_idx == 2) config.out_path = arg;
            pos_idx++;
        }
    }

    std::cout << "==================================================================\n";
    std::cout << "CSC14120 Panorama Image Stitching (Sequential V" << config.version << " - Reference)\n";
    std::cout << "  Left Image (A)  : " << config.left_path << "\n";
    std::cout << "  Right Image (B) : " << config.right_path << "\n";
    std::cout << "  Output File     : " << config.out_path << "\n";
    std::cout << "  Version Selected: V" << config.version << "\n";
    std::cout << "  Timed Runs (N)  : " << config.runs << " (with initial warm-up run)\n";
    std::cout << "==================================================================\n";

    // 1. Image loading from disk (EXCLUDED from timing per Section 4.2)
    ImageRGB img_a = load_image(config.left_path.c_str());
    ImageRGB img_b = load_image(config.right_path.c_str());
    std::cout << "Loaded Image A: " << img_a.width << "x" << img_a.height 
              << " | Image B: " << img_b.width << "x" << img_b.height << std::endl;

    std::cout << "Executing 1 untimed warm-up run..." << std::endl;

    PipelineSummary summary;
    ImageRGB final_panorama{0, 0, 0, nullptr};

    if (config.version == 0) {
        run_v0_sequential(img_a, img_b, config, summary, final_panorama);
    } else {
        std::cerr << "Version V" << config.version << " is reserved for future parallel milestones (V1/V2).\n";
        free_image_rgb(&img_a);
        free_image_rgb(&img_b);
        return 1;
    }

    // Save final stitched panorama to disk (Disk I/O outside timed region)
    save_image_png(&final_panorama, config.out_path.c_str());

    // Print Performance Table matching Section 4.2
    std::cout << "\n" << std::string(68, '-') << "\n";
    std::cout << "PERFORMANCE REPORT (Section 4.2 Specification Requirements)\n";
    std::cout << "  Platform / Version   : V" << config.version << " Sequential (CPU single-thread)\n";
    std::cout << "  Timing Mechanism     : std::chrono::high_resolution_clock\n";
    std::cout << "  Timed Runs Averaged  : " << summary.runs << "\n";
    std::cout << "  Disk I/O Status      : EXCLUDED from all measurements\n";
    std::cout << std::string(68, '-') << "\n";
    std::cout << std::left << std::setw(28) << "Stage"
              << " | " << std::right << std::setw(10) << "Mean (ms)"
              << " | " << std::right << std::setw(10) << "Min (ms)"
              << " | " << std::right << std::setw(8) << "Share (%)" << "\n";
    std::cout << std::string(68, '-') << "\n";

    auto print_row = [&](const std::string& name, double mean_val, double min_val) {
        double share = summary.mean_time.t_e2e_ms > 0 ? (mean_val / summary.mean_time.t_e2e_ms * 100.0) : 0.0;
        std::cout << std::left << std::setw(28) << name
                  << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << mean_val
                  << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << min_val
                  << " | " << std::right << std::setw(7) << std::fixed << std::setprecision(1) << share << "%\n";
    };

    print_row("Stage 1: Preprocessing", summary.mean_time.t_stage1_ms, summary.min_time.t_stage1_ms);
    print_row("Stage 2: Keypoint Detection", summary.mean_time.t_stage2_ms, summary.min_time.t_stage2_ms);
    print_row("Stage 3: Description", summary.mean_time.t_stage3_ms, summary.min_time.t_stage3_ms);
    print_row("Stage 4: Matching", summary.mean_time.t_stage4_ms, summary.min_time.t_stage4_ms);
    print_row("Stage 5: Homography (RANSAC)", summary.mean_time.t_stage5_ms, summary.min_time.t_stage5_ms);
    print_row("Stage 6: Warping & Blending", summary.mean_time.t_stage6_ms, summary.min_time.t_stage6_ms);
    std::cout << std::string(68, '-') << "\n";
    print_row("TOTAL KERNEL / STAGES TIME", summary.mean_time.t_kernel_total_ms, summary.min_time.t_kernel_total_ms);
    print_row("END-TO-END PIPELINE TIME", summary.mean_time.t_e2e_ms, summary.min_time.t_e2e_ms);
    std::cout << std::string(68, '-') << "\n";
    std::cout << "Keypoints Detected (A / B) : " << summary.kps_a << " / " << summary.kps_b << "\n";
    std::cout << "Matches (Ratio + CrossCheck): " << summary.matches << "\n";
    std::cout << "RANSAC Inliers             : " << summary.inliers << " / " << summary.matches 
              << " (" << std::fixed << std::setprecision(1) << (summary.matches > 0 ? (double)summary.inliers / summary.matches * 100.0 : 0.0) << "%)\n";
    std::cout << "Stitched Panorama Size     : " << final_panorama.width << "x" << final_panorama.height 
              << " saved to " << config.out_path << "\n";
    std::cout << std::string(68, '=') << std::endl;

    // Export Timing JSON if requested
    if (!config.timing_json_path.empty()) {
        std::error_code ec;
        fs::create_directories(fs::path(config.timing_json_path).parent_path(), ec);
        FILE* fj = fopen(config.timing_json_path.c_str(), "w");
        if (fj) {
            fprintf(fj, "{\n");
            fprintf(fj, "  \"version\": %d,\n", config.version);
            fprintf(fj, "  \"runs\": %d,\n", summary.runs);
            fprintf(fj, "  \"img_a_width\": %d,\n", img_a.width);
            fprintf(fj, "  \"img_a_height\": %d,\n", img_a.height);
            fprintf(fj, "  \"kps_a\": %d,\n", summary.kps_a);
            fprintf(fj, "  \"kps_b\": %d,\n", summary.kps_b);
            fprintf(fj, "  \"matches\": %d,\n", summary.matches);
            fprintf(fj, "  \"inliers\": %d,\n", summary.inliers);
            fprintf(fj, "  \"pano_width\": %d,\n", final_panorama.width);
            fprintf(fj, "  \"pano_height\": %d,\n", final_panorama.height);
            fprintf(fj, "  \"stages\": {\n");
            fprintf(fj, "    \"stage1_preprocess\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", summary.mean_time.t_stage1_ms, summary.min_time.t_stage1_ms);
            fprintf(fj, "    \"stage2_keypoints\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", summary.mean_time.t_stage2_ms, summary.min_time.t_stage2_ms);
            fprintf(fj, "    \"stage3_descriptors\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", summary.mean_time.t_stage3_ms, summary.min_time.t_stage3_ms);
            fprintf(fj, "    \"stage4_matching\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", summary.mean_time.t_stage4_ms, summary.min_time.t_stage4_ms);
            fprintf(fj, "    \"stage5_homography\": {\"mean_ms\": %.3f, \"min_ms\": %.3f},\n", summary.mean_time.t_stage5_ms, summary.min_time.t_stage5_ms);
            fprintf(fj, "    \"stage6_warp_blend\": {\"mean_ms\": %.3f, \"min_ms\": %.3f}\n", summary.mean_time.t_stage6_ms, summary.min_time.t_stage6_ms);
            fprintf(fj, "  },\n");
            fprintf(fj, "  \"total_kernel_mean_ms\": %.3f,\n", summary.mean_time.t_kernel_total_ms);
            fprintf(fj, "  \"total_kernel_min_ms\": %.3f,\n", summary.min_time.t_kernel_total_ms);
            fprintf(fj, "  \"e2e_mean_ms\": %.3f,\n", summary.mean_time.t_e2e_ms);
            fprintf(fj, "  \"e2e_min_ms\": %.3f\n", summary.min_time.t_e2e_ms);
            fprintf(fj, "}\n");
            fclose(fj);
        }
    }

    free_image_rgb(&img_a);
    free_image_rgb(&img_b);
    free_image_rgb(&final_panorama);
    return 0;
}
