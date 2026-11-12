/*
 * Author: Tran Xuan Son (Student ID: 19127321)
 * Group: Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * Purpose: Header defining pipeline execution interfaces for V0 (Sequential),
 *          V1 (CUDA Baseline), and V2 (CUDA Optimized).
 */

#pragma once

#include "common.h"
#include <string>
#include <vector>

struct PipelineConfig {
    std::string left_path;
    std::string right_path;
    std::string out_path;
    std::string out_dir = "data/output/run";
    int version = 0;
    int runs = 5;
    bool save_intermediate = false;
    std::string timing_json_path = "";
};

struct StageTimings {
    double t_stage1_ms = 0.0;
    double t_stage2_ms = 0.0;
    double t_stage3_ms = 0.0;
    double t_stage4_ms = 0.0;
    double t_stage5_ms = 0.0;
    double t_stage6_ms = 0.0;
    double t_kernel_total_ms = 0.0;
    double t_e2e_ms = 0.0;
};

struct PipelineSummary {
    int runs = 1;
    StageTimings mean_time;
    StageTimings min_time;
    int kps_a = 0;
    int kps_b = 0;
    int matches = 0;
    int inliers = 0;
    int pano_width = 0;
    int pano_height = 0;
};

// V0: Single-threaded Sequential C/C++ Reference Implementation
int run_v0_sequential(const ImageRGB& img_a, const ImageRGB& img_b,
                       const PipelineConfig& config, PipelineSummary& summary,
                       ImageRGB& final_pano);

// Planned for Milestone 2 (Final Submission):
// int run_v1_cuda_baseline(...);
// int run_v2_cuda_optimized(...);
