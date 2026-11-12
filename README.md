# CSC14120 - Panorama Image Stitching (Sequential V0 Baseline)

**Student**: Trần Xuân Sơn  
**Student ID**: 19127321  

A purely algorithmic, synchronous implementation of the 6-stage panorama image stitching pipeline written in standard C/C++. All algorithms are implemented from scratch using basic math and single-header image I/O with zero black-box libraries.

---

## 1. Project Structure

```text
ltss/
├── src/                           # Pure C/C++/CUDA source files
│   ├── common.h / common.c        # Core image structures & debug visualization
│   ├── cuda_utils.cuh             # CUDA error checking & memory utilities (V1/V2)
│   ├── versions.h                 # Unified pipeline interface for V0, V1, V2
│   ├── main.cu                    # CLI harness, Section 4.1 & 4.2 benchmarking
│   ├── stb_impl.cpp               # STB image loading and saving implementation
│   ├── v0_son.cpp                 # Sequential V0 reference implementation
│   ├── stages/
│   │   ├── s1_preprocess.h / .c   # Grayscale & separable Gaussian filter
│   │   ├── s2_keypoints.h / .c    # Sobel gradients, Harris, NMS, Top-K
│   │   ├── s3_descriptors.h / .c  # ORB (Rotated BRIEF) & intensity centroid
│   │   ├── s4_matching.h / .c     # Hamming popcount, ratio test, cross-check
│   │   ├── s5_homography.h / .c   # Hartley normalization, 4-point DLT, RANSAC
│   │   └── s6_warp_blend.h / .c   # Inverse homography, Bilinear, feathering
│   └── third_party/
│       ├── stb_image.h            # Single-header image loader
│       └── stb_image_write.h      # Single-header image writer
├── data/
│   ├── input/                     # Original inputs (1.jpg, 2.jpg, left.jpg, right.jpg)
│   └── input_resized/             # 16:9 benchmark sets (small, medium, large)
├── scripts/
│   └── prepare_dataset.py         # 16:9 center-crop and resizer tool
├── Makefile                       # Cross-platform compilation
└── Group16_Midterm.ipynb          # Midterm report notebook (Colab ready)
```

---

## 2. Dataset Preparation

Per Section 1.2 of the project specification, image pairs are cropped to 16:9 and resized to 3 resolutions:
- **Small**: 1280 × 720 (fast development & debugging)
- **Medium**: 1920 × 1080 (main benchmark)
- **Large**: 3840 × 2160 (high-resolution scaling evaluation)

```bash
python scripts/prepare_dataset.py --left data/input/1.jpg --right data/input/2.jpg --out-dir data/input_resized
```

---

## 3. Compilation & Execution

### 3.1 Build Command
```bash
# Using G++ directly:
g++ -O3 -Wall -Wextra -static -I. -Isrc -std=c++17 src/common.c src/stages/*.c src/main.cpp -o stitch.exe -lm

# Or using Makefile:
make
```

### 3.2 Specification-Compatible Execution (Section 4.1 & 4.2)
```bash
# Standard run (default 5 timed runs, warmup excluded, disk I/O excluded):
./stitch.exe data/input_resized/small/1.jpg data/input_resized/small/2.jpg data/output/c_panorama_small.png --version 0

# Custom repetitions on Medium 1080p:
./stitch.exe data/input_resized/medium/1.jpg data/input_resized/medium/2.jpg data/output/c_panorama_medium.png --version 0 --runs 5

# Run on user captured real-world pair:
./stitch.exe data/input/left.jpg data/input/right.jpg data/output/custom_panorama.png --version 0 --runs 1
```