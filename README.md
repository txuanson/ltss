# CSC14120 - Panorama Image Stitching (Sequential V0 Baseline)

**Student**: Trần Xuân Sơn  
**Student ID**: 19127321  

A purely algorithmic, synchronous implementation of the 6-stage panorama image stitching pipeline written in standard C/C++. All algorithms are implemented from scratch using basic math and single-header image I/O with zero black-box libraries.

---

## 1. Project Structure

```text
ltss/
├── src/                           # Pure C/C++ sequential pipeline source
│   ├── common.h / common.c        # Image structures, keypoints, memory helpers
│   ├── main.cpp                   # CLI harness, Section 4.1 & 4.2 benchmarking
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
├── Makefile                       # Compilation shortcut
└── report.ipynb                   # Interactive Jupyter Notebook report (Colab ready)
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

---

## 4. Pipeline Stages & Algorithms

| Stage | Module | Algorithms & Methods |
| :--- | :--- | :--- |
| **Stage 1: Preprocessing** | [`src/stages/s1_preprocess.c`](file:///d:/hcmus/ltss/src/stages/s1_preprocess.c) | Grayscale ($0.299R+0.587G+0.114B$) + Separable 1D Gaussian kernel |
| **Stage 2: Keypoints** | [`src/stages/s2_keypoints.c`](file:///d:/hcmus/ltss/src/stages/s2_keypoints.c) | Sobel gradients, Harris Corner response, NMS ($7\times 7$), Top-K quicksort |
| **Stage 3: Descriptors** | [`src/stages/s3_descriptors.c`](file:///d:/hcmus/ltss/src/stages/s3_descriptors.c) | ORB (steered BRIEF 256-bit) with Intensity Centroid angle |
| **Stage 4: Matching** | [`src/stages/s4_matching.c`](file:///d:/hcmus/ltss/src/stages/s4_matching.c) | All-pairs Hamming distance with 64-bit popcount, Lowe's ratio test ($0.75$), cross-check |
| **Stage 5: Homography** | [`src/stages/s5_homography.c`](file:///d:/hcmus/ltss/src/stages/s5_homography.c) | Hartley normalization, 4-point DLT via Gaussian elimination, RANSAC (2000 iters) |
| **Stage 6: Warping & Blending**| [`src/stages/s6_warp_blend.c`](file:///d:/hcmus/ltss/src/stages/s6_warp_blend.c) | Canvas bounding box, inverse homography ($H^{-1}$), Bilinear sampling, feathering |

---

## 5. References & Citations

### Papers & Books
1. **Shree K. Nayar**, *"Image Stitching"*, First Principles of Computer Vision (FPCV) Monograph, Columbia University, 2025. [https://fpcv.cs.columbia.edu/](https://fpcv.cs.columbia.edu/)
2. **Richard Szeliski**, *"Computer Vision: Algorithms and Applications"*, 2nd Edition, Springer, 2022. [https://szeliski.org/Book/](https://szeliski.org/Book/)
3. **C. Harris and M. Stephens**, *"A Combined Corner and Edge Detector"*, In Proceedings of the 4th Alvey Vision Conference, pp. 147–151, 1988.
4. **Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski**, *"ORB: An efficient alternative to SIFT or SURF"*, IEEE International Conference on Computer Vision (ICCV), 2011.
5. **Michael Calonder, Vincent Lepetit, Christoph Strecha, Pascal Fua**, *"BRIEF: Binary Robust Independent Elementary Features"*, European Conference on Computer Vision (ECCV), 2010.
6. **Martin A. Fischler and Robert C. Bolles**, *"Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography"*, Communications of the ACM, 24(6): 381–395, 1981.
7. **Richard Hartley**, *"In Defense of the Eight-Point Algorithm"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 19(6): 580–593, 1997.
8. **Matthew Brown and David G. Lowe**, *"Automatic Panoramic Image Stitching using Invariant Features"*, International Journal of Computer Vision (IJCV), 74(1): 59–73, 2007.

### Open-Source Code & Libraries
1. **stb (Sean Barrett)**: Single-file public domain image loader and writer (`stb_image.h`, `stb_image_write.h`). [https://github.com/nothings/stb](https://github.com/nothings/stb)
2. **AI Tool Usage Declaration**: As required by Section 5.2 and Section 6 of the course specification, Google Antigravity / Gemini was used for code scaffolding, syntax checking, and documentation structuring.
