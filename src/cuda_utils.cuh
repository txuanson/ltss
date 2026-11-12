/*
 * Author: Tran Xuan Son (Student ID: 19127321)
 * Group: Group 16
 * Course: CSC14120 - Parallel Programming Final Project
 * External Sources: Standard CUDA SDK error checking idioms.
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error in %s at line %d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#else
// Stubs when compiling with standard C++ compiler (G++/Clang) without CUDA toolchain
#define CHECK_CUDA(call) do {} while (0)
#define CHECK_LAST_CUDA_ERROR() do {} while (0)
#endif
