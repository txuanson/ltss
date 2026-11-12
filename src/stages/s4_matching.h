#ifndef S4_MATCHING_H
#define S4_MATCHING_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Brute-force Hamming distance matching with Lowe's ratio test and cross-check
MatchList match_descriptors_hamming(const Descriptor256* desc_a, int count_a,
                                    const Descriptor256* desc_b, int count_b,
                                    float ratio_thresh, bool cross_check);

#ifdef __cplusplus
}
#endif

#endif // S4_MATCHING_H
