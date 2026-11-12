#ifndef S3_DESCRIPTORS_H
#define S3_DESCRIPTORS_H

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float dx1, dy1;
    float dx2, dy2;
} BriefPair;

// Generate 256 deterministic sampling test pairs
BriefPair* generate_brief_pattern(int n_bits, int patch_size, unsigned int seed);

// Compute Intensity Centroid orientation angle in radians
void compute_ic_orientations(const ImageGray* src, KeypointList* kps, int patch_radius);

// Extract 256-bit binary descriptors (ORB / steered BRIEF or standard BRIEF)
Descriptor256* extract_binary_descriptors(const ImageGray* src, const KeypointList* kps,
                                          const BriefPair* pattern, int n_bits, bool use_orientation);

#ifdef __cplusplus
}
#endif

#endif // S3_DESCRIPTORS_H
