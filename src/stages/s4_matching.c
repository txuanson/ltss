#include "s4_matching.h"

// Fast popcount using built-in or 64-bit uint64 chunks
static inline int hamming_distance_256(const Descriptor256* a, const Descriptor256* b) {
    const uint64_t* u64_a = (const uint64_t*)a->bytes;
    const uint64_t* u64_b = (const uint64_t*)b->bytes;
    int dist = 0;
    for (int i = 0; i < 4; ++i) {
        dist += __builtin_popcountll(u64_a[i] ^ u64_b[i]);
    }
    return dist;
}

MatchList match_descriptors_hamming(const Descriptor256* desc_a, int count_a,
                                    const Descriptor256* desc_b, int count_b,
                                    float ratio_thresh, bool cross_check) {
    MatchList matches = create_match_list(count_a);
    if (count_a == 0 || count_b < 2) return matches;

    // Nearest and second-nearest matches for A -> B
    int* best_b_for_a = (int*)malloc(count_a * sizeof(int));
    int* dist1_for_a  = (int*)malloc(count_a * sizeof(int));
    int* dist2_for_a  = (int*)malloc(count_a * sizeof(int));

    for (int i = 0; i < count_a; ++i) {
        int best_b = -1;
        int d1 = 100000;
        int d2 = 100000;

        for (int j = 0; j < count_b; ++j) {
            int d = hamming_distance_256(&desc_a[i], &desc_b[j]);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                best_b = j;
            } else if (d < d2) {
                d2 = d;
            }
        }
        best_b_for_a[i] = best_b;
        dist1_for_a[i]  = d1;
        dist2_for_a[i]  = d2;
    }

    // Optional cross-check B -> A
    int* best_a_for_b = NULL;
    if (cross_check) {
        best_a_for_b = (int*)malloc(count_b * sizeof(int));
        for (int j = 0; j < count_b; ++j) {
            int best_a = -1;
            int min_d = 100000;
            for (int i = 0; i < count_a; ++i) {
                int d = hamming_distance_256(&desc_a[i], &desc_b[j]);
                if (d < min_d) {
                    min_d = d;
                    best_a = i;
                }
            }
            best_a_for_b[j] = best_a;
        }
    }

    // Apply Lowe's ratio test and cross-check
    for (int i = 0; i < count_a; ++i) {
        int d1 = dist1_for_a[i];
        int d2 = dist2_for_a[i];
        int b = best_b_for_a[i];

        if (d2 > 0 && ((float)d1 / (float)d2) < ratio_thresh) {
            if (!cross_check || best_a_for_b[b] == i) {
                Match m;
                m.idx_a = i;
                m.idx_b = b;
                m.distance = (float)d1;
                add_match(&matches, m);
            }
        }
    }

    free(best_b_for_a);
    free(dist1_for_a);
    free(dist2_for_a);
    if (best_a_for_b) free(best_a_for_b);

    return matches;
}
