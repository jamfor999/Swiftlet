#include "include/qgemv.h"

#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#define SWIFTLET_X86 1
#include <immintrin.h>
#else
#define SWIFTLET_X86 0
#endif

/* ---------- shared helpers ---------- */

static inline uint32_t load_u32(const unsigned char *p, size_t off) {
    uint32_t v;
    memcpy(&v, p + off, 4);
    return v;
}

static inline uint16_t load_u16(const unsigned char *p, size_t off) {
    uint16_t v;
    memcpy(&v, p + off, 2);
    return v;
}

/* f32 (0), f16 (1) or bf16 (2) scale element -> float. The f16 path is the
 * same bit rebiasing SwiftletCore does for Float16-less Intel macOS. */
static inline float load_scale(const unsigned char *p, size_t base, size_t idx, int type) {
    if (type == 0) {
        float f;
        memcpy(&f, p + base + idx * 4, 4);
        return f;
    }
    if (type == 1) {
        uint16_t h = load_u16(p, base + idx * 2);
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp = (uint32_t)(h & 0x7C00) >> 10;
        uint32_t frac = (uint32_t)(h & 0x03FF);
        if (exp == 0x1F) {
            uint32_t nan = sign | 0x7F800000u | frac << 13 | (frac ? 0x00400000u : 0);
            float f;
            memcpy(&f, &nan, 4);
            return f;
        }
        if (exp == 0) {
            if (frac == 0) {
                float f;
                memcpy(&f, &sign, 4);
                return f;
            }
            uint32_t e = 0, n = frac;
            while (!(n & 0x0400)) { n <<= 1; e++; }
            uint32_t bits = sign | (113u - e) << 23 | (n & 0x03FFu) << 13;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* exp in [1,30]: +112 == -15 + 127 without unsigned wrap. */
        uint32_t bits = sign | (exp + 112u) << 23 | frac << 13;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    /* bf16 */
    uint32_t bits = (uint32_t)load_u16(p, base + idx * 2) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---------- scalar reference ---------- */

static void qgemv_scalar(const unsigned char *w, const unsigned char *scales,
                         const unsigned char *biases, const float *x, float *y,
                         int out_dim, int in_dim, int group_size, int bits,
                         int scales_type, int row_begin, int row_end) {
    const int per_word = 32 / bits;
    const uint32_t mask = (uint32_t)(1 << bits) - 1;
    const int packed_cols = in_dim / per_word;
    const int groups = in_dim / group_size;
    const int words_per_group = group_size / per_word;

    for (int row = row_begin; row < row_end; ++row) {
        size_t row_base = (size_t)row * packed_cols * 4;
        float acc = 0;
        for (int g = 0; g < groups; ++g) {
            float qdot = 0, xsum = 0;
            size_t base = row_base + (size_t)g * words_per_group * 4;
            int xbase = g * group_size;
            for (int wi = 0; wi < words_per_group; ++wi) {
                uint32_t word = load_u32(w, base + (size_t)wi * 4);
                int xoff = xbase + wi * per_word;
                for (int j = 0; j < per_word; ++j) {
                    float xv = x[xoff + j];
                    qdot += (float)(word & mask) * xv;
                    xsum += xv;
                    word >>= bits;
                }
            }
            size_t gi = (size_t)row * groups + g;
            acc += load_scale(scales, 0, gi, scales_type) * qdot
                 + load_scale(biases, 0, gi, scales_type) * xsum;
        }
        y[row] = acc;
    }
}

/* ---------- AVX2 (x86 only; other arches use the scalar path) ---------- */

#if SWIFTLET_X86

/* Horizontal sum of the 8 lanes. */
__attribute__((target("avx2")))
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* One group's two sums for one row, vectorized over 8-element blocks.
 * 4-bit: each uint32 word is exactly 8 lanes. 8-bit: two words fill 8 lanes. */
__attribute__((target("avx2")))
static inline void group_dots_avx2(const unsigned char *w, size_t base, const float *x,
                                   int words_per_group, uint32_t mask, int bits,
                                   float *qdot_out, float *xsum_out) {
    __m256 qdot8 = _mm256_setzero_ps();
    __m256 xsum8 = _mm256_setzero_ps();
    if (bits == 4) {
        const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        const __m256i vmask = _mm256_set1_epi32((int)mask);
        for (int wi = 0; wi < words_per_group; ++wi) {
            __m256i word = _mm256_set1_epi32((int)load_u32(w, base + (size_t)wi * 4));
            __m256i q = _mm256_and_si256(_mm256_srlv_epi32(word, shifts), vmask);
            __m256 xv = _mm256_loadu_ps(x + wi * 8);
            qdot8 = _mm256_add_ps(qdot8, _mm256_mul_ps(_mm256_cvtepi32_ps(q), xv));
            xsum8 = _mm256_add_ps(xsum8, xv);
        }
    } else { /* bits == 8: two words per 8 lanes */
        for (int wi = 0; wi < words_per_group; wi += 2) {
            uint32_t w0 = load_u32(w, base + (size_t)wi * 4);
            uint32_t w1 = load_u32(w, base + (size_t)(wi + 1) * 4);
            __m256i q = _mm256_setr_epi32(
                (int)(w0 & 0xFF), (int)((w0 >> 8) & 0xFF),
                (int)((w0 >> 16) & 0xFF), (int)((w0 >> 24) & 0xFF),
                (int)(w1 & 0xFF), (int)((w1 >> 8) & 0xFF),
                (int)((w1 >> 16) & 0xFF), (int)((w1 >> 24) & 0xFF));
            __m256 xv = _mm256_loadu_ps(x + wi * 4);
            qdot8 = _mm256_add_ps(qdot8, _mm256_mul_ps(_mm256_cvtepi32_ps(q), xv));
            xsum8 = _mm256_add_ps(xsum8, xv);
        }
    }
    *qdot_out = hsum256(qdot8);
    *xsum_out = hsum256(xsum8);
}

__attribute__((target("avx2")))
static void qgemv_avx2(const unsigned char *w, const unsigned char *scales,
                       const unsigned char *biases, const float *x, float *y,
                       int out_dim, int in_dim, int group_size, int bits,
                       int scales_type, int row_begin, int row_end) {
    const int per_word = 32 / bits;
    const uint32_t mask = (uint32_t)(1 << bits) - 1;
    const int packed_cols = in_dim / per_word;
    const int groups = in_dim / group_size;
    const int words_per_group = group_size / per_word;

    for (int row = row_begin; row < row_end; ++row) {
        size_t row_base = (size_t)row * packed_cols * 4;
        float acc = 0;
        for (int g = 0; g < groups; ++g) {
            float qdot, xsum;
            group_dots_avx2(w, row_base + (size_t)g * words_per_group * 4,
                            x + g * group_size, words_per_group, mask, bits, &qdot, &xsum);
            size_t gi = (size_t)row * groups + g;
            acc += load_scale(scales, 0, gi, scales_type) * qdot
                 + load_scale(biases, 0, gi, scales_type) * xsum;
        }
        y[row] = acc;
    }
}
#endif /* SWIFTLET_X86 */

/* ---------- dispatcher ---------- */

void swiftlet_qgemv(const unsigned char *w, const unsigned char *scales,
                    const unsigned char *biases, const float *x, float *y,
                    int out_dim, int in_dim, int group_size, int bits, int scales_type,
                    int row_begin, int row_end) {
    if (row_begin < 0) row_begin = 0;
    if (row_end > out_dim) row_end = out_dim;
    if (row_begin >= row_end) return;

    /* The vector path handles the shapes MLX actually produces (group sizes
     * are powers of two >= 32, so multiples of 8). */
#if SWIFTLET_X86
    if (bits == 4 || bits == 8) {
        static int cached = -1;
        if (cached < 0) cached = __builtin_cpu_supports("avx2") ? 1 : 0;
        if (cached && group_size % 8 == 0 && in_dim % group_size == 0) {
            qgemv_avx2(w, scales, biases, x, y, out_dim, in_dim, group_size,
                       bits, scales_type, row_begin, row_end);
            return;
        }
    }
#endif
    qgemv_scalar(w, scales, biases, x, y, out_dim, in_dim, group_size,
                 bits, scales_type, row_begin, row_end);
}
