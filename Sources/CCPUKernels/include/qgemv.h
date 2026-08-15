#ifndef SWIFTLET_QGEMV_H
#define SWIFTLET_QGEMV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MLX affine quantized GEMV, the CPU port of the Metal `gemv_affine` kernel:
 *
 *   y[row] = sum over groups g of
 *              scales[row,g] * sum(q_i * x_i)  +  biases[row,g] * sum(x_i)
 *
 * with w[i] = scales[row,g] * q[i] + biases[row,g], q unpacked from uint32
 * words packed along the input dim (8x4-bit or 4x8-bit per word), one
 * scale+bias pair per `group_size` consecutive inputs.
 *
 * All weight pointers are BYTE addresses (tensors sit at arbitrary offsets
 * inside container blobs), exactly like the Metal binding.
 *
 *   w, scales, biases : byte pointers into the expert blob's sections
 *   x                 : input vector, `in_dim` floats
 *   y                 : output, `out_dim` floats (written for [row_begin,row_end))
 *   scales_type       : 0 = f32, 1 = f16, 2 = bf16
 *   bits              : 4 or 8
 *
 * Rows [row_begin, row_end) let callers parallelize across threads; a call
 * with 0, out_dim covers the whole matrix. Uses AVX2 when the CPU and the
 * shapes allow (group_size multiple of 8), scalar C otherwise.
 */
void swiftlet_qgemv(const unsigned char *w,
                    const unsigned char *scales,
                    const unsigned char *biases,
                    const float *x,
                    float *y,
                    int out_dim, int in_dim, int group_size, int bits, int scales_type,
                    int row_begin, int row_end);

#ifdef __cplusplus
}
#endif

#endif /* SWIFTLET_QGEMV_H */
