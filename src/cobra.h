#ifndef COBRA_H
#define COBRA_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define LOG_BLOCK_WIDTH 7  // Example block width (32)
#define BLOCK_WIDTH (1 << LOG_BLOCK_WIDTH)

// Helper function to reverse bits
static inline uint32_t reverse_bits(uint32_t num, unsigned int bits) {
    uint32_t result = 0;
    for (unsigned int i = 0; i < bits; ++i) {
        result <<= 1;
        result |= (num & 1);
        num >>= 1;
    }
    return result;
}

// COBRA bit-reverse algorithm implementation for separate real and imaginary arrays
static inline void cobra_apply(FLOAT *real, FLOAT *imag, int log_n) {
    if (log_n <= 2 * LOG_BLOCK_WIDTH) {
        // Fallback to a simpler bit-reversal if log_n is small
        for (int i = 0; i < (1 << log_n); ++i) {
            int j = reverse_bits(i, log_n);
            if (j > i) {
                // Swap real parts
                FLOAT temp_real = real[i];
                FLOAT temp_imag = imag[i];

                real[i] = real[j];
                imag[i] = imag[j];

                real[j] = temp_real;
                imag[j] = temp_imag;
            }
        }
        return;
    }

    int num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
    size_t b_size = 1 << num_b_bits;
    size_t block_width = 1 << LOG_BLOCK_WIDTH;

    FLOAT *buffer_real = (FLOAT *)aligned_alloc(64, BLOCK_WIDTH * BLOCK_WIDTH * sizeof(FLOAT));
    FLOAT *buffer_imag = (FLOAT *)aligned_alloc(64, BLOCK_WIDTH * BLOCK_WIDTH * sizeof(FLOAT));

    for (size_t b = 0; b < b_size; b++) {
        size_t b_rev = reverse_bits(b, num_b_bits) >> ((b_size - 1) - __builtin_clz(b_size - 1));

        // Copy block to buffer
        for (size_t a = 0; a < block_width; a++) {
            size_t a_rev = reverse_bits(a, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));
            for (size_t c = 0; c < BLOCK_WIDTH; c++) {
                size_t idx = (a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c;
                size_t buffer_idx = (a_rev << LOG_BLOCK_WIDTH) | c;

                buffer_real[buffer_idx] = real[idx];
                buffer_imag[buffer_idx] = imag[idx];
            }
        }

        for (size_t c = 0; c < BLOCK_WIDTH; c++) {
            size_t c_rev = reverse_bits(c, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));

            for (size_t a_rev = 0; a_rev < BLOCK_WIDTH; a_rev++) {
                size_t a = reverse_bits(a_rev, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));

                // Check the condition to swap
                int index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if (index_less_than_reverse) {
                    size_t v_idx = (c_rev << num_b_bits << LOG_BLOCK_WIDTH) | (b_rev << LOG_BLOCK_WIDTH) | a_rev;
                    size_t b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;

                    // Swap real parts
                    FLOAT temp_real = real[v_idx];
                    FLOAT temp_imag = imag[v_idx];
                    real[v_idx] = buffer_real[b_idx];
                    imag[v_idx] = buffer_imag[b_idx];
                    buffer_real[b_idx] = temp_real;
                    buffer_imag[b_idx] = temp_imag;
                }
            }
        }

        // Copy changes that were swapped into buffer
        for (size_t a = 0; a < BLOCK_WIDTH; a++) {
            size_t a_rev = reverse_bits(a, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));
            for (size_t c = 0; c < BLOCK_WIDTH; c++) {
                size_t c_rev = reverse_bits(c, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));
                int index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if (index_less_than_reverse) {
                    size_t v_idx = (a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c;
                    size_t b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;

                    // Swap real parts
                    FLOAT temp_real = real[v_idx];
                    FLOAT temp_imag = imag[v_idx];
                    real[v_idx] = buffer_real[b_idx];
                    imag[v_idx] = buffer_imag[b_idx];
                    buffer_real[b_idx] = temp_real;
                    buffer_imag[b_idx] = temp_imag;
                }
            }
        }
    }

    free(buffer_real);
    free(buffer_imag);
}

// Function to perform bit-reverse permutation on separate real and imaginary arrays
void bit_reverse_permutation(FLOAT *real, FLOAT *imag, int N) {
    int bits;
    frexp(N >> 1, &bits);
    cobra_apply(real, imag, bits);
}

#endif
