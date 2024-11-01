#ifndef COBRA_H
#define COBRA_H

#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#define LOG_BLOCK_WIDTH 7  // Example block width (128)
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

// COBRA bit-reverse algorithm implementation
void cobra_apply(complex FLOAT *v, int log_n) {
    if (log_n <= 2 * LOG_BLOCK_WIDTH) {
        // Fallback to a simpler bit-reversal if log_n is small
        for (int i = 0; i < (1 << log_n); ++i) {
            int j = reverse_bits(i, log_n);
            if (j > i) {
                complex FLOAT temp = v[i];
                v[i] = v[j];
                v[j] = temp;
            }
        }
        return;
    }

    int num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
    size_t b_size = 1 << num_b_bits;
    size_t block_width = 1 << LOG_BLOCK_WIDTH;

    //complex FLOAT buffer[BLOCK_WIDTH * BLOCK_WIDTH] = {0};
    complex FLOAT *buffer = (complex FLOAT *) calloc(BLOCK_WIDTH * BLOCK_WIDTH, sizeof(complex FLOAT));

    for (size_t b = 0; b < b_size; b++) {
        size_t b_rev = reverse_bits(b, num_b_bits) >> ((b_size - 1) - __builtin_clz(b_size - 1));

        // Copy block to buffer
        for (size_t a = 0; a < block_width; a++) {
            size_t a_rev = reverse_bits(a, LOG_BLOCK_WIDTH) >> ((block_width - 1) - __builtin_clz(block_width - 1));
            for (size_t c = 0; c < BLOCK_WIDTH; c++) {
                buffer[(a_rev << LOG_BLOCK_WIDTH) | c] =
                    v[(a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c];
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
                    complex FLOAT temp = v[v_idx];
                    v[v_idx] = buffer[b_idx];
                    buffer[b_idx] = temp;
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
                    complex FLOAT temp = v[v_idx];
                    v[v_idx] = buffer[b_idx];
                    buffer[b_idx] = temp;
                }
            }
        }
    }
}

// Function to perform bit-reverse permutation on the signal
void bit_reverse_permutation(complex FLOAT *signal, int N) {
    int bits;
    frexp(N, &bits);
    bits -= 1;
    cobra_apply(signal, bits);
}

#endif
