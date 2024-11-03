#ifndef COBRA_H
#define COBRA_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define LOG_BLOCK_WIDTH 1  // Example block width (32)
#define BLOCK_WIDTH (1 << LOG_BLOCK_WIDTH)

static inline uint32_t intlog2(uint32_t input){uint32_t output; frexp(input >> 1, (int*) &output); return output;}

// Precomputed lookup table for reversing 8 bits
static const uint8_t bit_reverse_table[256] ={
0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

// Helper function to reverse bits
static inline uint32_t reverse_bits(uint32_t num, uint32_t bits) {
    uint32_t result = 0;
    unsigned int bytes = (bits + 7) >> 3;  // Number of bytes to reverse

    for (unsigned int i = 0; i < bytes; ++i) {
        result <<= 8;
        result |= bit_reverse_table[num & 0xFF];
        num >>= 8;
    }

    // Shift result to align with the bit width
    result >>= (8 * bytes - bits);
    return result;
}

static inline void cobra_apply(FLOAT *real, FLOAT *imag, uint32_t log_n) { //doesn't work'
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
return;}

/* // uncomment when fixed
// COBRA bit-reverse algorithm implementation for separate real and imaginary arrays
static inline void cobra_apply(FLOAT *real, FLOAT *imag, uint32_t log_n) { //doesn't work'
    //if (log_n <= 2 * LOG_BLOCK_WIDTH) {
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
*/

// Function to perform bit-reverse permutation on separate real and imaginary arrays
inline static void bit_reverse_permutation(FLOAT *real, FLOAT *imag, uint32_t N) {cobra_apply(real, imag, intlog2(N));}

#endif
