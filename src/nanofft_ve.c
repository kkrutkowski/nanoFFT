#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef DOUBLE
    #define FLOAT float
    #define FLOAT_BYTES 4
    typedef uint32_t VEC_INT_T;
#else
    #define FLOAT double
    #define FLOAT_BYTES 8
    typedef uint64_t VEC_INT_T;
#endif

#include "./cobra.h"

static inline bool is_power_of_two(int N) {return (N > 0) && ((N & (N - 1)) == 0);}
static inline uint32_t intmin(uint32_t a, uint32_t b) {return (a < b) ? a : b;}
static inline uint32_t intmax(int32_t a, int32_t b) {return (a > b) ? a : b;}

// Configurable vector size (in bytes)
// 64 bytes = 512 bits, equivalent to AVX512
// 32 bytes = 256 bits, equivalent to AVX
// 16 bytes = 128 bits, equivalent to SSE / NEON

#ifndef VEC_BYTES
    #ifdef __AVX512F__
        #define VEC_BYTES 64
    #elif defined(__AVX__)
        #define VEC_BYTES 32
    #else
        #define VEC_BYTES 16
    #endif
#endif

// Now the preprocessor can evaluate this math
#define VEC_LEN (VEC_BYTES / FLOAT_BYTES)

// GNU Vector Extension definitions
typedef FLOAT VEC __attribute__((vector_size(VEC_BYTES)));
typedef VEC_INT_T VEC_INT __attribute__((vector_size(VEC_BYTES)));

// Direct casting for vector loads and stores
#define LOAD_VEC(ptr) (*(const VEC*)(ptr))
#define STORE_VEC(ptr, val) (*(VEC*)(ptr) = (val))

// Enable the intra-vector pass for 8-element vectors (typically float + 256-bit SIMD)
#if VEC_LEN == 8
    #define HAS_INTRA_VEC_PASS

    // Permutation keys for __builtin_shuffle
    static const VEC_INT permutations[3] __attribute__((aligned(64))) = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 4, 5, 2, 3, 6, 7},
        {0, 2, 4, 6, 1, 3, 5, 7}};

    static const VEC_INT inv_permutations[3] __attribute__((aligned(64))) = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 4, 5, 2, 3, 6, 7},
        {0, 4, 1, 5, 2, 6, 3, 7}};

    // Finalization twiddles
    static const VEC real_twiddles[3] __attribute__((aligned(64))) = {
        {(FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2},
        {(FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0},
        {(FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0}};

    static const VEC imag_twiddles[3] __attribute__((aligned(64))) = {
        {(FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2},
        {(FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0},
        {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}};

    static inline void nanofft_vec_shuffle(VEC *a, VEC *b) {
        VEC tmp = __builtin_shuffle(*a, *b, (VEC_INT){0, 1, 2, 3, 8, 9, 10, 11});
        *b      = __builtin_shuffle(*a, *b, (VEC_INT){4, 5, 6, 7, 12, 13, 14, 15});
        *a      = tmp;
    }

    static inline void nanofft_vec_perm(VEC *a, VEC *b, uint32_t idx){
        *a = __builtin_shuffle(*a, permutations[idx]);
        *b = __builtin_shuffle(*b, permutations[idx]);
    }

    static inline void nanofft_vec_inv_perm(VEC *a, VEC *b, uint32_t idx){
        *a = __builtin_shuffle(*a, inv_permutations[idx]);
        *b = __builtin_shuffle(*b, inv_permutations[idx]);
    }
#endif

void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N) {
    uint32_t shift = 0;

    // Vectorized main loop
    for (uint32_t step = N; step > VEC_LEN; step >>= 1) {
        uint32_t half_step = step >> 1;
        for (uint32_t i = 0; i < N; i += step) {
            for (uint32_t j = 0; j < half_step; j += VEC_LEN) {
                VEC r_even = LOAD_VEC(&real_signal[i + j]);
                VEC i_even = LOAD_VEC(&imag_signal[i + j]);
                VEC r_odd  = LOAD_VEC(&real_signal[i + j + half_step]);
                VEC i_odd  = LOAD_VEC(&imag_signal[i + j + half_step]);

                STORE_VEC(&real_signal[i + j], r_even + r_odd);
                STORE_VEC(&imag_signal[i + j], i_even + i_odd);

                VEC r_tmp = r_even - r_odd;
                VEC i_tmp = i_even - i_odd;
                VEC b_real = LOAD_VEC(&real_buffer[shift + j]);
                VEC b_imag = LOAD_VEC(&imag_buffer[shift + j]);

                STORE_VEC(&real_signal[i + j + half_step], (r_tmp * b_real) - (i_tmp * b_imag));
                STORE_VEC(&imag_signal[i + j + half_step], (r_tmp * b_imag) + (i_tmp * b_real));
            }
        }
        shift += half_step;
    }

    #ifdef HAS_INTRA_VEC_PASS
        for (uint32_t i = intmax(0, (int32_t)intlog2(VEC_LEN) - (int32_t)intlog2(N)); i < (uint32_t)intlog2(VEC_LEN); i++) {
            for (uint32_t j = 0; j < N; j += VEC_LEN * 2) {
                VEC r_even = LOAD_VEC(&real_signal[j]);
                VEC r_odd  = LOAD_VEC(&real_signal[j + VEC_LEN]);
                nanofft_vec_perm(&r_even, &r_odd, i);
                nanofft_vec_shuffle(&r_even, &r_odd);

                VEC i_even = LOAD_VEC(&imag_signal[j]);
                VEC i_odd  = LOAD_VEC(&imag_signal[j + VEC_LEN]);
                nanofft_vec_perm(&i_even, &i_odd, i);
                nanofft_vec_shuffle(&i_even, &i_odd);

                VEC r_tmp = r_even - r_odd;
                VEC i_tmp = i_even - i_odd;

                r_even = r_even + r_odd;
                i_even = i_even + i_odd;

                r_odd = (r_tmp * real_twiddles[i]) - (i_tmp * imag_twiddles[i]);
                i_odd = (r_tmp * imag_twiddles[i]) + (i_tmp * real_twiddles[i]);

                nanofft_vec_shuffle(&r_even, &r_odd);
                nanofft_vec_inv_perm(&r_even, &r_odd, i);
                STORE_VEC(&real_signal[j], r_even);
                STORE_VEC(&real_signal[j + VEC_LEN], r_odd);

                nanofft_vec_shuffle(&i_even, &i_odd);
                nanofft_vec_inv_perm(&i_even, &i_odd, i);
                STORE_VEC(&imag_signal[j], i_even);
                STORE_VEC(&imag_signal[j + VEC_LEN], i_odd);
            }
        }
    #else
        for (uint32_t step = intmin(VEC_LEN, N); step > 1; step >>= 1) {
            uint32_t half_step = step >> 1;
            for (uint32_t i = 0; i < N; i += step) {
                for (uint32_t j = 0; j < half_step; j++) {
                    FLOAT re = real_signal[i + j], ie = imag_signal[i + j];
                    FLOAT ro = real_signal[i + j + half_step], io = imag_signal[i + j + half_step];

                    real_signal[i + j] = re + ro;
                    imag_signal[i + j] = ie + io;

                    FLOAT rt = re - ro, it = ie - io;
                    real_signal[i + j + half_step] = rt * real_buffer[shift + j] - it * imag_buffer[shift + j];
                    imag_signal[i + j + half_step] = rt * imag_buffer[shift + j] + it * real_buffer[shift + j];
                }
            }
            shift += half_step;
        }
    #endif
}

void generate_buffer(uint32_t N, FLOAT *real_buffer, FLOAT *imag_buffer) {
    uint32_t shift = 0;
    for (uint32_t step = N; step > 1; step >>= 1) {
        uint32_t half_step = step >> 1;
        for (uint32_t j = 0; j < half_step; j++) {
            FLOAT angle = -2.0 * M_PI * j / step;
            real_buffer[shift + j] = cos(angle);
            imag_buffer[shift + j] = - sin(angle);
        }
        shift += half_step;
    }
}

void nanofft_execute(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N) {
    if (!is_power_of_two(N)) {fprintf(stderr, "Signal length must be a power of 2\n"); exit(EXIT_FAILURE);}
    sande_tukey_in_place(real_signal, imag_signal, real_buffer, imag_buffer, N);
    bit_reverse_permutation(real_signal, imag_signal, N);
}
