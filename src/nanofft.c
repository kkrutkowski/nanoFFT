#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef DOUBLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "./cobra.h"

static inline bool is_power_of_two(int N) {return (N > 0) && ((N & (N - 1)) == 0);}
static inline int intmin(int a, int b) {return (a < b) ? a : b;}


#ifndef DOUBLE // Single precision
    #ifdef __AVX__
        #include <immintrin.h>
        #define VEC __m256
        #define LOAD_VEC _mm256_load_ps
        #define STORE_VEC _mm256_store_ps
        #define ADD_VEC _mm256_add_ps
        #define SUB_VEC _mm256_sub_ps
        #define MUL_VEC _mm256_mul_ps
    #elif defined(__SSE__)
        #include <xmmintrin.h>
        #define VEC __m128
        #define LOAD_VEC _mm_load_ps
        #define STORE_VEC _mm_store_ps
        #define ADD_VEC _mm_add_ps
        #define SUB_VEC _mm_sub_ps
        #define MUL_VEC _mm_mul_ps
    #endif
#else // double precision vectors
    #ifdef __AVX__
        #include <immintrin.h>
        #define VEC __m256d
        #define LOAD_VEC _mm256_load_pd
        #define STORE_VEC _mm256_store_pd
        #define ADD_VEC _mm256_add_pd
        #define SUB_VEC _mm256_sub_pd
        #define MUL_VEC _mm256_mul_pd
    #elif defined(__SSE__)
        #include <xmmintrin.h>
        #define VEC __m128d
        #define LOAD_VEC _mm_load_pd
        #define STORE_VEC _mm_store_pd
        #define ADD_VEC _mm_add_pd
        #define SUB_VEC _mm_sub_pd
        #define MUL_VEC _mm_mul_pd
    #endif
#endif

#ifdef VEC
    #define VEC_LEN (sizeof(VEC) / sizeof(FLOAT))
#endif

#ifndef VEC
void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, int N) {
    int shift = 0;
    for (int step = N; step > 1; step >>= 1) {
        int half_step = step >> 1;
        for (int i = 0; i < N; i += step) {
            for (int j = 0; j < half_step; j++) {
                FLOAT real_even = real_signal[i + j];
                FLOAT imag_even = imag_signal[i + j];
                FLOAT real_odd = real_signal[i + j + half_step];
                FLOAT imag_odd = imag_signal[i + j + half_step];

                real_signal[i + j] = real_even + real_odd;
                imag_signal[i + j] = imag_even + imag_odd;

                FLOAT real_temp = real_even - real_odd;
                FLOAT imag_temp = imag_even - imag_odd;

                real_signal[i + j + half_step] = real_temp * real_buffer[shift + j] - imag_temp * imag_buffer[shift + j];
                imag_signal[i + j + half_step] = real_temp * imag_buffer[shift + j] + imag_temp * real_buffer[shift + j];
            }
        }
        shift += half_step;
    }
}
#else

void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, int N) {
    int shift = 0;
    for (int step = N; step > VEC_LEN; step >>= 1) { // Right bit shift for division by 2
        int half_step = step >> 1; // Right bit shift for division by 2
        for (int i = 0; i < N; i += step) {
            for (int j = 0; j < half_step; j += VEC_LEN) {
                // Load data into VEC variables
                VEC real_even = LOAD_VEC(&real_signal[i + j]);
                VEC imag_even = LOAD_VEC(&imag_signal[i + j]);
                VEC real_odd = LOAD_VEC(&real_signal[i + j + half_step]);
                VEC imag_odd = LOAD_VEC(&imag_signal[i + j + half_step]);

                // Butterfly operation
                STORE_VEC(&real_signal[i + j], ADD_VEC(real_even, real_odd));
                STORE_VEC(&imag_signal[i + j], ADD_VEC(imag_even, imag_odd));

                // Calculate (even - odd) * buffer
                VEC real_temp = SUB_VEC(real_even, real_odd);
                VEC imag_temp = SUB_VEC(imag_even, imag_odd);
                VEC buffer_real = LOAD_VEC(&real_buffer[shift + j]);
                VEC buffer_imag = LOAD_VEC(&imag_buffer[shift + j]);

                // Update real_signal and imag_signal
                STORE_VEC(&real_signal[i + j + half_step],
                SUB_VEC(MUL_VEC(real_temp, buffer_real),MUL_VEC(imag_temp, buffer_imag)));
                STORE_VEC(&imag_signal[i + j + half_step], ADD_VEC(MUL_VEC(real_temp, buffer_imag), MUL_VEC(imag_temp, buffer_real)));
            }
        }
        shift += half_step;
    }
        for (int step = intmin(VEC_LEN, N); step > 1; step >>= 1) { // Right bit shift for division by 2
        int half_step = step >> 1; // Right bit shift for division by 2
        for (int i = 0; i < N; i += step) {
            for (int j = 0; j < half_step; j++) {
                FLOAT real_even = real_signal[i + j];
                FLOAT imag_even = imag_signal[i + j];
                FLOAT real_odd = real_signal[i + j + half_step];
                FLOAT imag_odd = imag_signal[i + j + half_step];

                // Butterfly operation
                real_signal[i + j] = real_even + real_odd;
                imag_signal[i + j] = imag_even + imag_odd;

                // Calculate (even - odd) * buffer
                FLOAT real_temp = real_even - real_odd;
                FLOAT imag_temp = imag_even - imag_odd;
                real_signal[i + j + half_step] = real_temp * real_buffer[shift + j] - imag_temp * imag_buffer[shift + j];
                imag_signal[i + j + half_step] = real_temp * imag_buffer[shift + j] + imag_temp * real_buffer[shift + j];
            }
        }
        shift += half_step;
    }
}
#endif


void generate_buffer(int N, FLOAT *real_buffer, FLOAT *imag_buffer) {
    int shift = 0;
    for (int step = N; step > 1; step >>= 1) {
        int half_step = step >> 1;
        for (int j = 0; j < half_step; j++) {
            FLOAT angle = -2.0 * M_PI * j / step;
            real_buffer[shift + j] = cos(angle);
            imag_buffer[shift + j] = sin(angle);
        }
        shift += half_step;
    }
}

void nanofft_execute(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, int N) {
    if ((N & (N - 1)) != 0) {
        fprintf(stderr, "Signal length must be a power of 2\n");
        exit(EXIT_FAILURE);
    }
    sande_tukey_in_place(real_signal, imag_signal, real_buffer, imag_buffer, N);
    bit_reverse_permutation(real_signal, imag_signal, N);
}
