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
static inline uint32_t intmin(uint32_t a, uint32_t b) {return (a < b) ? a : b;}
static inline uint32_t intmax(int32_t a, int32_t b) {return (a > b) ? a : b;}

#ifndef DOUBLE // Single precision
    #ifdef __AVX512F__
        #include <immintrin.h>
        #define VEC __m512
        #define LOAD_VEC _mm512_load_ps
        #define STORE_VEC _mm512_store_ps
        #define ADD_VEC _mm512_add_ps
        #define SUB_VEC _mm512_sub_ps
        #define MUL_VEC _mm512_mul_ps
    #elif defined(__AVX__)
        #include <immintrin.h>
        #define VEC __m256
        #define LOAD_VEC _mm256_load_ps
        #define STORE_VEC _mm256_store_ps
        #define ADD_VEC _mm256_add_ps
        #define SUB_VEC _mm256_sub_ps
        #define MUL_VEC _mm256_mul_ps
            #ifdef __AVX2__
                typedef union {__m256i m256i; uint32_t i[8];} m256i_union;
                typedef union {__m256 vec; float f[8];} m256_union;

                // Permutation keys
                static const m256i_union permutations[3] __attribute__((aligned(64))) = {
                    {.i = {0, 1, 2, 3, 4, 5, 6, 7}},
                    {.i = {0, 1, 4, 5, 2, 3, 6, 7}},
                    {.i = {0, 2, 4, 6, 1, 3, 5, 7}}}; //doesn't invert itself
                static const m256i_union inv_permutations[3] __attribute__((aligned(64))) = {
                    {.i = {0, 1, 2, 3, 4, 5, 6, 7}},
                    {.i = {0, 1, 4, 5, 2, 3, 6, 7}},
                    {.i = {0, 4, 1, 5, 2, 6, 3, 7}}};

                // Finalization twiddles
                static const m256_union real_twiddles[3] __attribute__((aligned(64))) = {
                    {.f = {1.0f, M_SQRT1_2, 0.0f, -M_SQRT1_2, 1.0f, M_SQRT1_2, 0.0f, -M_SQRT1_2}},
                    {.f = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}},
                    {.f = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}}};
                static const m256_union imag_twiddles[3] __attribute__((aligned(64))) = {
                    {.f = {0.0f, M_SQRT1_2, 1.0f, M_SQRT1_2, 0.0f, M_SQRT1_2, 1.0f, M_SQRT1_2}},
                    {.f = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}},
                    {.f = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};

                static inline void nanofft_mm256_shuffle(__m256 *a, __m256 *b) {
                    __m256 tmp = _mm256_permute2f128_ps(*a, *b, 0x20);  // 0x20 selects lower lane of `a` and lower lane of `b`
                    *b = _mm256_permute2f128_ps(*a, *b, 0x31);          // 0x31 selects upper lane of `a` and upper lane of `b`
                    *a = tmp;
                }
                static inline void nanofft_mm256_perm(__m256 *a, __m256 *b, uint32_t idx){
                    *a = _mm256_permutevar8x32_ps(*a, permutations[idx].m256i);
                    *b = _mm256_permutevar8x32_ps(*b, permutations[idx].m256i);}
                static inline void nanofft_mm256_inv_perm(__m256 *a, __m256 *b, uint32_t idx){
                    *a = _mm256_permutevar8x32_ps(*a, inv_permutations[idx].m256i);
                    *b = _mm256_permutevar8x32_ps(*b, inv_permutations[idx].m256i);}

                #define RTWIDDLES real_twiddles
                #define ITWIDDLES imag_twiddles
                #define SHUFFLE_VEC nanofft_mm256_shuffle
                #define PERM_VEC nanofft_mm256_perm
                #define INVPERM_VEC nanofft_mm256_inv_perm
            #endif
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
    #ifdef __AVX512F__
        #include <immintrin.h>
        #define VEC __m512d
        #define LOAD_VEC _mm512_load_pd
        #define STORE_VEC _mm512_store_pd
        #define ADD_VEC _mm512_add_pd
        #define SUB_VEC _mm512_sub_pd
        #define MUL_VEC _mm512_mul_pd
    #elif defined(__AVX__)
        #include <immintrin.h>
        #define VEC __m256d
        #define LOAD_VEC _mm256_load_pd
        #define STORE_VEC _mm256_store_pd
        #define ADD_VEC _mm256_add_pd
        #define SUB_VEC _mm256_sub_pd
        #define MUL_VEC _mm256_mul_pd
            #ifdef __AVX2__
                #define LOADPERM_VEC _mm256_perm_pd
            #endif
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

// #undef VEC //used to test the generic version, comment for release version

#ifdef VEC
    #define VEC_LEN (sizeof(VEC) / sizeof(FLOAT))
#else
    #define VEC_LEN UINT32_MAX
#endif

void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N) {
    uint32_t shift = 0;
    #ifdef VEC //vectorized main loop of the FFT
    for (uint32_t step = N; step > VEC_LEN; step >>= 1) { // Right bit shift for division by 2
        uint32_t half_step = step >> 1; // Right bit shift for division by 2
        for (uint32_t i = 0; i < N; i += step) {
            for (uint32_t j = 0; j < half_step; j += VEC_LEN) {
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
                STORE_VEC(&real_signal[i + j + half_step], SUB_VEC(MUL_VEC(real_temp, buffer_real),MUL_VEC(imag_temp, buffer_imag)));
                STORE_VEC(&imag_signal[i + j + half_step], ADD_VEC(MUL_VEC(real_temp, buffer_imag), MUL_VEC(imag_temp, buffer_real)));
            }
        }
        shift += half_step;
    } //*
    #endif
    #ifdef SHUFFLE_VEC
        for (uint32_t i = intmax(0, (int32_t)intlog2(VEC_LEN) - (int32_t)intlog2(N)); i < intlog2(VEC_LEN); i+= 1) {
            for (uint32_t j = 0; j < N; j+= VEC_LEN * 2) {
                // Load data into VEC variables
                VEC real_even = LOAD_VEC(&real_signal[j]);
                VEC real_odd = LOAD_VEC(&real_signal[j + VEC_LEN]);
                PERM_VEC(&real_even, &real_odd, i);
                SHUFFLE_VEC(&real_even, &real_odd);
                VEC imag_even = LOAD_VEC(&imag_signal[j]);
                VEC imag_odd = LOAD_VEC(&imag_signal[j + VEC_LEN]);
                PERM_VEC(&imag_even, &imag_odd, i);
                SHUFFLE_VEC(&imag_even, &imag_odd);

                VEC real_temp = SUB_VEC(real_even, real_odd);
                VEC imag_temp = SUB_VEC(imag_even, imag_odd);

                // Butterfly operation
                real_even = ADD_VEC(real_even, real_odd);
                imag_even = ADD_VEC(imag_even, imag_odd);

                real_odd = SUB_VEC(MUL_VEC(real_temp, RTWIDDLES[i].vec), MUL_VEC(imag_temp, ITWIDDLES[i].vec));
                imag_odd = ADD_VEC(MUL_VEC(real_temp, ITWIDDLES[i].vec), MUL_VEC(imag_temp, RTWIDDLES[i].vec));

                //resture vectors to original permutation for next iteration

                SHUFFLE_VEC(&real_even, &real_odd);
                INVPERM_VEC(&real_even, &real_odd, i);
                STORE_VEC(&real_signal[j], real_even);
                STORE_VEC(&real_signal[j + VEC_LEN], real_odd);

                SHUFFLE_VEC(&imag_even, &imag_odd);
                INVPERM_VEC(&imag_even, &imag_odd, i);
                STORE_VEC(&imag_signal[j], imag_even);
                STORE_VEC(&imag_signal[j + VEC_LEN], imag_odd);
            } //for (uint32_t j = 0; j < N; j+= 1) {printf("%.1f %.1f \t", real_signal[j], imag_signal[j]);} printf("\n"); //debug printf
        }
    #else
        for (uint32_t step = intmin(VEC_LEN, N); step > 1; step >>= 1) { // Required addition of SIMD secondary loop to reach reasonable performance levels
        uint32_t half_step = step >> 1;
        for (uint32_t i = 0; i < N; i += step) {
            for (uint32_t j = 0; j < half_step; j++) {
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
    if ((N & (N - 1)) != 0) {fprintf(stderr, "Signal length must be a power of 2\n"); exit(EXIT_FAILURE);}
    sande_tukey_in_place(real_signal, imag_signal, real_buffer, imag_buffer, N);
    bit_reverse_permutation(real_signal, imag_signal, N);
}
